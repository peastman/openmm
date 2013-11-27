#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

/**
 * Compute nonbonded interactions.
 */

 __kernel void computeNonbonded(
#ifdef SUPPORTS_64_BIT_ATOMICS
        __global long* restrict forceBuffers,
#else
        __global real4* restrict forceBuffers,
#endif
        __global real* restrict energyBuffer, __global const real4* restrict posq, __global const unsigned int* restrict exclusions,
        __global const ushort2* restrict exclusionTiles, unsigned int startTileIndex, unsigned int numTileIndices
#ifdef USE_CUTOFF
        , __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        unsigned int maxTiles, __global const real4* restrict blockCenter, __global const real4* restrict blockSize, __global const int* restrict interactingAtoms
#endif
        PARAMETER_ARGUMENTS) {
    const int totalWarps = get_global_size(0)/TILE_SIZE;
    const int warp = get_global_id(0)/TILE_SIZE;
    const int tgx = get_local_id(0);
    real energy = 0;

    // First loop: process tiles that contain exclusions.
    
    const int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const int x = tileIndices.x;
        const int y = tileIndices.y;
        const int tileIdxX = x*TILE_SIZE;
        const int tileIdxY = y*TILE_SIZE;

        // Load the data for this tile.
        
        const bool hasExclusions = true;
        real4 force = 0;
        int atom1 = tileIdxX+tgx;
        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS
#ifdef USE_EXCLUSIONS
        unsigned int excl = exclusions[pos*TILE_SIZE+tgx];
        excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
        for (int j = 0; j < TILE_SIZE; j++) {
            int tj = (j+tgx) & (TILE_SIZE - 1);
            int atom2 = tileIdxY+tj;
            real4 posq2 = posq[atom2];
            real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
            delta.xyz -= floor(delta.xyz*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;
#endif
            real r2 = dot(delta.xyz, delta.xyz);
#ifdef USE_CUTOFF
            if (r2 < CUTOFF_SQUARED) {
#endif
                LOAD_ATOM2_PARAMETERS
                real invR = RSQRT(r2);
                real r = RECIP(invR);
#ifdef USE_SYMMETRIC
                real dEdR = 0;
#else
                real4 dEdR1 = (real4) 0;
                real4 dEdR2 = (real4) 0;
#endif
#ifdef USE_EXCLUSIONS
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
#endif
                real tempEnergy = 0;
                COMPUTE_INTERACTION
                if (x == y)
                    energy += 0.5f*tempEnergy;
                else
                    energy += tempEnergy;
#ifdef USE_SYMMETRIC
                delta.xyz *= dEdR;
                force.xyz -= delta.xyz;
                if (x != y) {
                    int offset = atom2 + warp*PADDED_NUM_ATOMS;
                    forceBuffers[offset].xyz += delta.xyz;
                }
#else
                force.xyz -= dEdR1.xyz;
                if (x != y) {
                    int offset = atom2 + warp*PADDED_NUM_ATOMS;
                    forceBuffers[offset] += dEdR2.xyz;
                }
#endif
#ifdef USE_CUTOFF
            }
#endif
#ifdef USE_EXCLUSIONS
            excl >>= 1;
#endif
        }
        
        // Write results.
        
#ifdef SUPPORTS_64_BIT_ATOMICS
        atom_add(&forceBuffers[atom1], (long) (force.x*0x100000000));
        atom_add(&forceBuffers[atom1+PADDED_NUM_ATOMS], (long) (force.y*0x100000000));
        atom_add(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (long) (force.z*0x100000000));
#else
        int offset = atom1 + warp*PADDED_NUM_ATOMS;
        forceBuffers[offset].xyz += force.xyz;
#endif
    }
    SYNC_WARPS;
    
    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    const int numTiles = interactionCount[0];
    int pos = (numTiles > maxTiles ? startTileIndex+warp*numTileIndices/totalWarps : warp*numTiles/totalWarps);
    int end = (numTiles > maxTiles ? startTileIndex+(warp+1)*numTileIndices/totalWarps : (warp+1)*numTiles/totalWarps);
#else
    const int numTiles = numTileIndices;
    int pos = startTileIndex+warp*numTiles/totalWarps;
    int end = startTileIndex+(warp+1)*numTiles/totalWarps;
#endif
    int nextToSkip = -1;
    int currentSkipIndex = 0;

    while (pos < end) {
        const bool hasExclusions = false;
        bool includeTile = true;
        
        // Extract the coordinates of this tile.
        
        int x, y;
#ifdef USE_CUTOFF
        if (numTiles <= maxTiles)
            x = tiles[pos];
        else
#endif
        {
            y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            }

            // Skip over tiles that have exclusions, since they were already processed.
            
            while (nextToSkip < pos) {
                if (currentSkipIndex < NUM_TILES_WITH_EXCLUSIONS) {
                    ushort2 tile = exclusionTiles[currentSkipIndex++];
                    nextToSkip = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
                }
                else
                    nextToSkip = end;
            }
            includeTile = (nextToSkip != pos);
        }
        const int tileIdxY = y*TILE_SIZE;
        const int tileIdxPos = pos*TILE_SIZE;
        if (includeTile) {
            // Load the data for this tile.
            
            int atom1 = x*TILE_SIZE+tgx;
            real4 force = 0;
            real4 posq1 = posq[atom1];
#ifdef USE_CUTOFF
            real4 blockCenterX = blockCenter[x];
#endif
            LOAD_ATOM1_PARAMETERS
            for (int j = 0; j < TILE_SIZE; j++) {
                int tj = (j+tgx) & (TILE_SIZE - 1);
                int atom2 = tileIdxY+tj;
#ifdef USE_CUTOFF
                atom2 = (numTiles <= maxTiles ? interactingAtoms[tileIdxPos+tj] : atom2);
#endif
                real4 posq2 = posq[atom2];
                real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                delta.xyz -= floor(delta.xyz*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;
#endif
                real r2 = dot(delta.xyz, delta.xyz);
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = RECIP(invR);
                    LOAD_ATOM2_PARAMETERS
#ifdef USE_SYMMETRIC
                    real dEdR = 0;
#else
                    real4 dEdR1 = (real4) 0;
                    real4 dEdR2 = (real4) 0;
#endif
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                    real tempEnergy = 0;
                    COMPUTE_INTERACTION
                    energy += tempEnergy;
                    int offset = atom2 + warp*PADDED_NUM_ATOMS;
                    real4 f = forceBuffers[offset];
#ifdef USE_SYMMETRIC
                    delta.xyz *= dEdR;
                    force.xyz -= delta.xyz;
                    f.xyz += delta.xyz;
#else
                    force.xyz -= dEdR1.xyz;
                    f.xyz += dEdR2.xyz;
#endif
                    forceBuffers[offset] = f;
#ifdef USE_CUTOFF
                }
#endif
            }

            // Write results for atom1.

#ifdef SUPPORTS_64_BIT_ATOMICS
            atom_add(&forceBuffers[atom1], (long) (force.x*0x100000000));
            atom_add(&forceBuffers[atom1+PADDED_NUM_ATOMS], (long) (force.y*0x100000000));
            atom_add(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (long) (force.z*0x100000000));
#else
            int offset = atom1 + warp*PADDED_NUM_ATOMS;
            forceBuffers[offset].xyz += force.xyz;
#endif
        }
        pos++;
    }
    energyBuffer[get_global_id(0)] += energy;
}