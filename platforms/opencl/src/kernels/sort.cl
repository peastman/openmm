#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

KEY_TYPE getValue(DATA_TYPE value) {
    return SORT_KEY;
}

/**
 * Sort a list that is short enough to entirely fit in local memory.  This is executed as
 * a single thread block.
 */
__kernel void sortShortList(__global DATA_TYPE* __restrict__ data, uint length, __local DATA_TYPE* dataBuffer) {
    // Load the data into local memory.
    
    for (int index = get_local_id(0); index < length; index += get_local_size(0))
        dataBuffer[index] = data[index];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform a bitonic sort in local memory.

    for (unsigned int k = 2; k < 2*length; k *= 2) {
        for (unsigned int j = k/2; j > 0; j /= 2) {
            for (unsigned int i = get_local_id(0); i < length; i += get_local_size(0)) {
                int ixj = i^j;
                if (ixj > i && ixj < length) {
                    DATA_TYPE value1 = dataBuffer[i];
                    DATA_TYPE value2 = dataBuffer[ixj];
                    bool ascending = ((i&k) == 0);
                    for (unsigned int mask = k*2; mask < 2*length; mask *= 2)
                        ascending = ((i&mask) == 0 ? !ascending : ascending);
                    KEY_TYPE lowKey  = (ascending ? getValue(value1) : getValue(value2));
                    KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                    if (lowKey > highKey) {
                        dataBuffer[i] = value2;
                        dataBuffer[ixj] = value1;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write the data back to global memory.

    for (int index = get_local_id(0); index < length; index += get_local_size(0))
        data[index] = dataBuffer[index];
}

/**
 * Calculate the minimum and maximum value in the array to be sorted.  This kernel
 * is executed as a single work group.
 */
__kernel void computeRange(__global const DATA_TYPE* restrict data, uint length, __global KEY_TYPE* restrict range, __local KEY_TYPE* restrict minBuffer,
        __local KEY_TYPE* restrict maxBuffer, uint numBuckets, __global uint* restrict bucketOffset) {
    KEY_TYPE minimum = MAX_KEY;
    KEY_TYPE maximum = MIN_KEY;

    // Each thread calculates the range of a subset of values.

    for (uint index = get_local_id(0); index < length; index += get_local_size(0)) {
        KEY_TYPE value = getValue(data[index]);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    // Now reduce them.

    minBuffer[get_local_id(0)] = minimum;
    maxBuffer[get_local_id(0)] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = 1; step < get_local_size(0); step *= 2) {
        if (get_local_id(0)+step < get_local_size(0) && get_local_id(0)%(2*step) == 0) {
            minBuffer[get_local_id(0)] = min(minBuffer[get_local_id(0)], minBuffer[get_local_id(0)+step]);
            maxBuffer[get_local_id(0)] = max(maxBuffer[get_local_id(0)], maxBuffer[get_local_id(0)+step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0) {
        range[0] = minBuffer[0];
        range[1] = maxBuffer[0];
    }
    
    // Clear the bucket counters in preparation for the next kernel.

    for (uint index = get_local_id(0); index < numBuckets; index += get_local_size(0))
        bucketOffset[index] = 0;
}

/**
 * Assign elements to buckets.
 */
__kernel void assignElementsToBuckets(__global const DATA_TYPE* restrict data, uint length, uint numBuckets, __global const KEY_TYPE* restrict range,
        __global uint* bucketOffset, __global uint* restrict bucketOfElement, __global uint* restrict offsetInBucket) {
#ifdef AMD_ATOMIC_WORK_AROUND
    // Do a byte write to force all memory accesses to interactionCount to use the complete path.
    // This avoids the atomic access from causing all word accesses to other buffers from using the slow complete path.
    // The IF actually causes the write to never be executed, its presence is all that is needed.
    // AMD APP SDK 2.4 has this problem.
    if (get_global_id(0) == get_local_id(0)+1)
        ((__global char*)bucketOffset)[sizeof(int)*numBuckets+1] = 0;
#endif
    float minValue = (float) (range[0]);
    float maxValue = (float) (range[1]);
    float bucketWidth = (maxValue-minValue)/numBuckets;
    for (uint index = get_global_id(0); index < length; index += get_global_size(0)) {
        float key = (float) getValue(data[index]);
        uint bucketIndex = min((uint) ((key-minValue)/bucketWidth), numBuckets-1);
        offsetInBucket[index] = atom_inc(&bucketOffset[bucketIndex]);
        bucketOfElement[index] = bucketIndex;
    }
}

/**
 * Sum the bucket sizes to compute the start position of each bucket.  This kernel
 * is executed as a single work group.
 */
__kernel void computeBucketPositions(uint numBuckets, __global uint* restrict bucketOffset, __local uint* restrict buffer) {
    uint globalOffset = 0;
    for (uint startBucket = 0; startBucket < numBuckets; startBucket += get_local_size(0)) {
        // Load the bucket sizes into local memory.

        uint globalIndex = startBucket+get_local_id(0);
        buffer[get_local_id(0)] = (globalIndex < numBuckets ? bucketOffset[globalIndex] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform a parallel prefix sum.

        for (uint step = 1; step < get_local_size(0); step *= 2) {
            uint add = (get_local_id(0) >= step ? buffer[get_local_id(0)-step] : 0);
            barrier(CLK_LOCAL_MEM_FENCE);
            buffer[get_local_id(0)] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write the results back to global memory.

        if (globalIndex < numBuckets)
            bucketOffset[globalIndex] = buffer[get_local_id(0)]+globalOffset;
        globalOffset += buffer[get_local_size(0)-1];
    }
}

/**
 * Copy the input data into the buckets for sorting.
 */
__kernel void copyDataToBuckets(__global const DATA_TYPE* restrict data, __global DATA_TYPE* restrict buckets, uint length, __global const uint* restrict bucketOffset, __global const uint* restrict bucketOfElement, __global const uint* restrict offsetInBucket) {
    for (uint index = get_global_id(0); index < length; index += get_global_size(0)) {
        DATA_TYPE element = data[index];
        uint bucketIndex = bucketOfElement[index];
        uint offset = (bucketIndex == 0 ? 0 : bucketOffset[bucketIndex-1]);
        buckets[offset+offsetInBucket[index]] = element;
    }
}

/**
 * This is called by mergeBlocks().  It identifies the starting point for the piece of the merge a particular thread will do.
 */
int findPositionOnDiagonal(__global DATA_TYPE* block1, __global DATA_TYPE* block2, int length1, int length2) {
    int diagonalIndex = get_local_id(0)*(length1+length2)/get_local_size(0);
    int searchMin = max(0, diagonalIndex-length1); // Minimum element from block2
    int searchMax = min(length2, diagonalIndex); // Maximum element from block2
    while (searchMin < searchMax) {
        int index2 = (searchMin+searchMax)/2;
        int index1 = diagonalIndex-index2;
        if (index1 == 0 || getValue(block1[index1-1]) > getValue(block2[index2]))
            searchMin = index2+1;
        else
            searchMax = index2;
    }
    return searchMin;
}

/**
 * Merge two sorted blocks together.  This is called by sortBuckets().
 */
void mergeBlocks(__global DATA_TYPE* block1, __global DATA_TYPE* block2, int length1, int length2, __global DATA_TYPE* result) {
    int diagonalIndex = get_local_id(0)*(length1+length2)/get_local_size(0);
    int pathLength = (get_local_id(0)+1)*(length1+length2)/get_local_size(0)-diagonalIndex;
    int index2 = findPositionOnDiagonal(block1, block2, length1, length2);
    int index1 = diagonalIndex-index2;
    for (int i = 0; i < pathLength; i++) {
        DATA_TYPE nextValue;
        if (index1 == length1)
            nextValue = block2[index2++];
        else if (index2 == length2)
            nextValue = block1[index1++];
        else if (getValue(block1[index1]) > getValue(block2[index2]))
            nextValue = block2[index2++];
        else
            nextValue = block1[index1++];
        result[index1+index2-1] = nextValue;
    }

    __syncthreads();
}

/**
 * Sort the data in each bucket.
 */
__kernel void sortBuckets(__global DATA_TYPE* restrict data, __global DATA_TYPE* restrict buckets, uint numBuckets, __global const uint* restrict bucketOffset, __local DATA_TYPE* restrict dataBuffer) {
    for (int index = get_group_id(0); index < numBuckets; index += get_num_groups(0)) {
        int startIndex = (index == 0 ? 0 : bucketOffset[index-1]);
        int endIndex = bucketOffset[index];
        int length = endIndex-startIndex;
        
        // First divide the bucket into blocks that are small enough to sort in local memory.
        
        for (int blockStart = 0; blockStart < length; blockStart += get_local_size(0)) {
            // Load the data into local memory.

            barrier(CLK_LOCAL_MEM_FENCE);;
            if (get_local_id(0) < length-blockStart)
                dataBuffer[get_local_id(0)] = buckets[startIndex+blockStart+get_local_id(0)];
            else
                dataBuffer[get_local_id(0)] = MAX_VALUE;
            barrier(CLK_LOCAL_MEM_FENCE);;

            // Perform a bitonic sort in local memory.

            for (int k = 2; k <= get_local_size(0); k *= 2) {
                for (int j = k/2; j > 0; j /= 2) {
                    int ixj = get_local_id(0)^j;
                    if (ixj > get_local_id(0)) {
                        DATA_TYPE value1 = dataBuffer[get_local_id(0)];
                        DATA_TYPE value2 = dataBuffer[ixj];
                        bool ascending = (get_local_id(0)&k) == 0;
                        KEY_TYPE lowKey = (ascending ? getValue(value1) : getValue(value2));
                        KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                        if (lowKey > highKey) {
                            dataBuffer[get_local_id(0)] = value2;
                            dataBuffer[ixj] = value1;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);;
                }
            }

            // Write the data to the sorted array.

            if (get_local_id(0)+blockStart < length)
                data[startIndex+blockStart+get_local_id(0)] = dataBuffer[get_local_id(0)];
        }
        if (length <= get_local_size(0))
            continue;
        
        // Now merge blocks together.
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        __global DATA_TYPE* from = data;
        __global DATA_TYPE* to = buckets;
        for (int mergeSize = get_local_size(0); mergeSize < length; mergeSize *= 2) {
            // Merge pairs of blocks of length mergeSize into single blocks of size 2*mergeSize.
            
            int blockStart = 0;
            for (; blockStart+mergeSize < length; blockStart += 2*mergeSize)
                mergeBlocks(from+startIndex+blockStart, from+startIndex+blockStart+mergeSize, mergeSize, min(mergeSize, length-blockStart-mergeSize), to+startIndex+blockStart);
            
            // If there's an additional single block at the end, copy it over.
            
            for (int i = blockStart+get_local_id(0); i < length; i += get_local_size(0))
                to[startIndex+i] = from[startIndex+i];
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            // Swap the to and from arrays.
            
            __global DATA_TYPE* temp = from;
            from = to;
            to = temp;
        }
        
        // If the last merge wrote to the buckets array, copy it to the output array.
        
        if (to == data) {
            for (int i = get_local_id(0); i < length; i += get_local_size(0))
                data[startIndex+i] = buckets[startIndex+i];
        }
    }
}
