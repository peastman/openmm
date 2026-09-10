/**
 * Copy 4-byte values to a reordered array.
 */
KERNEL void reorderValues4(GLOBAL int* RESTRICT original, GLOBAL int* RESTRICT reordered, GLOBAL int* RESTRICT atomOrder, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        reordered[i] = original[atomOrder[i]];
}

/**
 * Copy 8-byte values to a reordered array.
 */
KERNEL void reorderValues8(GLOBAL int2* RESTRICT original, GLOBAL int2* RESTRICT reordered, GLOBAL int* RESTRICT atomOrder, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        reordered[i] = original[atomOrder[i]];
}

/**
 * Copy 16-byte values to a reordered array.
 */
KERNEL void reorderValues16(GLOBAL int4* RESTRICT original, GLOBAL int4* RESTRICT reordered, GLOBAL int* RESTRICT atomOrder, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        reordered[i] = original[atomOrder[i]];
}

#ifdef SUPPORTS_DOUBLE_PRECISION
/**
 * Copy 32-byte values to a reordered array.
 */
KERNEL void reorderValues32(GLOBAL double4* RESTRICT original, GLOBAL double4* RESTRICT reordered, GLOBAL int* RESTRICT atomOrder, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        reordered[i] = original[atomOrder[i]];
}
#endif