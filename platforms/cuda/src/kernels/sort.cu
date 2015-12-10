__device__ KEY_TYPE getValue(DATA_TYPE value) {
    return SORT_KEY;
}

extern "C" {

/**
 * Sort a list that is short enough to entirely fit in local memory.  This is executed as
 * a single thread block.
 */
__global__ void sortShortList(DATA_TYPE* __restrict__ data, int length, int bufferSize) {
    // Load the data into local memory.
    
    extern __shared__ DATA_TYPE dataBuffer[];
    for (int index = threadIdx.x; index < bufferSize; index += blockDim.x)
        dataBuffer[index] = (index < length ? data[index] : MAX_VALUE);
    __syncthreads();

    // Perform a bitonic sort in local memory.

    for (int k = 2; k < 2*bufferSize; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for (int i = threadIdx.x; i < bufferSize; i += blockDim.x) {
                int ixj = i^j;
                if (ixj > i) {
                    DATA_TYPE value1 = dataBuffer[i];
                    DATA_TYPE value2 = dataBuffer[ixj];
                    bool ascending = ((i&k) == 0);
                    KEY_TYPE lowKey  = (ascending ? getValue(value1) : getValue(value2));
                    KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                    if (lowKey > highKey) {
                        dataBuffer[i] = value2;
                        dataBuffer[ixj] = value1;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write the data back to global memory.

    for (int index = threadIdx.x; index < length; index += blockDim.x)
        data[index] = dataBuffer[index];
}

/**
 * Calculate the minimum and maximum value in the array to be sorted.  This kernel
 * is executed as a single work group.
 */
__global__ void computeRange(const DATA_TYPE* __restrict__ data, unsigned int length, KEY_TYPE* __restrict__ range,
        unsigned int numBuckets, unsigned int* __restrict__ bucketOffset) {
    extern __shared__ KEY_TYPE minBuffer[];
    KEY_TYPE* maxBuffer = minBuffer+blockDim.x;
    KEY_TYPE minimum = MAX_KEY;
    KEY_TYPE maximum = MIN_KEY;

    // Each thread calculates the range of a subset of values.

    for (unsigned int index = threadIdx.x; index < length; index += blockDim.x) {
        KEY_TYPE value = getValue(data[index]);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    // Now reduce them.

    minBuffer[threadIdx.x] = minimum;
    maxBuffer[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x+step < blockDim.x && threadIdx.x%(2*step) == 0) {
            minBuffer[threadIdx.x] = min(minBuffer[threadIdx.x], minBuffer[threadIdx.x+step]);
            maxBuffer[threadIdx.x] = max(maxBuffer[threadIdx.x], maxBuffer[threadIdx.x+step]);
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        range[0] = minBuffer[0];
        range[1] = maxBuffer[0];
    }
    
    // Clear the bucket counters in preparation for the next kernel.

    for (unsigned int index = threadIdx.x; index < numBuckets; index += blockDim.x)
        bucketOffset[index] = 0;
}

/**
 * Assign elements to buckets.
 */
__global__ void assignElementsToBuckets(const DATA_TYPE* __restrict__ data, unsigned int length, unsigned int numBuckets, const KEY_TYPE* __restrict__ range,
        unsigned int* bucketOffset, unsigned int* __restrict__ bucketOfElement, unsigned int* __restrict__ offsetInBucket) {
    float minValue = (float) (range[0]);
    float maxValue = (float) (range[1]);
    float bucketWidth = (maxValue-minValue)/numBuckets;
    for (unsigned int index = blockDim.x*blockIdx.x+threadIdx.x; index < length; index += blockDim.x*gridDim.x) {
        float key = (float) getValue(data[index]);
        unsigned int bucketIndex = min((unsigned int) ((key-minValue)/bucketWidth), numBuckets-1);
        offsetInBucket[index] = atomicAdd(&bucketOffset[bucketIndex], 1);
        bucketOfElement[index] = bucketIndex;
    }
}

/**
 * Sum the bucket sizes to compute the start position of each bucket.  This kernel
 * is executed as a single work group.
 */
__global__ void computeBucketPositions(unsigned int numBuckets, unsigned int* __restrict__ bucketOffset) {
    extern __shared__ unsigned int posBuffer[];
    unsigned int globalOffset = 0;
    for (unsigned int startBucket = 0; startBucket < numBuckets; startBucket += blockDim.x) {
        // Load the bucket sizes into local memory.

        unsigned int globalIndex = startBucket+threadIdx.x;
        posBuffer[threadIdx.x] = (globalIndex < numBuckets ? bucketOffset[globalIndex] : 0);
        __syncthreads();

        // Perform a parallel prefix sum.

        for (unsigned int step = 1; step < blockDim.x; step *= 2) {
            unsigned int add = (threadIdx.x >= step ? posBuffer[threadIdx.x-step] : 0);
            __syncthreads();
            posBuffer[threadIdx.x] += add;
            __syncthreads();
        }

        // Write the results back to global memory.

        if (globalIndex < numBuckets)
            bucketOffset[globalIndex] = posBuffer[threadIdx.x]+globalOffset;
        globalOffset += posBuffer[blockDim.x-1];
    }
}

/**
 * Copy the input data into the buckets for sorting.
 */
__global__ void copyDataToBuckets(const DATA_TYPE* __restrict__ data, DATA_TYPE* __restrict__ buckets, unsigned int length, const unsigned int* __restrict__ bucketOffset, const unsigned int* __restrict__ bucketOfElement, const unsigned int* __restrict__ offsetInBucket) {
    for (unsigned int index = blockDim.x*blockIdx.x+threadIdx.x; index < length; index += blockDim.x*gridDim.x) {
        DATA_TYPE element = data[index];
        unsigned int bucketIndex = bucketOfElement[index];
        unsigned int offset = (bucketIndex == 0 ? 0 : bucketOffset[bucketIndex-1]);
        buckets[offset+offsetInBucket[index]] = element;
    }
}

/**
 * This is called by mergeBlocks().  It identifies the starting point for the piece of the merge a particular thread will do.
 */
__device__ int findPositionOnDiagonal(DATA_TYPE* block1, DATA_TYPE* block2, int length1, int length2) {
    int diagonalIndex = threadIdx.x*(length1+length2)/blockDim.x;
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
__device__ void mergeBlocks(DATA_TYPE* block1, DATA_TYPE* block2, int length1, int length2, DATA_TYPE* result) {
    int diagonalIndex = threadIdx.x*(length1+length2)/blockDim.x;
    int pathLength = (threadIdx.x+1)*(length1+length2)/blockDim.x-diagonalIndex;
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
__global__ void sortBuckets(DATA_TYPE* __restrict__ data, DATA_TYPE* __restrict__ buckets, unsigned int numBuckets, const unsigned int* __restrict__ bucketOffset) {
    extern __shared__ DATA_TYPE dataBuffer[];
    for (unsigned int index = blockIdx.x; index < numBuckets; index += gridDim.x) {
        unsigned int startIndex = (index == 0 ? 0 : bucketOffset[index-1]);
        unsigned int endIndex = bucketOffset[index];
        unsigned int length = endIndex-startIndex;
        
        // First divide the bucket into blocks that are small enough to sort in local memory.
        
        for (int blockStart = 0; blockStart < length; blockStart += blockDim.x) {
            // Load the data into local memory.

            __syncthreads();
            if (threadIdx.x < length-blockStart)
                dataBuffer[threadIdx.x] = buckets[startIndex+blockStart+threadIdx.x];
            else
                dataBuffer[threadIdx.x] = MAX_VALUE;
            __syncthreads();

            // Perform a bitonic sort in local memory.

            for (unsigned int k = 2; k <= blockDim.x; k *= 2) {
                for (unsigned int j = k/2; j > 0; j /= 2) {
                    int ixj = threadIdx.x^j;
                    if (ixj > threadIdx.x) {
                        DATA_TYPE value1 = dataBuffer[threadIdx.x];
                        DATA_TYPE value2 = dataBuffer[ixj];
                        bool ascending = (threadIdx.x&k) == 0;
                        KEY_TYPE lowKey = (ascending ? getValue(value1) : getValue(value2));
                        KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                        if (lowKey > highKey) {
                            dataBuffer[threadIdx.x] = value2;
                            dataBuffer[ixj] = value1;
                        }
                    }
                    __syncthreads();
                }
            }

            // Write the data to the sorted array.

            if (threadIdx.x+blockStart < length)
                data[startIndex+blockStart+threadIdx.x] = dataBuffer[threadIdx.x];
        }
        if (length <= blockDim.x)
            continue;
        
        // Now merge blocks together.
        
        __syncthreads();
        DATA_TYPE* from = data;
        DATA_TYPE* to = buckets;
        for (int mergeSize = blockDim.x; mergeSize < length; mergeSize *= 2) {
            // Merge pairs of blocks of length mergeSize into single blocks of size 2*mergeSize.
            
            int blockStart = 0;
            for (; blockStart+mergeSize < length; blockStart += 2*mergeSize)
                mergeBlocks(from+startIndex+blockStart, from+startIndex+blockStart+mergeSize, mergeSize, min(mergeSize, length-blockStart-mergeSize), to+startIndex+blockStart);
            
            // If there's an additional single block at the end, copy it over.
            
            for (int i = blockStart+threadIdx.x; i < length; i += blockDim.x)
                to[startIndex+i] = from[startIndex+i];
            __syncthreads();
            
            // Swap the to and from arrays.
            
            DATA_TYPE* temp = from;
            from = to;
            to = temp;
        }
        
        // If the last merge wrote to the buckets array, copy it to the output array.
        
        if (to == data) {
            for (int i = threadIdx.x; i < length; i += blockDim.x)
                data[startIndex+i] = buckets[startIndex+i];
        }
    }
}

}
