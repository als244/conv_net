#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// #pragma unroll

#define CHUNK_SIZE 64

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024


int main(void)
{

	// NEED TO LOAD WEIGHTS + BIASES FROM SAVED MODEL

	// NEED TO LOAD TEST DATA AND OUTPUT PREDICTIONS...

}