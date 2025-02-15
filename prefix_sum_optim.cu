#include <stdlib.h>
#include <iostream>
#include <assert.h>
#define BLOCK_DIM 1024

using namespace std;

__global__ void prefix_sum(int *nums, int n, int *block_prefix) {
    __shared__ int nums_s[BLOCK_DIM];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (i >= n) return;
    nums_s[t] = nums[i];
    __syncthreads();

    // Perform inclusive scan within the block
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        int temp = 0;
        if (t >= stride) {
            temp = nums_s[t - stride];
        }
        __syncthreads();
        nums_s[t] += temp;
        __syncthreads();
    }

    nums[i] = nums_s[t];
    if (t == BLOCK_DIM - 1) {
        block_prefix[blockIdx.x] = nums_s[t];
    }
}

__global__ void prefix_sum_add_offset(int *nums, int n, int *block_prefix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || blockIdx.x == 0) return;
    nums[i] += block_prefix[blockIdx.x - 1];
}

int main(void) {
    int n = 2048;
    int *h_nums, *h_output;
    h_nums = (int *)malloc(n * sizeof(int));
    h_output = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_nums[i] = rand() % 1000;
    }

    int *d_nums, *d_block_prefix, *h_block_prefix;
    int m = n / BLOCK_DIM; // Number of blocks
    h_block_prefix = (int *)malloc(m * sizeof(int));

    cudaMalloc((void **)&d_nums, n * sizeof(int));
    cudaMalloc((void **)&d_block_prefix, m * sizeof(int));

    cudaMemcpy(d_nums, h_nums, n * sizeof(int), cudaMemcpyHostToDevice);

    // Compute block-level prefix sums
    prefix_sum<<<m, BLOCK_DIM>>>(d_nums, n, d_block_prefix);

    // Copy block sums to host and compute their prefix sums
    cudaMemcpy(h_block_prefix, d_block_prefix, m * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute inclusive prefix sum of block sums
    for (int i = 1; i < m; i++) {
        h_block_prefix[i] += h_block_prefix[i - 1];
    }

    // Copy block prefixes back to device
    cudaMemcpy(d_block_prefix, h_block_prefix, m * sizeof(int), cudaMemcpyHostToDevice);

    // Add block offsets to each element
    prefix_sum_add_offset<<<m, BLOCK_DIM>>>(d_nums, n, d_block_prefix);

    // Copy result back to host and verify
    cudaMemcpy(h_output, d_nums, n * sizeof(int), cudaMemcpyDeviceToHost);

    int temp = 0;
    for (int i = 0; i < n; i++) {
        temp += h_nums[i];
        assert(temp == h_output[i]);
    }

    free(h_nums);
    free(h_output);
    free(h_block_prefix);
    cudaFree(d_nums);
    cudaFree(d_block_prefix);

    return 0;
}