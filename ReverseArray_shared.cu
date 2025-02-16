#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Array size

// CUDA kernel to reverse an array in place
__global__ void reverseKernel(int *d_arr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mirror_idx = size - 1 - idx;

    if (idx < mirror_idx) {
        // Swap elements
        int temp = d_arr[idx];
        d_arr[idx] = d_arr[mirror_idx];
        d_arr[mirror_idx] = temp;
    }
}

int main() {
    int h_arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // Original array
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    reverseKernel<<<numBlocks, blockSize>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Reversed Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}

