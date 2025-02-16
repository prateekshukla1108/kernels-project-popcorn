#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define BLOCK_SIZE 256

__global__ void reduceSumKernel(float *d_input, float *d_output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < N) ? d_input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (optional)
    // if (tid < 32) {
    //     for (int s = 16; s > 0; s >>= 1) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    // }

    // Write result to output
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // host memory
    float *h_input;
    float *h_output;


    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, sizeof(float));

    // Init with random
    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand() % 10 + 1); 
    }

    // device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // temporary storage for block sums
    float *d_blockSums;
    cudaMalloc(&d_blockSums, gridSize * sizeof(float));

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Reduce within blocks
    reduceSumKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_blockSums, N);

    // Reduce block sums
    reduceSumKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_blockSums, d_output, gridSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU reduction time: " << elapsedTime << " ms" << endl;

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Sum: " << h_output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    // dont forget to destroy the events too
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
