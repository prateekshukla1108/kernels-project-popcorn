#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to normalize an array
__global__ void normalizeKernel(float *d_out, float sum, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[idx] /= sum;  // Normalize each element
    }
}

int main() {
    const int N = 5;  
    float h_out[N] = {2.0, 4.0, 6.0, 8.0, 10.0}; // Example input
    float h_sum = 30.0;  // Example sum

    float *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMemcpy(d_out, h_out, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    normalizeKernel<<<numBlocks, blockSize>>>(d_out, h_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Normalized Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_out);
    return 0;
}

