#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello, World from thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}