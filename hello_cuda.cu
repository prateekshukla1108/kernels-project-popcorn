#include <stdio.h>

// CUDA Kernel
__global__ void helloCUDA() {
    printf("Hello, CUDA from GPU!\n");
}

int main() {
    // Launch the kernel with 1 block of 1 thread
    helloCUDA<<<1, 1>>>();

    // Synchronize to ensure printf execution
    cudaDeviceSynchronize();

    return 0;
}
