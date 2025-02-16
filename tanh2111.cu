#include <iostream>
#include <cuda_runtime.h>
#include <math.h>  

#define N 10   

__global__ void tanh_activation(float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float x = input[idx];
        input[idx] = (expf(x) - expf(-x)) / (expf(x) + expf(-x));  // Tanh function
    }
}

int main() {
    // Host array
    float h_input[N] = {-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 5.0, -4.0, 3.0, -0.5};

    // Device pointer
    float *d_input;
    cudaMalloc(&d_input, N * sizeof(float));  // Allocate device memory

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with dynamic thread and block calculations
    int threadsPerBlock = 256;  // Standard thread block size
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Round up to cover all elements
    tanh_activation<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Tanh output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);

    return 0;
}
