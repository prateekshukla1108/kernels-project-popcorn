#include <iostream>
#include <cuda_runtime.h>

#define N 10   // Size of the array
#define LEAKY_SLOPE 0.01f  // Slope for negative values in Leaky ReLU

__global__ void leaky_relu(float *input, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : LEAKY_SLOPE * input[idx];  
    }
}

int main() {
    float h_input[N] = {-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 5.0, -4.0, 3.0, -0.5};
    float h_output[N] = {0};  // Initialize output array

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with dynamic thread and block calculations
    int threadsPerBlock = 256;  // Standard thread block size
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Round up to cover all elements
    leaky_relu<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Leaky ReLU output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
