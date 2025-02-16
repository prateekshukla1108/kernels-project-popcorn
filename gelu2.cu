#include <iostream>
#include <cuda_runtime.h>

#define N 10   // Size of the array

__global__ void gelu(float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = 0.5*data[i]*(1.0+erff(data[i]/sqrt(2.0))); // GeLU function
    }
}

int main() {
    float h_input[N] = {-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 5.0, -4.0, 3.0, -0.5};
    // float h_output[N] = {0};  // Initialize output array

    float *d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    // cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with dynamic thread and block calculations
    int threadsPerBlock = 256;  // Standard thread block size
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Round up to cover all elements
    gelu<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sigmoid output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    // cudaFree(d_output);

    return 0;
}
