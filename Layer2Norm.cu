#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 10  
#define EPSILON 1e-5f

__global__ void layer_norm(float *input, float *output, float *gamma, float *beta, int n) {
    int idx = threadIdx.x;

    __shared__ float mean;
    if (idx == 0) {
        mean = 0.0f;
        for (int i = 0; i < n; i++) {
            mean += input[i];
        }
        mean /= n;
    }
    __syncthreads();

    __shared__ float variance;
    if (idx == 0) {
        variance = 0.0f;
        for (int i = 0; i < n; i++) {
            variance += (input[i] - mean) * (input[i] - mean);
        }
        variance /= n;
    }
    __syncthreads();

    if (idx < n) {
        output[idx] = ((input[idx] - mean) / sqrtf(variance + EPSILON)) * gamma[idx] + beta[idx];
    }
}

int main() {
    float h_input[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float h_output[N], h_gamma[N], h_beta[N];

    for (int i = 0; i < N; i++) {
        h_gamma[i] = 1.0f;  // No scaling
        h_beta[i] = 0.0f;   // No shifting
    }

    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_gamma, N * sizeof(float));
    cudaMalloc(&d_beta, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with one block of N threads
    layer_norm<<<1, N>>>(d_input, d_output, d_gamma, d_beta, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "LayerNorm Output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}

