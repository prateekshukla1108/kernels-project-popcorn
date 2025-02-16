#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N 4  // Number of features per input
#define EPS 1e-5

__global__ void layerNormKernel(float* d_input, float* d_output, float* d_gamma, float* d_beta) {
    __shared__ float mean, var;
    
    int tid = threadIdx.x;
    
    // Step 1: Compute Mean
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += d_input[i];
    }
    mean = sum / N;
    
    // Step 2: Compute Variance
    float var_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        var_sum += (d_input[i] - mean) * (d_input[i] - mean);
    }
    var = var_sum / N;
    
    // Step 3: Normalize + Scale + Shift
    d_output[tid] = d_gamma[tid] * ((d_input[tid] - mean) / sqrtf(var + EPS)) + d_beta[tid];
}

int main() {
    float h_input[N] = {1.0, 2.0, 3.0, 4.0};
    float h_output[N], h_gamma[N] = {1.0, 1.0, 1.0, 1.0}, h_beta[N] = {0.0, 0.0, 0.0, 0.0};
    
    float *d_input, *d_output, *d_gamma, *d_beta;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_gamma, N * sizeof(float));
    cudaMalloc(&d_beta, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, N * sizeof(float), cudaMemcpyHostToDevice);
    
    layerNormKernel<<<1, N>>>(d_input, d_output, d_gamma, d_beta);
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