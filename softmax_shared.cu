#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 4  

__global__ void softmax(float *input, float *output, int n) {
    __shared__ float sum;

    int idx = threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }

    __syncthreads();
    if (idx == 0) {
        sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += output[i];
        }
    }

    __syncthreads();

    if (idx < n) {
        output[idx] /= sum;
    }
}

int main() {
    float h_input[N] = {1.0, 2.0, 3.0, 4.0};
    float h_output[N];

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax<<<1, N>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Softmax Output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

