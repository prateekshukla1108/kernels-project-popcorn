#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 4  

__global__ void softmax(float *input, float *output, float *sum, int n) {
    int idx = threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx]);
    }

    // Make sure all threads finished writing before sum computation
    __syncthreads();

    // Thread 0 computes the sum of exponentials in global memory
    if (idx == 0) {
        *sum = 0.0f;
        for (int i = 0; i < n; i++) {
            *sum += output[i];
        }
    }


    __syncthreads();


    if (idx < n) {
        output[idx] /= *sum;
    }
}

int main() {
    float h_input[N] = {1.0, 2.0, 3.0, 4.0};
    float h_output[N], h_sum;

    float *d_input, *d_output, *d_sum;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax<<<1, N>>>(d_input, d_output, d_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Softmax Output:\n";
    float ksum = 0;
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
        ksum = ksum +  h_output[i];
        std::cout<<ksum<<"\n";
    }


    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sum);

    return 0;
}

