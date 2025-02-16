#include <iostream>
#include <cuda_runtime.h>

#define N 10   

__global__ void relu(float *input, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);  //ReLU function
    }
}

int main() {
    float h_input[N] = {-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 5.0, -4.0, 3.0, -0.5};
    float h_output[N];

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    relu<<<1, N>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "ReLU :\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

