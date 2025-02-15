#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define N 8

__global__ void naive_sum_reduction(int *input, int *output)
{
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
        {
            if ((i + stride) < N)
                input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *output = input[0];
    }
}

int main()
{
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8}; // Example input
    int *d_input, *d_output;
    int h_output;

    // Allocate memory on the device
    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_output, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 4 threads (N/2 threads, assuming N=8)
    naive_sum_reduction<<<1, 8>>>(d_input, d_output);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Sum: " << h_output << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}