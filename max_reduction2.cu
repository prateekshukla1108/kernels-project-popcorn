#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define N 1073741824
#define BLOCK_DIM 1024

using namespace std;

__global__ void max_reduction(int *input, int *output)
{
    __shared__ int input_s[BLOCK_DIM];
    int segment = 2 * BLOCK_DIM * blockIdx.x;
    int t = threadIdx.x;
    int i = t + segment;
    input_s[t] = MAX(input[i], input[i + BLOCK_DIM]);
    for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
        {
            input_s[t] = MAX(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0)
    {
        atomicMax(output, input_s[0]);
    }
}

int main(void)
{
    int *h_input;
    cudaMallocHost(&h_input, N * sizeof(int));
    int max_element = -pow(2, 31);
    for (int i = 0; i < N; i++)
    {
        h_input[i] = rand();
        max_element = MAX(max_element, h_input[i]);
    }

    // Kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &max_element, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    max_reduction<<<blocks, 1024>>>(d_input, d_output);

    int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    // Stop total timer
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Print timing results
    cout << "\nTiming Results:" << endl;
    cout << "Kernel execution time: " << kernel_time << " ms" << endl;

    assert(h_output == max_element);
}