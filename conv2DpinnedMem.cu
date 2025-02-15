#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono> // Added for timing
using namespace std;

#define n 2
__constant__ float d_kernel[2 * n + 1][2 * n + 1];

__global__ void conv2D(float *input, int k, float *output)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    float value = 0;
    for (int i = -n; i <= n; i++)
    {
        for (int j = -n; j <= n; j++)
        {
            if (0 <= x + i && x + i < k && 0 <= y + j && y + j < k)
                value += input[k * (y + j) + x + i] * d_kernel[n + i][n + j];
        }
    }
    output[k * y + x] = value;
}

int main(void)
{

    // kernel declaration
    int kernel_dim = 2 * n + 1;
    int kernel_size = kernel_dim * kernel_dim * sizeof(float);
    float h_kernel[kernel_dim][kernel_dim];
    for (int i = 0; i < kernel_dim; i++)
    {
        for (int j = 0; j < kernel_dim; j++)
        {
            h_kernel[i][j] = 5 - (abs(2 - i) + abs(2 - j));
        }
    }

    // input declaration
    int k = 32;
    int input_size = k * k * sizeof(float);
    float *h_input;
    cudaMallocHost(&h_input, input_size);

    
    // float h_input[k][k];
    for (int r = 0; r < k; r++)
    {
        int temp = r + 1;
        for (int c = 0; c < k; c++)
        {
            h_input[r*k+c] = temp + c;
        }
    }
    

    /*
    cout<<"kernel"<<endl;
    for(int i=0;i<kernel_dim;i++){
        for(int j = 0; j<kernel_dim;j++){
            cout<<h_kernel[i][j]<<" ";
        }
         cout<<endl;
    }
    cout<<"input"<<endl;
    for(int r = 0; r < k; r++){
        for(int c = 0; c < k; c++){
            cout<<h_input[r][c]<<" ";
        }
         cout<<endl;
    }
    */

    // Start total timer
    auto total_start = chrono::high_resolution_clock::now();

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, input_size);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size);

    dim3 block_size(k, k);

    // Kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2D<<<1, block_size>>>(d_input, k, d_output);

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    // Stop total timer
    auto total_end = chrono::high_resolution_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    float h_output[input_size];
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

    // Calculate timings
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Free the memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end - total_start).count();

    // Print timing results
    cout << "\nTiming Results:" << endl;
    cout << "Kernel execution time: " << kernel_time << " ms" << endl;
    cout << "Total time (including memory transfers): "
         << total_duration / 1000.0 << " ms" << endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*
    cout<<"Output"<<endl;
    for(int i=0;i<k;i++){
        for(int j = 0; j<k;j++){
            cout<<h_output[k*i+j]<<" ";
        }
        cout<<endl;
    }
    */

    return 0;
}