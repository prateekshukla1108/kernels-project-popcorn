#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define THREADS_PER_BLOCK 1024
#define COARSE_FACTOR  4

__global__ void reduction_gpu(float *input, float *output, int size) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x*COARSE_FACTOR  + threadIdx.x;
    float temp = 0.0f;
    // Load data into shared memory
    for (int i =0; i< COARSE_FACTOR; i++){
        int new_idx = idx + i * blockDim.x;
        if(new_idx < size){
        temp += input[new_idx];
        }
        
    }
    shared_data[tid] = temp;
    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Store block result in global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

float reduction_cpu(float *input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
}

int main() {
    int size = 1 <<25;
    float *input = new float[size];

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n"
         << "Shared mem per block: " << prop.sharedMemPerBlock << " bytes\n";

    // Initialize input array with random values
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU Reduction
    auto start_cpu = chrono::high_resolution_clock::now();
    float cpu_output = reduction_cpu(input, size);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    cout << "CPU time: " << cpu_duration.count() << " seconds" << endl;
    cout << "CPU result: " << cpu_output << endl;

    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));

    // Define grid and block sizes
    int numThreads = THREADS_PER_BLOCK;
    int numBlocks = (size + numThreads*COARSE_FACTOR - 1) / (COARSE_FACTOR*numThreads);
    float *gpu_output = new float[numBlocks];

    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU reduction
    auto start_gpu = chrono::high_resolution_clock::now();
    reduction_gpu<<<numBlocks, numThreads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    // Copy partial results back to CPU and finish reduction
    cudaMemcpy(gpu_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_result = reduction_cpu(gpu_output, numBlocks);
    
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_duration = end_gpu - start_gpu;

    cout << "GPU time: " << gpu_duration.count() << " seconds" << endl;
    cout << "GPU result: " << gpu_result << endl;

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free CPU memory
    delete[] input;
    delete[] gpu_output;

    return 0;
}
