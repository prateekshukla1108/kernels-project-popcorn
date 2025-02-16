#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

__global__ void reduction_gpu(float * input, float * output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int str=1; str < blockDim.x;str*=2){
        if( idx % (2*str) == 0 and idx + str < size){
            input[idx] += input[idx + str];
        }
        __syncthreads();
    }
    if (idx % blockDim.x == 0 and idx < size){
    output[blockIdx.x] = input[idx];
    }
}
float reduction_cpu(float * input, int size){
    float output = 0.0f;
    for (int i=0;i<size;i++){
        output += input[i];
    }
    return output;
}
int main() {
    // Initialize input and output arrays
    int size = 100000;
    float* input = new float[size];
    ;
    // float* gpu_output = new float[size];

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    cout<<"Max threads per block: " << prop.maxThreadsPerBlock << "\n"<< "Shared mem per block: " << prop.sharedMemPerBlock << " bytes\n";

    // Fill input array with 1.0f for simplicity
    for (int i = 0; i < size; ++i) {
        input[i] = rand();
    }
    // cout << "Expected Result : "<<(size * (size+1))/2. << endl;

    
    // Run CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_output = reduction_cpu(input, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;
    cout << "CPU result: " << cpu_output << endl;


    // for (int i = 0; i < size; ++i) {
    //     cout << input[i] << " ";
    // }
    // Allocate memory on the GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    // cudaMalloc((void**)&d_output, size * sizeof(float));

    // // Copy input data to the GPU
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // // Define block and grid sizes for CUDA
    int numThreads = 1024;
    int numBlocks = (size + numThreads - 1)/numThreads;
    // The space on cpu
    float* gpu_output = new float[numBlocks];

    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));
    // Run GPU implementation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    reduction_gpu<<<numBlocks, numThreads>>>(d_input, d_output, size);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    // Copy output data back to the host
    cudaMemcpy(gpu_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_res = reduction_cpu(gpu_output, numBlocks);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;
    cout << "GPU result: " << gpu_res << endl;
    // Copy output data back to the host
    // cudaMemcpy(gpu_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    // if (compare_results(cpu_output, gpu_output, size)) {
    //     std::cout << "CPU and GPU results match!" << std::endl;
    // } else {
    //     std::cerr << "CPU and GPU results do not match!" << std::endl;
    // }

    cout << endl;
    // cout<<size<<endl;

    // Free GPU memory

    // cudaFree(d_input);
    // cudaFree(d_output);

    // Free CPU memory
    delete[] input;
    // delete[] gpu_output;

    return 0;
}