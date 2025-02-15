#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

using namespace std;
#define BUFFER_LENGTH 100000000

__global__ void navie_kernel(int* buffer, int* output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while(i<BUFFER_LENGTH){
        atomicAdd(&output[buffer[i]], 1);
        i+=stride;
    }
}

__global__ void optimized_kernel(int* buffer, int* output){
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while(i<BUFFER_LENGTH){
        atomicAdd(&temp[buffer[i]],1);
        i+=stride;
    }
    __syncthreads();
    atomicAdd(&output[threadIdx.x], temp[threadIdx.x]);
}


int main(){
    ////////////////////////////////////////
    ///  Generate data for benchmarking ////
    ////////////////////////////////////////
    int* buffer = (int*)malloc(BUFFER_LENGTH * sizeof(int));
    int* counter = (int*)malloc(256 * sizeof(int));

    srand(time(NULL));
    for(int i=0; i<BUFFER_LENGTH;i++){
        int n = rand() % 256;
        buffer[i] = n;
    }

    ////////////////////////////////////////
    ///////// Benchmark CPU code ///////////
    ////////////////////////////////////////

    struct timespec begin, terminate;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    for(int i=0; i<BUFFER_LENGTH;i++){
            int n = rand() % 256;
            buffer[i] = n;
            counter[n]++;
        }

    clock_gettime(CLOCK_MONOTONIC, &terminate);
    float elapsed_time = (terminate.tv_sec - begin.tv_sec) +
                          (terminate.tv_nsec - begin.tv_nsec) / 1e9;
    printf("Execution time for CPU code: %f seconds\n", elapsed_time);


    ////////////////////////////////////////
    ///// Naive GPU implementation  ///////
    ////////////////////////////////////////

    //Start Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 );
    int blocks = prop.multiProcessorCount;
    cout<<"Total number of blocks in GPU "<<blocks<<endl;


    int *d_buffer, *d_output, *h_naive_output;
    h_naive_output =  (int*)malloc(256 * sizeof(int));

    cudaMalloc((void**)&d_buffer, BUFFER_LENGTH * sizeof(int));
    cudaMemcpy(d_buffer, buffer,  BUFFER_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output, 256 * sizeof(int));
    cudaMemset(d_output, 0, 256 * sizeof(int));

    navie_kernel<<<2*blocks, 256>>>(d_buffer, d_output);
    cudaMemcpy(h_naive_output, d_output, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check the correctness of program
    for(int i=0; i<256; i++) assert(h_naive_output[i]==counter[i]);

    cudaFree(d_buffer);
    cudaFree(d_output);
    free(h_naive_output);

    printf("Execution time for navie GPU code: %f seconds\n", elapsed_time/1000);
    /////////////////////////////////////
    /// Optimized GPU implementation  ///
    /////////////////////////////////////

    //Start Timer
    //cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //int *d_buffer, *d_output, *h_naive_output;
    h_naive_output =  (int*)malloc(256 * sizeof(int));

    cudaMalloc((void**)&d_buffer, BUFFER_LENGTH * sizeof(int));
    cudaMemcpy(d_buffer, buffer,  BUFFER_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output, 256 * sizeof(int));
    cudaMemset(d_output, 0, 256 * sizeof(int));

    optimized_kernel<<<2*blocks, 256>>>(d_buffer, d_output);
    cudaMemcpy(h_naive_output, d_output, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check the correctness of program
    for(int i=0; i<256; i++) assert(h_naive_output[i]==counter[i]);

    cudaFree(d_buffer);
    cudaFree(d_output);
    free(h_naive_output);
    printf("Execution time for optimized GPU code: %f seconds\n", elapsed_time/1000);


    free(buffer);
    free(counter);
    return 0;
}


/*
Execution time for CPU code: 2.565612 seconds
Total number of blocks in GPU 68
Execution time for navie GPU code: 0.147478 seconds
Execution time for optimized GPU code: 0.125348 seconds
*/