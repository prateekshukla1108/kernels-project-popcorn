#include "timer.h"

__host__ __device__ float f(float x, float y){
    return x + y;
}

void vecadd_cpu(float *h_A, float *h_B, float *h_C, unsigned int N){
    for(unsigned int i = 0; i < N; ++i){
        h_C[i] = f(h_A[i], h_B[i]);
    }
}

// Kernel function that will be executed on the GPU
// Keyword __global__ is used to define a kernel function
__global__ void vecadd_kernel (float* h_A, float* h_B, float* h_C, unsigned int N){
    /* uses special keywords like threadIdx, blockIdx, blockDim, gridDim to distinguish between threads in a grid
    threadIdx: unique identifier for each thread in a block
    blockIdx: unique identifier for each block in a grid
    blockDim: number of threads per block
    gridDim: number of blocks in a grid
    */
    // We need to get the global index of the thread
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // (Size of block) * (Index of block) + (Index of thread)
    // We need to check if the thread is within the bounds of the blocks
    // We effectively ignore the threads that are out of bounds; See README.md for more details>
    if(i < N){
        h_C[i] = f(h_A[i], h_B[i]);
    }
}

// N represents the number of threads in the grid
void vecadd_gpu(float* h_A, float* h_B, float* h_C, unsigned int N){

    // We receive the pointers to the data on the host (CPU)

    //Allocate GPU Memory
    float *d_A, *d_B, *d_C;

    // cudaMalloc returns cudeError_t type that's why we are not doing d_A = cudaMalloc(N*sizeof(float));
    cudaMalloc((void**)&d_A, N*sizeof(float)); // cudaMalloc expects a void pointer to a pointer
    cudaMalloc((void**)&d_B, N*sizeof(float));
    cudaMalloc((void**)&d_C, N*sizeof(float));

    //Copy data from host to device
    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice); // destination, source, size, direction
    cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);

    //Perform computation on GPU i.e. call a GPU kernel function (launch a grid of threads)
    const unsigned int threads_per_block = 256; // Maximum number of threads per block is 1024
    // const unsigned int num_blocks = N / threads_per_block; // Assuming N is divisible by threads_per_block
    const unsigned int num_blocks = (N + threads_per_block - 1) / threads_per_block; // To handle cases where N is not divisible by threads_per_block
    Timer timer;
    startTIme(&timer);
    vecadd_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N); // blocks and how many threads per block
    cudaDeviceSynchronize(); // Wait for all threads in the grid to finish
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel time", GREEN);

    //Copy data from Device(GPU) to host (CPU)
    cudaMemcpy(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

    //Free GPU Memory
    cudaFree(d_A); // cudaFree expects a void pointer
    cudaFree(d_B);
    cudaFree(d_C);
}

int main( int argc, char**argv){
    cudaDeviceSynchronize();

    //Allocate memory on host (CPU) and initialize data
    Timer timer;
    // If the user provides a command line argument, use it as the size of the array, otherwise use 2^25
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1<<25);
    float *h_A = (float*)malloc(N*sizeof(float));
    float *h_B = (float*)malloc(N*sizeof(float));
    float *h_C = (float*)malloc(N*sizeof(float));

    //Initialize data
    for(unsigned int i = 0; i < N; i++){
        h_A[i] = rand();
        h_B[i] = rand();
    }

    // Vector addition on host (CPU)
    startTIme(&timer);
    vecadd_cpu(h_A, h_B, h_C, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Vector addition on GPU
    startTIme(&timer);
    vecadd_gpu(h_A, h_B, h_C, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN );




}


