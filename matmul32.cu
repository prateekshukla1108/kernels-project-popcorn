#include<iostream>
#include"timer.h"

__global__ void gpu_mul(int *a, int *b, int *c, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N){
        for(int k = 0; k < N; k++){
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}

void cpu_mul(int *a, int *b, int *c, int N){
    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            for(int k = 0; k < N; k++){
                c[row * N + col] += a[row * N + k] * b[k * N + col];
            }
        }
    }
}

int main(){
    int N = 1024;
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];
    int *c_cpu = new int[N * N]; // Add c_cpu for CPU result

    for(int i = 0; i < N * N; i++){ 
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c[i] = 0;
        c_cpu[i] = 0; // Initialize c_cpu
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    Timer copyTimer; // Timer for copy
    startTime(&copyTimer);
    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    stopTime(&copyTimer);
    printElapsedTime(copyTimer, "Data copy to GPU");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    Timer kernelTimer; // Timer for kernel
    startTime(&kernelTimer);
    gpu_mul<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    stopTime(&kernelTimer);
    printElapsedTime(kernelTimer, "GPU Kernel Execution");

    Timer gpuTimer; // Timer for GPU to CPU copy
    startTime(&gpuTimer);
    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure copy is finished before stopping timer
    stopTime(&gpuTimer);
    printElapsedTime(gpuTimer, "GPU to CPU copy");

    Timer cpuTimer; // Timer for CPU
    startTime(&cpuTimer);
    cpu_mul(a, b, c_cpu, N);
    stopTime(&cpuTimer);
    printElapsedTime(cpuTimer, "CPU Matrix Multiplication");

    // Verification
    bool correct = true;
    for(int i=0; i<N*N; ++i) {
        if (c[i] != c_cpu[i]) {
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "GPU and CPU results DO NOT match!" << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_cpu; // Delete c_cpu
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}