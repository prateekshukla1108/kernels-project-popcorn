#include <iostream>
#include <cuda_runtime.h>
#include "timer.h" // Include Timer for accurate time measurement

// TILE_DIM is set to 32 because modern GPUs typically have warp sizes of 32.
// This allows efficient memory access and parallel execution.
#define TILE_DIM 32

__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < N / TILE_DIM; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
        
        __syncthreads();
        
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

void initializeMatrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = rand() % 100 / 10.0f;
    }
}

void cpuMatrixMul(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);
    
    float *h_A, *h_B, *h_C, *h_C_CPU;
    float *d_A, *d_B, *d_C;
    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_CPU = (float*)malloc(size);
    
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);
    
    Timer timer;
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    startTime(&timer);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Copy Time (A)", GREEN);
    
    startTime(&timer);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Copy Time (B)", GREEN);
    
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    
    startTime(&timer);
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU Execution Time", GREEN);
    
    startTime(&timer);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host Copy Time (C)", GREEN);
    
    startTime(&timer);
    cpuMatrixMul(h_A, h_B, h_C_CPU, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Execution Time", GREEN);
    
    free(h_A); free(h_B); free(h_C); free(h_C_CPU);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
