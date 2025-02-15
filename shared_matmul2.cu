#include <iostream>
#include <cuda_runtime.h>
#include <cmath> 

#define BLOCK_SIZE 16
#define MATRIX_SIZE 256

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(error);
    }
}

__global__ void matrixMulShared(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    float result = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        if (row < n && k * BLOCK_SIZE + threadIdx.y < n)
            sharedA[threadIdx.y][threadIdx.x] = A[row * n + k * BLOCK_SIZE + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;

        if (k * BLOCK_SIZE + threadIdx.x < n && col < n)
            sharedB[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            result += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
         C[row * n + col] = result;
}

void verifyResults(float* h_A, float* h_B, float* h_C, int n) {
    float* h_C_cpu = new float[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_C_cpu[i * n + j] = 0.0f;
            for (int k = 0; k < n; ++k) {
                h_C_cpu[i * n + j] += h_A[i * n + k] * h_B[k * n + j];
            }
        }
    }

    double error = 0.0;
    for (int i = 0; i < n * n; ++i) {
        error += fabs(h_C[i] - h_C_cpu[i]); 
    }

    if (error < 1e-4 * n * n) { 
        std::cout << "Results verified successfully." << std::endl;
    } else {
        std::cout << "Results verification failed. Error: " << error << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (fabs(h_C[i * n + j] - h_C_cpu[i * n + j]) > 1e-4) {
                    std::cout << "Mismatch at (" << i << ", " << j << "): GPU=" << h_C[i * n + j] << ", CPU=" << h_C_cpu[i * n + j] << std::endl;
                }
            }
        }
    }

    delete[] h_C_cpu;
}


int main() {
    int n = MATRIX_SIZE;

    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    for (int i = 0; i < n * n; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_C[i] = 0.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaError(cudaMalloc(&d_A, n * n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_B, n * n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_C, n * n * sizeof(float)));

    checkCudaError(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    verifyResults(h_A, h_B, h_C, n);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    checkCudaError(cudaFree(d_A));
    checkCudaError(cudaFree(d_B));
    checkCudaError(cudaFree(d_C));
    cudaDeviceReset();

    return 0;
}
