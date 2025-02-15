#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define TOTAL_NUMS 512

__global__ void matMul(float *device_A, float *device_B, float *device_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += device_A[row * n + k] * device_B[k * n + col];
        }
        device_C[row * n + col] = sum;
    }
}

void verify(float *host_A, float *host_B, float *host_C, int n) {
    float *expected_C = (float*)malloc(n * n * sizeof(float));
    if (expected_C == NULL) {
        perror("Memory allocation failed in verify");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            expected_C[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                expected_C[i * n + j] += host_A[i * n + k] * host_B[k * n + j];
            }
        }
    }

    float tolerance = 1e-5f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(host_C[i * n + j] - expected_C[i * n + j]) > tolerance) {
                printf("Error at C[%d][%d]: Expected %f, Got %f\n", i, j, expected_C[i * n + j], host_C[i * n + j]);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("Matrix multiplication verified successfully.\n");
    } else {
        printf("Matrix multiplication failed with %d errors.\n", errors);
    }

    free(expected_C);
}

int main() {
    int size = TOTAL_NUMS * TOTAL_NUMS * sizeof(float);
    float *host_A, *host_B, *host_C, *device_A, *device_B, *device_C;

    host_A = (float*)malloc(size);
    host_B = (float*)malloc(size);
    host_C = (float*)malloc(size);

    for (int i = 0; i < TOTAL_NUMS * TOTAL_NUMS; i++) {
        host_A[i] = (float)rand() / (float)RAND_MAX;
        host_B[i] = (float)rand() / (float)RAND_MAX;
    }

    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMalloc(&device_C, size);

    cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((TOTAL_NUMS + 15) / 16, (TOTAL_NUMS + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMul<<<gridSize, blockSize>>>(device_A, device_B, device_C, TOTAL_NUMS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

    verify(host_A, host_B, host_C, TOTAL_NUMS);

    free(host_A); free(host_B); free(host_C);
    cudaFree(device_A); cudaFree(device_B); cudaFree(device_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
