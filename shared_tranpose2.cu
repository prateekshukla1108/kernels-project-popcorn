#include <stdio.h>
#include <cuda_runtime.h>

#define TOTAL_VALS 1024
#define BLOCK_SIZE 32

__global__ void SharedMemoryTranspose(float *input, float *output, int width) {
    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE + 1];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < width && col < width) {
        block[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    __syncthreads();

    int transposedRow = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int transposedCol = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (transposedRow < width && transposedCol < width) {
        output[transposedRow * width + transposedCol] = block[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int size = TOTAL_VALS * TOTAL_VALS * sizeof(float);
    float *host_input, *host_output;
    float *device_input, *device_output;

    host_input = (float*)malloc(size);
    host_output = (float*)malloc(size);

    for (int i = 0; i < TOTAL_VALS * TOTAL_VALS; i++) {
        host_input[i] = i % 100;
    }

    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);

    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((TOTAL_VALS + BLOCK_SIZE - 1) / BLOCK_SIZE, (TOTAL_VALS + BLOCK_SIZE - 1) / BLOCK_SIZE);

    SharedMemoryTranspose<<<gridDim, blockDim>>>(device_input, device_output, TOTAL_VALS);

    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

    printf("Original Matrix :\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%2.0f ", host_input[i * TOTAL_VALS + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix :\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%2.0f ", host_output[i * TOTAL_VALS + j]);
        }
        printf("\n");
    }

    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
