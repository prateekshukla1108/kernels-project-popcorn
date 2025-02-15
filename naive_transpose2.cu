#include <stdio.h>
#include <cuda_runtime.h>

#define TOTAL_VALUE_POINTS 1024

__global__ void naiveTranspose(float *input, float *output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        output[col * width + row] = input[row * width + col]; 
    }
}

int main() {
    int size = TOTAL_VALUE_POINTS * TOTAL_VALUE_POINTS * sizeof(float);
    float *host_input, *host_output;
    float *device_input, *device_output;


    host_input = (float*)malloc(size);
    host_output = (float*)malloc(size);

    for (int i = 0; i < TOTAL_VALUE_POINTS * TOTAL_VALUE_POINTS; i++) {
        host_input[i] = i % 100; 
    }

    
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);


    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32); 
    dim3 gridDim((TOTAL_VALUE_POINTS + 31) / 32, (TOTAL_VALUE_POINTS + 31) / 32); 

    naiveTranspose<<<gridDim, blockDim>>>(device_input, device_output, TOTAL_VALUE_POINTS);

    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

    printf("Original Matrix (first 4x4 block):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%2.0f ", host_input[i * TOTAL_VALUE_POINTS + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix (first 4x4 block):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%2.0f ", host_output[i * TOTAL_VALUE_POINTS + j]);
        }
        printf("\n");
    }


    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
