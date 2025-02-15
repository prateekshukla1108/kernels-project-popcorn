#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

#define TOTAL_VALS 1000
#define THREADS 256

__global__ void relu(float *array_1, float *array_2, int n) {
    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    if (indices < n) {
        array_2[indices] = fmaxf(0.0f, array_1[indices]);
    }
}

__global__ void sigmoid(float *array_1, float *array_2, int n) {
    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    if (indices < n) {
        array_2[indices] = 1.0f / (1.0f + expf(-array_1[indices]));
    }
}

__global__ void softmax(float *array_1, float *array_2, float *global_sum, int n) {
    __shared__ float exp_sum[THREADS];
    int indices = blockDim.x * blockIdx.x + threadIdx.x;

    float expn = (indices < n) ? expf(array_1[indices]) : 0.0f;
    exp_sum[threadIdx.x] = expn;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            exp_sum[threadIdx.x] += exp_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(global_sum, exp_sum[0]);
    }
    __syncthreads();

    if (indices < n) {
        array_2[indices] = expn / (*global_sum);
    }
}

__host__ void verifyResults(float *cpu_result, float *gpu_result, int n, const char *label, float tolerance = 1e-6) {
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("%s Mismatch at index %d: CPU = %f, GPU = %f\n", label, i, cpu_result[i], gpu_result[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("%s Output Verified Successfully!\n", label);
    } else {
        printf("%s Verification Failed!\n", label);
    }
}

int main() {
    int blocks = (TOTAL_VALS + THREADS - 1) / THREADS;
    int mem_size = sizeof(float) * TOTAL_VALS;

    float *host_x = (float *)malloc(mem_size);
    float *host_relu = (float *)malloc(mem_size);
    float *host_sigmoid = (float *)malloc(mem_size);
    float *host_softmax = (float *)malloc(mem_size);

    float *device_x, *device_relu, *device_sigmoid, *device_softmax, *global_exp_add;
    
    cudaMalloc(&device_x, mem_size);
    cudaMalloc(&device_relu, mem_size);
    cudaMalloc(&device_sigmoid, mem_size);
    cudaMalloc(&device_softmax, mem_size);
    cudaMalloc(&global_exp_add, sizeof(float));

    for (int i = 0; i < TOTAL_VALS; i++) {
        host_x[i] = (float)(rand() % 200 - 100) / 10.0f;
    }

    cudaMemcpy(device_x, host_x, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(global_exp_add, 0, sizeof(float));

    relu<<<blocks, THREADS>>>(device_x, device_relu, TOTAL_VALS);
    cudaMemcpy(host_relu, device_relu, mem_size, cudaMemcpyDeviceToHost);
    verifyResults(host_x, host_relu, TOTAL_VALS, "ReLU");

    sigmoid<<<blocks, THREADS>>>(device_x, device_sigmoid, TOTAL_VALS);
    cudaMemcpy(host_sigmoid, device_sigmoid, mem_size, cudaMemcpyDeviceToHost);
    verifyResults(host_x, host_sigmoid, TOTAL_VALS, "Sigmoid");

    softmax<<<blocks, THREADS>>>(device_x, device_softmax, global_exp_add, TOTAL_VALS);
    cudaMemcpy(host_softmax, device_softmax, mem_size, cudaMemcpyDeviceToHost);
    verifyResults(host_x, host_softmax, TOTAL_VALS, "Softmax");

    cudaFree(device_x);
    cudaFree(device_relu);
    cudaFree(device_sigmoid);
    cudaFree(device_softmax);
    cudaFree(global_exp_add);

    free(host_x);
    free(host_relu);
    free(host_sigmoid);
    free(host_softmax);

    return 0;
}
