#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float *x, float *y, float *z, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

void vec_add(float *x, float *y, float *z, int N){
    float *x_d, *y_d, *z_d;
    int size = N * sizeof(float);

    // Allocate space in GPU
    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);
    cudaMalloc((void **)&z_d, size);

    // Copy from Host to Device
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    const unsigned int threadsPerBlock = 1024;
    const unsigned int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;

    // Launch the kernel
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, z_d, N);

    // Copy from Device to Host
    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    // Free the pointers
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

void vec_add_cpu(float *x, float *y,float *z, int N){
    for (unsigned int i = 0; i < N; i++){
        z[i] = x[i] + y[i];
    }
}

int main(void){
    const int n = 1 << 25; // 2^25
    float *A = (float*) malloc(n * sizeof(float));
    float *B = (float*) malloc(n * sizeof(float));
    float *C = (float*) malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    // Timing CPU execution
    clock_t start = clock();
    vec_add_cpu(A, B, C, n);
    clock_t stop = clock();
    double cpu_time_used = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("CPU Time taken: %f milliseconds\n", cpu_time_used * 1000);

    // Timing GPU execution using CUDA events
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time_used;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    vec_add(A, B, C, n);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
    printf("GPU Time taken: %f milliseconds\n", gpu_time_used);

    // Free host memory
    free(A);
    free(B);
    free(C);

    // Destroy CUDA events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
