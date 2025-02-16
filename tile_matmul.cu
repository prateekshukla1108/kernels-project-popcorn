#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_WIDTH 32

__global__ void vecmul_tile_kernel(float *a, float *b, float *c, int m, int k, int n) {
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    
    // The row and col indicates the index in the output matrix which thread is currently working on.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float c_val = 0.0;
    for (int ph=0;ph<((k+TILE_WIDTH-1)/TILE_WIDTH);ph++){

        if(row < m && ph*TILE_WIDTH + tx < k) {
            a_tile[ty][tx] = a[row*k + ph*TILE_WIDTH + tx];
        }
        else {
            a_tile[ty][tx] = 0.0f;
        }
        if (ph*TILE_WIDTH+ty < k && ty < n) {
            b_tile[ty][tx] = b[(ph*TILE_WIDTH+ty)*n + col];
        }
        else {
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for(int i=0; i<TILE_WIDTH; i++){
                c_val += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();

    }
    c[row*n + col] = c_val;


}

void vecmul_tile(float *a, float *b, float *c, int m, int k, int n) {

    float *a_d, *b_d, *c_d;
    // allocate gpu memory
    cudaMalloc((void **)&a_d, m*k*sizeof(float));
    cudaMalloc((void **)&b_d, k*n*sizeof(float));
    cudaMalloc((void **)&c_d, m*n*sizeof(float));


    // copy contents from CPU to GPU
    cudaMemcpy(a_d, a, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, k*n*sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernels
    dim3 numThreads(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((n + numThreads.x - 1)/numThreads.x, (m+ numThreads.y-1)/numThreads.y);
    vecmul_tile_kernel<<<numBlocks, numThreads>>>(a_d, b_d, c_d, m, k, n);
    // copy back the contents from GPU to CPU
    cudaMemcpy(c, c_d, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

bool compareMatrices(float *c, float *c_gpu, int m, int n, float tolerance = 1e-4) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(c[i * n + j] - c_gpu[i * n + j]) > tolerance) {
                cout << "Mismatch at (" << i << ", " << j << "): CPU = " << c[i * n + j] << ", GPU = " << c_gpu[i * n + j] << " "<< c[i * n + j] - c_gpu[i * n + j] <<endl;
                return false;
            }
        }
    }
    return true;
}

int main(void) {
    // declare the arrays
    const int m = 3200;
    const int k = 320;
    const int n = 3200;

    float *a = (float*)malloc(m*k*sizeof(float));
    float *b = (float*)malloc(k*n*sizeof(float));
    float *c = (float*)malloc(m*n*sizeof(float));

    for (int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            a[i*k + j] =  static_cast<float>(rand()) / RAND_MAX;
            // cout << a[i*k + j]<< " ";
        }
        // cout << "\n";
    }

    for (int i=0;i<k;i++){
        for(int j=0;j<n;j++){
            b[i*n + j] =  static_cast<float>(rand()) / RAND_MAX;
            // cout << b[i*n + j]<< " ";
        }
        // cout << "\n";
    }

    clock_t start, stop;
    start = clock();

    for (int i=0; i<m;i++){
        for(int j=0; j<n;j++){
            float temp = 0.0f;
            for (int l=0; l<k; l++){
                temp += a[i*k + l] * b[l*n + j];
            }
            c[i*n + j] = temp;
        }
    }
    stop = clock();
    double cpu_time_used = (double)(stop - start)/ CLOCKS_PER_SEC;
    cout << "CPU Time : " << cpu_time_used * 1000 << " ms"<<endl;

    // GPU //
    float *c_gpu = (float*)malloc(m*n*sizeof(float));
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    vecmul_tile(a, b, c_gpu, m, k, n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_used;
    cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
    cout << "GPU Time: " << gpu_time_used << " ms" << endl;
    
 
    if (compareMatrices(c, c_gpu, m, n)) {
        std::cout << "Matrices match!" << std::endl;
    } else {
        std::cout << "Matrices do not match!" << std::endl;
    }


    free(a);
    free(b);
    free(c);
    free(c_gpu);


}