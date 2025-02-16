#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matmul_gpu_kernel(float *m_d, float*n_d, float *p_d, int h, int w){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k=0; k<w; k++) {
        p_d[row*w + col] += m_d[row*w + k] * n_d[ k*w+ col];
    }
}


void matmul_gpu(float *m, float*n, float *p, int h, int w){
    // allocate gpu memory
    int size = h *w * sizeof(float);
    float *m_d, *n_d, *p_d;

    cudaMalloc((void **)&m_d, size);
    cudaMalloc((void **)&n_d, size);
    cudaMalloc((void **)&p_d, size);

    // copy memory from CPU to CUDa

    cudaMemcpy(m_d, m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(n_d, n, size, cudaMemcpyHostToDevice);

    // Declare the kernal and run it
    dim3 BlockDim(32, 32);
    dim3 GridDim((w + BlockDim.x - 1)/ BlockDim.x, (h + BlockDim.y -1)/ BlockDim.y);

    matmul_gpu_kernel<<<GridDim, BlockDim>>>(m_d, n_d, p_d, h, w);
    // Copy data back to CPU from CUDA
    cudaMemcpy(p, p_d, size, cudaMemcpyDeviceToHost);
    


    //free the allocated memory
    cudaFree(m_d);
    cudaFree(n_d);
    cudaFree(p_d);



}

int main(void){
    // declare the arrays
    const int h = 320;
    const int w = 320;
    // int k = h*w;
    float *m = (float*)malloc(h * w *sizeof(float));
    float *n = (float*)malloc(h * w*sizeof(float));
    float *p= (float*)malloc(h * w*sizeof(float));
    // memset(p, 0, h * w * sizeof(float));
    
    // assign random values
    for (int i=0; i<h;i++){
        for (int j=0;j<w;j++){
            m[i*h + j] = i+j;
            n[i*w + j] = i+j;
        }
    }

    // cpu operation
    clock_t start, stop;
    start = clock();

    for (int i =0; i<h;i++){
        for(int j=0; j<w;j++){
            for(int k=0; k<h; k++){
                p[i*w + j] += m[w*i + k] * n[k*w + j];
            }

        }
    }
    stop = clock();
    double cpu_time_used = (double)(stop - start)/ CLOCKS_PER_SEC;
    cout << "CPU Time " << cpu_time_used * 1000 << " ms"<<endl;

    // gpu_operation
    float *p_d= (float*)malloc(h * w*sizeof(float));

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    matmul_gpu(m, n, p_d, h, w);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_used;
    cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
    cout << "GPU Time: " << gpu_time_used << " ms" << endl;
    for(int i=0; i<h*w;i++){
        if (p[i] != p_d[i]){
            cout << "MisMatch" <<endl;
            break;
        }

    }


    free(m);
    free(n);
    free(p);
    free(p_d);

    //return
    return 0;



}