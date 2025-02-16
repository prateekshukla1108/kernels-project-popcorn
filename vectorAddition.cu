#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>


#define N 512

__global__ void device_add(int *d_x, int *d_y, int *d_z){
      int idx = blockIdx.x;
      d_z[idx] = d_x[idx] + d_y[idx]; 
}
void host_add(int *a, int *b, int *c){
    for(int i=0;i<N;i++){
        c[i] = a[i] + b[i];
    }
}
void fill_array(int *a){
    for(int i=0;i<N;i++){
        a[i] = i; 
    }
}
void print_out(int *a, int *b, int *c){
    for(int i=0;i<N;i++){
        printf("[%d] + [%d] = %d\n", a[i], b[i], c[i]);
    }
}
int main()  {
    int *a,*b,*c;
    int size = N * sizeof(int);
    a = (int *)malloc(size);
    fill_array(a);
    b = (int *)malloc(size);
    fill_array(b);
    c = (int *)malloc(size);
    fill_array(c);
    auto start_host = std::chrono::high_resolution_clock::now();

    host_add(a,b,c);
        auto stop_host = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> host_time = stop_host - start_host;

    print_out(a,b,c);

    int *d_x;
    int *d_y;
    int *d_z;
    cudaMalloc(&d_x,N * sizeof(int));
    cudaMalloc(&d_y,N * sizeof(int));
    cudaMalloc(&d_z,N * sizeof(int));
    
    cudaMemcpy(d_x, a, N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, b, N * sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start recording time
    cudaEventRecord(start, 0);
    device_add<<<N,1>>>(d_x,d_y,d_z);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c, d_z, N*sizeof(int),cudaMemcpyDeviceToHost );

    print_out(a,b,c);

     printf("GPU Kernel execution time: %f ms\n", milliseconds);
     printf("CPU execution time: %f ms\n", host_time.count());

    free(a); free(b); free(c);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    return 0;

}

