/*
Program to increment each element of an array by 1 using GPU
*/
#include <iostream>

__global__ void increment_gpu(int* a){
    int i = threadIdx.x;
    a[i] += 1;
}

int main(void) {
    int N = 5;
    int h_a[] = {2,3,5,7,11};
    int* d_a;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMemcpy(d_a, h_a, 5*sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid_size(1);
    dim3 block_size(N);
    increment_gpu<<<grid_size, block_size>>>(d_a);
    cudaMemcpy(h_a, d_a, 5*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++){
        printf("%d\n", h_a[i]);
    }
}
