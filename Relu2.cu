#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10      // a preprocessor macro for defining the number of elements in array

// simple CUDA kernel for ReLU activation function
__global__ void ReLU(float *vec, float *res, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        res[idx] =  vec[idx] > 0 ? vec[idx] : 0;
    }
}

void printArray(float *arr, int size){
    for(int i=0; i<size; i++){
        printf("%f ", arr[i]);
    }printf("\n");
}
int main(){
    size_t bytes = N * sizeof(float);    

    float *vec_h, *res_h;
    
    // allocating the the memory
    vec_h = (float*)malloc(bytes);
    res_h = (float*)malloc(bytes);

    // initializing the vector
    srand(time(NULL));
    for(int i=0; i<N; i++){
        vec_h[i] = ((float)rand() / (float)(RAND_MAX)) * 50.0f - 25.0f;  // Generate a random float between -25.0 and 25.0
    }

    float *vec_d, *res_d;
    cudaMalloc((void**)&vec_d,bytes);
    cudaMalloc((void**)&res_d, bytes);

    cudaMemcpy(vec_d, vec_h, bytes, cudaMemcpyHostToDevice);

    // defining the blockDim and gridDim
    dim3 THREADS(32, 1, 1);
    dim3 BLOCKS(N + THREADS.x / float(THREADS.x), 1, 1);

    // launching the kernel
    ReLU<<<BLOCKS, THREADS>>>(vec_d, res_d, N);

    cudaMemcpy(res_h, res_d, bytes, cudaMemcpyDeviceToHost);

    printf("vector:\n");
    printArray(vec_h, N);
    printf("vector after ReLU:\n");
    printArray(res_h, N);
    return 0;
}
