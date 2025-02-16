#include <stdio.h>
#include <cuda_runtime.h>

__global__
void vector_addition_kernel(float* a, float* b, float* c, int arr_len){
    int idx = blockIdx.x * blockDim.x  + threadIdx.x;
    if (idx < arr_len) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_addition(float* a_h, float* b_h, float* c_h, int arr_len){
    int size_vectors = sizeof(float) * arr_len;
    float *a_d, *b_d, *c_d;
    
    //Allocate device memory
    cudaMalloc((void**) &a_d, size_vectors);
    cudaMalloc((void**) &b_d, size_vectors);
    cudaMalloc((void**) &c_d, size_vectors);

    //Copy inputs to device
    cudaMemcpy(a_d, a_h, size_vectors, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_vectors, cudaMemcpyHostToDevice);
    
    int block_dims = 256;
    int num_blocks = (arr_len + block_dims - 1)/block_dims;

    vector_addition_kernel<<<num_blocks, block_dims>>>(a_d, b_d, c_d, arr_len);

    //Copy results to host
    cudaMemcpy(c_h, c_d, size_vectors, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


void test_vector_addition(float* C, int arr_len){
    for (int i=0; i<arr_len; ++i){
        if (C[i] != 3*i) {
            printf("Vector Addition failed\n");
            return;
        }
    }
    printf("Vector Addition successful!\n");
}


void initialize_vectors(float* A, float* B, int arr_len){
    for (int i=0; i<arr_len; ++i){
        A[i] = i;
        B[i] = 2*i;
    }
}


int main(){
    const int block_dim = 256;
    const int arr_len = 1000;

    //allocated on stack frame
    float A[arr_len];
    float B[arr_len];
    float C[arr_len];

    initialize_vectors(A, B, arr_len);
    vector_addition(A, B, C, arr_len);
    test_vector_addition(C, arr_len);
}