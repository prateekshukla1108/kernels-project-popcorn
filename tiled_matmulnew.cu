#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA_CALL(err)                                                \
    {                                                                       \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            fprintf(stderr, "CUDA error in file %s at line %d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#define TILE_SIZE 16

__global__
void matrix_multiplication_tiled(float *A, float* B, float *result, int rows_result, int col_result, int inner_dim){
    __shared__ float A_ds[TILE_SIZE][TILE_SIZE];
    __shared__ float B_ds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Loop over all tiles
    for (int ph=0; ph < (inner_dim + TILE_SIZE - 1)/TILE_SIZE; ++ph ){

        //r = row, c = TILE_SIZE * ph + tx;
        if (row < rows_result && (TILE_SIZE*ph + tx) < inner_dim){
            A_ds[ty][tx] = A[row * inner_dim + TILE_SIZE * ph + tx];
        }
        else{
            A_ds[ty][tx] = 0.0;
        }
        
        //row = TILE_SIZE * ph + ty, c = col
        if ((TILE_SIZE * ph + ty) < inner_dim && col < col_result){
            B_ds[ty][tx] = B[(TILE_SIZE * ph + ty) * col_result + col];
        }
        else{
            B_ds[ty][tx] = 0.0;
        }
    
        __syncthreads();

        for (int k=0; k < TILE_SIZE; ++k){
            if (row < rows_result && col < col_result){
                result[row * col_result + col] += A_ds[ty][k] * B_ds[k][tx];
            }
        }
        __syncthreads();
        
    }
}


float* matrix_multiplication(float *h_a, float *h_b, int row_a, int col_a, int row_b, int col_b){
    float *d_a, *d_b, *d_result;
    int size_a = sizeof(float) * row_a * col_a;
    int size_b = sizeof(float) * row_b * col_b;
    int size_result = sizeof(float) * row_a * col_b;
    float *h_result = new float[size_result];

    //Allocate device memory
    cudaError_t err = cudaMalloc((void**) &d_a, size_a);
    CHECK_CUDA_CALL(err);
    err = cudaMalloc((void**) &d_b, size_b);
    CHECK_CUDA_CALL(err);
    err = cudaMalloc((void**) &d_result, size_result);
    CHECK_CUDA_CALL(err);
    
    //copy matrices to device
    err = cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    CHECK_CUDA_CALL(err);
    err = cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    CHECK_CUDA_CALL(err);
    
    
    int thread_x = TILE_SIZE;
    int thread_y = TILE_SIZE;
    dim3 block_dims(thread_x, thread_y, 1);
    int blocks_x = (col_b + thread_x - 1)/thread_x;
    int blocks_y = (row_a + thread_y - 1)/thread_y;
    dim3 grid_dims(blocks_x, blocks_y, 1);
    

    matrix_multiplication_tiled<<<grid_dims, block_dims>>>(d_a, d_b, d_result, row_a, col_b, col_a);

    // copy result to host
    err = cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost);
    CHECK_CUDA_CALL(err);

    //free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return h_result;
}


void test_matrix_multiplication(){
    cout << "Running Test 1:\n";
    float *A = new float[6];
    float *B = new float[8];

    fill_n(A, 6, 1.0f);
    fill_n(B, 8, 1.0f);

    float *C = matrix_multiplication(A, B, 3, 2, 2, 4);

    for (int i=0; i< 3; ++i){
        for (int j=0; j<4; ++j){
            cout << C[i * 4 + j] << " ";
        }
        cout << "\n";
    }


    cout << "\nRunning test 2:\n";

    A = new float[50*50];
    B = new float[2500];

    for (int i=0; i < 50; ++i){
        A[i*50 + i] = 1;
        B[i*50 + i] = 1;
    }
    C = matrix_multiplication(A, B, 50, 50, 50, 50);
    for (int i=0; i< 50; ++i){
        for (int j=0; j<50; ++j){
            cout << C[i * 50 + j] << " ";
        }
        cout << "\n";
    }
}


int main(){
    test_matrix_multiplication();
}