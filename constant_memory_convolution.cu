#include <iostream>
#include <cuda_runtime.h>
// #include <torch/extension.h>

using namespace std;

# define BLOCKSIZE 32
# define FILTER_RADIUS 2
__constant__ float Filter[2*FILTER_RADIUS+1][2*FILTER_SIZE+1]

#define CHECK_CUDA_CALL(err)                                                \
    {                                                                       \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            fprintf(stderr, "CUDA error in file %s at line %d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


__global__
void convolution_shared_memory(float* input, float* output, int rows, int cols){
    int o_row = blockDim.y * blockIdx.y + threadIdx.y;
    int o_col = blockDim.x * blockIdx.x + threadIdx.x;

    float prod_accumulation = 0.0;

    for (int f_row=0; f_row < 2*FILTER_RADIUS + 1; ++f_row){
        for (int f_col=0; f_col < 2*FILTER_RADIUS + 1; ++f_col){
            int in_row = o_row - FILTER_RADIUS + f_row;
            int in_col = o_col - FILTER_RADIUS + f_col;
            if (in_row > -1 && in_row < rows && in_col > -1 && in_col < cols){
                prod_accumulation += Filter[f_row][f_col] * input[in_row * cols + in_col];
            }
        }
    }

    output[o_row * cols + o_col] = prod_accumulation;

}
