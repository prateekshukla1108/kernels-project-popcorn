#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));


#define TILE_DIM 16
# define MAX_FILTER_RADIUS 7
__constant__ float Filter[(2*MAX_FILTER_RADIUS+1) * (2*MAX_FILTER_RADIUS+1)];


__global__
void convolution_tiled(float* input, float* output, int rows, int cols, int filter_radius){
    int o_col = blockIdx.x * blockDim.x + threadIdx.x;
    int o_row = blockIdx.y * blockDim.y + threadIdx.y;

    //shared memory
    __shared__ float input_shared[TILE_DIM * TILE_DIM];
    if (o_col < cols && o_row < rows){
        input_shared[threadIdx.y * TILE_DIM + threadIdx.x] = input[o_row * cols + o_col];
    }
    else {//the else condition is not necessary as these will never be accessed
        input_shared[threadIdx.y * TILE_DIM + threadIdx.x] = 0.0;
    }
    __syncthreads();


    int block_bound_c1 = blockIdx.x * blockDim.x; 
    int block_bound_c2 = (blockIdx.x + 1) * blockDim.x - 1;
    int block_bound_r1 = blockIdx.y * blockDim.y;
    int block_bound_r2 = (blockIdx.y + 1) * blockDim.y - 1;

    // This should be around the for loop. If condition not met, only load the shared memory
    if (o_col < cols && o_row < rows){
        float accumulation = 0.0;
        int filter_edge = (2*filter_radius+1);
        for (int filter_row=0; filter_row < filter_edge; ++filter_row){
            for (int filter_col=0; filter_col < filter_edge; ++filter_col){
                int input_row = o_row - filter_radius + filter_row;
                int input_col = o_col - filter_radius + filter_col;

                //if input in loaded tiled use it
                if (block_bound_r1 <=input_row  && input_row<=block_bound_r2 && block_bound_c1<=input_col && input_col<=block_bound_c2){
                    int s_row = input_row - block_bound_r1;
                    int s_col = input_col - block_bound_c1;
                    accumulation += Filter[filter_row*filter_edge + filter_col] * input_shared[s_row * TILE_DIM + s_col];
                } // if not in loaded tile and not outside input bounds load from global - proably will hit cache
                else if (input_row > -1 && input_row < rows && input_col > -1 && input_col < cols){
                    accumulation += Filter[filter_row * filter_edge + filter_col] * input[input_row * cols + input_col];
                }
            }
        }

        // write output accumulation
        output[o_row * cols + o_col] = accumulation;
    }
}

torch::Tensor convolution(torch::Tensor input, torch::Tensor filter){
    TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(filter.device().is_cpu(), "Filter tensor must be on CPU");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(filter.dim() == 2, "Filter tensor must be 2D");
    TORCH_CHECK(filter.size(0) == filter.size(1), "Filter must be square shaped");
    TORCH_CHECK(filter.size(0) % 2 == 1, "Filter must have odd sized edge");
    TORCH_CHECK(filter.size(0) <= MAX_FILTER_RADIUS * 2 + 1, "Filter must be smaller than MAX_FILTER_RADIUS");

    int rows = input.size(0); int cols = input.size(1);
    input = input.cuda();
    auto output = torch::zeros({rows, cols}, input.options());
    float* filter_d;
    cudaMemcpyToSymbol(Filter, filter.data_ptr<float>(), filter.numel()*sizeof(float));

    int thread_x = TILE_DIM; int thread_y = TILE_DIM;
    dim3 block_size(thread_x, thread_y);

    int block_x = (cols + thread_x - 1)/ thread_x;
    int block_y = (rows + thread_y - 1)/ thread_y;
    dim3 grid_size(block_x, block_y);

    convolution_tiled<<<grid_size, block_size>>>(
                                      input.data_ptr<float>(),
                                      output.data_ptr<float>(),
                                      rows,
                                      cols,
                                      filter.size(0)/2
                                      );
    return output.cpu();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(convolution)
}
