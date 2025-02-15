#include<iostream>
// Although nvcc can do this automatically, it's good to include it explicitly
// as a good practice
#include<cuda_runtime.h> 

__global__ void my_kernel(){
    /*
    This kernel function gets number of blocks and threads per block, so grid size is determined here.
    And we get IDs for everything.
    */
    // can't do this because blockIDx.x is a HOST variable??? What???
    // or I think we can not print inside the kernel function
    // std::cout<<"Block ID: "<<blockIdx.x<<" Thread ID: "<<threadIdx.x<<std::endl;
    // Print grid, block, and thread details
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("row = %d, col = %d, GridDim.x = %d | GridDim.y = %d | BlockIdx.x = %d | BlockIdx.y = %d | ThreadIdx.x = %d | Thread ID.y = %d\n",
           row, col, gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

}

// argc is the number of command line arguments
// argv is the array of command line arguments
// usage ./object_file arg1 arg2 arg3 (argc=4 here)
// all args are optional and are character arrays
int main(int argc, char** argv){

    // This program is about understanding how CUDA gives us block and thread indices
    // We will define a 2D grid in CUDA and then get all the information about the block and thread indices

    // Let's define a 2D grid of 1024 threads in a block
    
    dim3 my_grid(4, 2); // 4 Blocks in X direction (i.e. Columns) and 2 Blocks in Y direction (i.e. Rows)
    dim3 my_block(2, 2); // 2 threads in X direction and 2 threads in Y direction in each block

    // So blocks or grids are only defined in the kernel launch
    // function_name<<<grid_size, block_size>>>(arguments); happends to be <<<block_size, number_of_threads_in_block>>> for 1D grid.
    /*
    BlockIdx.y = 3  [ (0,3)  (1,3)  (2,3)  (3,3)  (4,3)  (5,3) ]
    BlockIdx.y = 2  [ (0,2)  (1,2)  (2,2)  (3,2)  (4,2)  (5,2) ]
    BlockIdx.y = 1  [ (0,1)  (1,1)  (2,1)  (3,1)  (4,1)  (5,1) ]
    BlockIdx.y = 0  [ (0,0)  (1,0)  (2,0)  (3,0)  (4,0)  (5,0) ]
                ^       ^       ^       ^       ^       ^
       BlockIdx.x = 0   1       2       3       4       5
    */
    my_kernel<<<my_grid, my_block>>>(); 
    cudaDeviceSynchronize();
    return 0; // for successful execution of the program


}