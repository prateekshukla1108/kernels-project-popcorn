#include <iostream>

/*

[[1,2],   .  [[1,3],    =   [[5, 11],
 [3,4]]       [2,4]]         [11, 25]]

*/

__global__ void matmul(int* X, int* Y, int* Z, int width){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int temp = 0;
    for(int k = 0; k<width; k++){
        temp += X[tx * width + k] * Y[width * k + ty];
    }
    Z[tx*width+ty] = temp;
}

int main( void ) {

    int N = 2;
    int size = N*N*sizeof(int);

    int h_X[4] = {1, 2, 3, 4};  
    int h_Y[4] = {1, 3, 2, 4};
    int h_Z[N*N];

    int *d_X, *d_Y, *d_Z;

    cudaMalloc((void**)&d_X, size); // allocate memory
    cudaMalloc((void**)&d_Y, size);
    cudaMalloc((void**)&d_Z, size);

    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, size, cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(N, N);
    
    matmul<<<grid_size, block_size>>>(d_X, d_Y, d_Z, N);

    cudaMemcpy(h_Z, d_Z, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);   // free memory
    cudaFree(d_Y);
    cudaFree(d_Z);

    std::cout << "Result matrix Z:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_Z[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}