#include <iostream>
#define TILE_WIDTH 16

__global__ void matmul(float* A, float* B, float* C, int width){
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = bx * blockDim.x + tx;
    int j = by * blockDim.y + ty;

    float value = 0;

    for(int phase = 0; phase < width/TILE_WIDTH; phase++){
        sh_A[tx][ty] = A[i*width + phase*TILE_WIDTH + j];
        sh_B[tx][ty] = B[(phase*TILE_WIDTH + ty)*width+j];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++){
            value += sh_A[tx][k] * sh_B[k][ty];
        }
        __syncthreads();
         
    }
    C[i*width+j] = value;
}

int main(void){
    int N = 1024;
    float h_A[N*N];
    float h_B[N*N];
    float h_C[N*N];

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float d_A[N*N], d_B[N*N], d_C[N*N];
    int size = N * N * sizeof(float);

    cudaMalloc( (void**)&d_A, size);
    cudaMalloc( (void**)&d_B, size);
    cudaMalloc( (void**)&d_C, size);
    cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice );

    dim3 grid_size(N/TILE_WIDTH, N/TILE_WIDTH);
    dim3 block_size(TILE_WIDTH, TILE_WIDTH);

    matmul<<< grid_size, block_size >>>(d_A, d_B, d_C, N);

    cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for( int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            std::cout << h_C[i*N + j] << " ";
        }
        std::cout<<std::endl;
    }

    return 0;
}