#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <cstdlib>   
#include <iostream>  // For std::cout
using namespace std; // For cout

#define TileWidth 2
__global__ void tiledMatMul(float *d_A, float *d_B, float *d_C, int Width){

    //Shared memory declaration
    __shared__ float Mds[TileWidth][TileWidth];
    __shared__ float Nds[TileWidth][TileWidth];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Compute row and column indices
    int row = by * TileWidth + ty;
    int col = bx * TileWidth + tx;

    // Boundary Check
    if (row < Width && col < Width){

    // Init sum
    float Pvalue = 0.0;

    // number of tiles needed
    int nTiles = Width/TileWidth;

    for (int ph=0; ph < nTiles; ++ph){

        Mds[ty][tx] = d_A[row * Width + (ph * TileWidth + tx)];
        Nds[ty][tx] = d_B[(ph * TileWidth + ty) * Width + col];
        __syncthreads(); // wait for all threads to fetch their items

        for (int k=0; k < TileWidth; k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // wait for all threads to finish matmul
    }
        d_C[row * Width + col] = Pvalue; //assign
    }
}

int main(){

    int Width = 4; // ensure divisible by TileWidth
    // int TileWidth = 2;

    int Size = Width * Width * sizeof(float); // dont forget float

    //Init and Allocate Host ptrs and memory
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C; 
    cudaMallocHost(&h_A, Size);
    cudaMallocHost(&h_B, Size);
    cudaMallocHost(&h_C, Size);
    cudaMalloc(&d_A, Size);
    cudaMalloc(&d_B, Size);
    cudaMalloc(&d_C, Size);

    // Init host matrices
    for(int i = 0; i < Width*Width; ++i) {
        h_A[i] = rand() % 10 ;  // Random integers between 0 and 9
        h_B[i] = rand() % 10 ;  // Random integers between 0 and 9
    }
        // Print Matrix A
    cout << "\nMatrix A:\n";
    for(int i = 0; i < Width; ++i) {
        for(int j = 0; j < Width; ++j) {
            cout << static_cast<int>(h_A[i*Width + j]) << " ";
        }
        cout << endl;
    }

    // Print Matrix B
    cout << "\nMatrix B:\n";
    for(int i = 0; i < Width; ++i) {
        for(int j = 0; j < Width; ++j) {
            cout << static_cast<int>(h_B[i*Width + j]) << " ";
        }
        cout << endl;
    }
    
    // Copy from Host to Device
    cudaMemcpy(d_A, h_A, Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions and launch kernel
    dim3 blockSize(TileWidth, TileWidth);
    dim3 gridSize((Width + TileWidth - 1) / TileWidth,
    (Width + TileWidth - 1) / TileWidth);
    
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, Width);

    // error checking
    cudaError_t err;
    err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
    if (err != cudaSuccess)
    {
      printf("Error: %s\n", cudaGetErrorString(err));
    }

    //copy back the result
    cudaMemcpy(h_C, d_C, Size, cudaMemcpyDeviceToHost);

       // Print Result Matrix C
    cout << "\nResult Matrix C:\n";
    for(int i = 0; i < Width; ++i) {
        for(int j = 0; j < Width; ++j) {
            cout << static_cast<int>(h_C[i*Width + j]) << " ";
        }
        cout << endl;
    }

    // Free memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
