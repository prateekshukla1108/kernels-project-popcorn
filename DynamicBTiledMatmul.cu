#include <cuda_runtime.h>
#include <iostream>
#include <cmath> // For sqrt

using namespace std;

int computeOptimalTileWidth() {
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0); // Usually device 0

    // Shared memory required per block (for two tiles: M and N)
    const size_t bytes_per_float = sizeof(float);
    const size_t shared_mem_per_tile = 2 * bytes_per_float; // Per-element memory for two tiles

    // Max threads per block (e.g., 1024 → TILE_WIDTH ≤ 32)
    int max_tile_threads = static_cast<int>(sqrt(device_props.maxThreadsPerBlock));

    // Max tile size allowed by shared memory (sharedMemPerBlock / (2 * TILE_WIDTH² * bytes_per_float))
    int max_tile_shared = static_cast<int>(
        sqrt(device_props.sharedMemPerBlock / (2 * bytes_per_float))
    );

    // Optimal TILE_WIDTH is the minimum of the two limits
    int tile_width = min(max_tile_threads, max_tile_shared);

    // Clamp to a practical range (e.g., 16-32 for performance) not sure
    tile_width = max(tile_width, 16); // not sure
    tile_width = min(tile_width, 32); 

    return tile_width;
}

__global__ void MatrixMulKernel(float* P, float* M, float* N, 
unsigned int j, unsigned int k, unsigned int l, int TILE_WIDTH) {

    extern __shared__ float shared_mem[]; 
    float* MTile = shared_mem; // start of memory for MTile
    float* NTile = &shared_mem[TILE_WIDTH * TILE_WIDTH]; // start for NTile

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    
    // Calculate row and column indices for output matrix P
    unsigned int Row = blockIdx.y * TILE_WIDTH + ty;
    unsigned int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;

    // Loop over tiles along the K dimension
    for (int ph=0; ph < (k + TILE_WIDTH -1)/TILE_WIDTH; ++ph){
        // Load M element with boundary checks
        int M_Col = ph * TILE_WIDTH + tx;
        if (Row < j && M_Col < k){

            MTile[ty * TILE_WIDTH + tx] = M[Row * k + M_Col];
        }
        else{
            MTile[ty * TILE_WIDTH + tx] = 0.0f;
        }
        // Load N element with boundary checks

        int N_Row = ph * TILE_WIDTH + ty;
        if(N_Row < k && Col < l){
        NTile[ty * TILE_WIDTH + tx] = N[N_Row * l + Col];
        }else{
        NTile[ty * TILE_WIDTH + tx] = 0.0f;

        }
        __syncthreads();
        // Compute partial product
        for(unsigned int m = 0; m<TILE_WIDTH; ++m){
            Pvalue += MTile[ty * TILE_WIDTH + m] * NTile[m * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    // Write final result with boundary check
    if (Row < j && Col < l){
    P[Row * l +Col] = Pvalue;
    }
}
   
int main() {
    
    // Get dimensions from user
    int J, K, L;  // J x K * K x L matrices
    cout << "Enter dimensions for matrix multiplication (J K L): ";
    cout << "Space or newline seperated ";
    cin >> J >> K >> L;

    // Calculate sizes
    size_t size_A = J * K * sizeof(float);
    size_t size_B = K * L * sizeof(float);
    size_t size_C = J * L * sizeof(float);

    // Allocate host memory
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    cudaMallocHost(&h_A, size_A);
    cudaMallocHost(&h_B, size_B);
    cudaMallocHost(&h_C, size_C);

    // Initialize matrices with random values
    for(int i = 0; i < J * K; ++i) {
        h_A[i] = rand() % 10;
    }
    for(int i = 0; i < K * L; ++i) {
        h_B[i] = rand() % 10;
    }

    // Print Matrix A
    cout << "\nMatrix A (" << J << "x" << K << "):\n";
    for(int i = 0; i < J; ++i) {
        for(int j = 0; j < K; ++j) {
            cout << static_cast<int>(h_A[i*K + j]) << " ";
        }
        cout << endl;
    }

    // Print Matrix B
    cout << "\nMatrix B (" << K << "x" << L << "):\n";
    for(int i = 0; i < K; ++i) {
        for(int j = 0; j < L; ++j) {
            cout << static_cast<int>(h_B[i*L + j]) << " ";
        }
        cout << endl;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Set up execution parameters
    int TileWidth = computeOptimalTileWidth();
    dim3 blockSize(TileWidth, TileWidth);
    dim3 gridSize((J + TileWidth - 1) / TileWidth, 
                  (L + TileWidth - 1) / TileWidth);

    // Launch kernel
    size_t shared_mem_size = 2 * TileWidth * TileWidth * sizeof(float);
    MatrixMulKernel<<<gridSize, blockSize, shared_mem_size>>>(
        d_C, d_A, d_B, L, K, L, TileWidth);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize(); // Ensure kernel completes before copying results

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print Result Matrix C
    cout << "\nResult Matrix C (" << J << "x" << L << "):\n";
    for(int i = 0; i < J; ++i) {
        for(int j = 0; j < L; ++j) {
            cout << static_cast<int>(h_C[i*L + j]) << " ";
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
