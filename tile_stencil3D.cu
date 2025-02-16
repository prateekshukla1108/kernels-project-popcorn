#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath> // For fabs
using namespace std;
#define TILE_DIM 10
#define OUT_TILE_DIM 8


// Define the size of the 3D grid
const int NX = 512;
const int NY = 512;
const int NZ = 512;

// Define the radius of the stencil (e.g., 1 for a 3x3x3 stencil)
// const int RADIUS = 1;
// const int TILE_DIM = 4;
// const int out_tile_dim = 2;
// CPU implementation of the 3D stencil
void stencil_3d_cpu(const float* input, float* output, int nx, int ny, int nz) {
    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int idx = z * ny * nx + y * nx + x;

                output[idx] =
                    input[idx] +
                    input[(z - 1) * ny * nx + y * nx + x] +
                    input[(z + 1) * ny * nx + y * nx + x] +
                    input[z * ny * nx + (y - 1) * nx + x] +
                    input[z * ny * nx + (y + 1) * nx + x] +
                    input[z * ny * nx + y * nx + (x - 1)] +
                    input[z * ny * nx + y * nx + (x + 1)];
            }
        }
    }
}


// const int TILE_DIM = 16;  // Tile size for shared memory

__global__ void stencil_3d_gpu(const float* input, float* output, int nx, int ny, int nz) {
    // Shared memory tile
    __shared__ float tile[TILE_DIM][TILE_DIM][TILE_DIM];

    // Calculate global indices
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;  
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;  
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1; 

    // Load data into shared memory (with bounds checking)
    if (i >= 0 && i < nz && j >= 0 && j < ny && k >= 0 && k < nx) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[i * ny * nx + j * nx + k];
    } 
     else {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f; // Optional: Initialize to zero if out of bounds
    }
    __syncthreads();

    // Perform stencil operation (only on interior elements)
        if (i >= 1 && i < nz - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nx - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < TILE_DIM - 1 && 
            threadIdx.y >= 1 && threadIdx.y < TILE_DIM - 1 && 
            threadIdx.x >= 1 && threadIdx.x < TILE_DIM - 1) {
            
            output[i * ny * nx + j * nx + k] = 
                tile[threadIdx.z][threadIdx.y][threadIdx.x] + 
                tile[threadIdx.z - 1][threadIdx.y][threadIdx.x] + 
                tile[threadIdx.z + 1][threadIdx.y][threadIdx.x] + 
                tile[threadIdx.z][threadIdx.y - 1][threadIdx.x] +  
                tile[threadIdx.z][threadIdx.y + 1][threadIdx.x] + 
                tile[threadIdx.z][threadIdx.y][threadIdx.x - 1] + 
                tile[threadIdx.z][threadIdx.y][threadIdx.x + 1];
        }
    }
}

// Function to compare CPU and GPU results
bool compare_results(const float* cpu_result, const float* gpu_result, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpu_result[i] << ", GPU = " << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Initialize input and output arrays
    int size = NX * NY * NZ;
    float* input = new float[size];
    float* cpu_output = new float[size];
    float* gpu_output = new float[size];

    // Fill input array with 1.0f for simplicity
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0f;
    }
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    cout<<"Max threads per block: " << prop.maxThreadsPerBlock << "\n"<< "Shared mem per block: " << prop.sharedMemPerBlock << " bytes\n";
    // Run CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    stencil_3d_cpu(input, cpu_output, NX, NY, NZ);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // Allocate memory on the GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy input data to the GPU
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes for CUDA
    dim3 blockSize(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 gridSize((NZ + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                  (NY + OUT_TILE_DIM- 1) / OUT_TILE_DIM,
                  (NX + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Run GPU implementation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    stencil_3d_gpu<<<gridSize, blockSize>>>(d_input, d_output, NX, NY, NZ);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

    // Copy output data back to the host
    cudaMemcpy(gpu_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    if (compare_results(cpu_output, gpu_output, size)) {
        std::cout << "CPU and GPU results match!" << std::endl;
    } else {
        std::cerr << "CPU and GPU results do not match!" << std::endl;
    }
    // cout << endl;
    // cout << "input" << endl;
    // for(int i=0;i<size;i++){
    //     cout << input[i] << " ";
    // }
    // cout << endl;
    // cout << "CPU" << endl;
    // for(int i=0;i<size;i++){
    //     cout << cpu_output[i] << " ";
    // }
    // cout << endl;
    // // cout << "GPU" << endl;
    // for(int i=0;i<size;i++){
    //     cout << gpu_output[i] << " ";
    // }
    cout << endl;
    cout<<size<<endl;

    // Free GPU memory

    cudaFree(d_input);
    cudaFree(d_output);

    // Free CPU memory
    delete[] input;
    delete[] cpu_output;
    delete[] gpu_output;

    return 0;
}