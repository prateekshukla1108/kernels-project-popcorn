#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream> 

// -------------------------------------------------------------------------------------------------
// Utility Macros for Error Checking
// -------------------------------------------------------------------------------------------------
#define CUDA_CHECK(err) { \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(err) { \
    if(err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// -------------------------------------------------------------------------------------------------
// CPU-based GEMM (Naïve Implementation)
// Computes C = A * B for each batch (assumes row–major layout)
// -------------------------------------------------------------------------------------------------
void cpuGemm(float **A, float **B, float **C, int M, int N, int K, int batchCount) {
    for (int b = 0; b < batchCount; b++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[b][i * K + k] * B[b][k * N + j];
                }
                C[b][i * N + j] = sum;
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Naïve GPU Kernel for GEMM (Row–Major)
// Each thread computes one element of the output matrix for one batch.
// The 3D grid's z–dimension indexes the batch.
// -------------------------------------------------------------------------------------------------
__global__ void naiveGemmKernel(const float **A, const float **B, float **C, int M, int N, int K) {
    int batch = blockIdx.z;  // Batch index from grid z–dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in output matrix
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[batch][row * K + k] * B[batch][k * N + col];
        }
        C[batch][row * N + col] = sum;
    }
}

// -------------------------------------------------------------------------------------------------
// Optimized GPU Kernel for GEMM (Row–Major)
// Incorporates: Shared Memory Tiling, Memory Coalescing, Loop Unrolling.
// Each thread computes one output element.
// -------------------------------------------------------------------------------------------------

// Define tile size for shared memory tiling (and also block dimensions)
#define TILE_SIZE 16

__global__ void optimizedGemmKernel(const float **A, const float **B, float **C, int M, int N, int K) {
    int batch = blockIdx.z;  // Batch index
    // Compute global row and column for the C matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Declare shared memory for a tile of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Loop over all tiles needed to compute C(row,col)
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;  // column index for A tile
        int tiledRow = t * TILE_SIZE + threadIdx.y;    // row index for B tile
        
        // Load elements of A into shared memory (with bounds checking)
        if (row < M && tiledCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[batch][row * K + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load elements of B into shared memory (with bounds checking)
        if (tiledRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[batch][tiledRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Unroll loop over tile dimension to compute partial sums
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write the computed value to the output matrix (with bounds check)
    if (row < M && col < N)
        C[batch][row * N + col] = sum;
}

// -------------------------------------------------------------------------------------------------
// Helper function to verify that two matrices (for one batch) are nearly equal.
// Returns the maximum absolute difference.
// -------------------------------------------------------------------------------------------------
float verifyMatrix(const float *ref, const float *test, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabs(ref[i] - test[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

// -------------------------------------------------------------------------------------------------
// Main Function: Runs CPU GEMM, Naive GPU GEMM, Optimized GPU GEMM, and cuBLAS Batched GEMM,
// verifies correctness, measures runtimes, and writes a Markdown report containing only the runtimes.
// -------------------------------------------------------------------------------------------------
int main() {
    // --------------------------------------------------------------------------------------------
    // Problem Setup (MNIST Fully–Connected Layer Example)
    // --------------------------------------------------------------------------------------------
    // Each MNIST image (28x28) is flattened to 784 elements.
    // Assume a fully–connected layer with 256 neurons and a mini–batch size of 128.
    // "batchCount" represents the number of independent GEMM operations processed.
    int M = 256;       // Number of neurons (output features)
    int K = 784;       // Input feature size (28x28 flattened)
    int N = 128;       // Mini–batch size (number of images processed concurrently)
    int batchCount = 10;  // Number of batched operations

    // --------------------------------------------------------------------------------------------
    // GeForce RTX 3050 Hardware Approximate Specs:
    // - Peak FP32 Performance: ~9.1 TFLOPS
    // - Memory Bandwidth: ~224 GB/s
    // For each GEMM: FLOPs ≈ 2 * M * K * N, which is ~51.38e6 FLOPs in our example.
    // Data Transfer ≈ 1.335 MB and Computational Intensity ≈ 38.5 FLOPs/byte (compute-bound).
    
    // --------------------------------------------------------------------------------------------
    // Allocate Host Memory for Each Batch for Inputs (A, B) and Outputs (C)
    // --------------------------------------------------------------------------------------------
    float **h_A = (float**)malloc(batchCount * sizeof(float*));          // Weight matrices
    float **h_B = (float**)malloc(batchCount * sizeof(float*));          // Input activations
    float **h_C_cpu = (float**)malloc(batchCount * sizeof(float*));      // CPU GEMM result
    float **h_C_gpu = (float**)malloc(batchCount * sizeof(float*));      // Naive GPU kernel result
    float **h_C_cublas = (float**)malloc(batchCount * sizeof(float*));   // cuBLAS result
    float **h_C_optimized = (float**)malloc(batchCount * sizeof(float*)); // Optimized GPU kernel result

    for (int b = 0; b < batchCount; b++) {
        h_A[b] = (float*)malloc(M * K * sizeof(float));
        h_B[b] = (float*)malloc(K * N * sizeof(float));
        h_C_cpu[b] = (float*)malloc(M * N * sizeof(float));
        h_C_gpu[b] = (float*)malloc(M * N * sizeof(float));
        h_C_cublas[b] = (float*)malloc(M * N * sizeof(float));
        h_C_optimized[b] = (float*)malloc(M * N * sizeof(float));
        // Initialize matrices with ones.
        for (int i = 0; i < M * K; i++) h_A[b][i] = 1.0f;
        for (int i = 0; i < K * N; i++) h_B[b][i] = 1.0f;
    }

    // --------------------------------------------------------------------------------------------
    // 1. Run CPU GEMM and Time It
    // --------------------------------------------------------------------------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuGemm(h_A, h_B, h_C_cpu, M, N, K, batchCount);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("CPU GEMM time: %f ms\n", cpu_time);

    // --------------------------------------------------------------------------------------------
    // Allocate Device Memory for Each Batch for GPU Implementations
    // --------------------------------------------------------------------------------------------
    float **d_A = (float**)malloc(batchCount * sizeof(float*));
    float **d_B = (float**)malloc(batchCount * sizeof(float*));
    float **d_C = (float**)malloc(batchCount * sizeof(float*));
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMalloc(&d_A[b], M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[b], K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C[b], M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A[b], h_A[b], M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[b], h_B[b], K * N * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Create device arrays of pointers for use by GPU kernels.
    float **d_A_array, **d_B_array, **d_C_array;
    CUDA_CHECK(cudaMalloc(&d_A_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_B_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_C_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_A_array, d_A, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_array, d_B, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_array, d_C, batchCount * sizeof(float*), cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------------------------
    // 2. Run Naive GPU Kernel and Time It
    // --------------------------------------------------------------------------------------------
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y, 
                 batchCount);
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    CUDA_CHECK(cudaEventRecord(start_gpu, 0));
    naiveGemmKernel<<<gridDim, blockDim>>>( (const float **)d_A_array, (const float **)d_B_array, d_C_array, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop_gpu, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));
    float gpu_naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_naive_time, start_gpu, stop_gpu));
    printf("Naive GPU Kernel GEMM time: %f ms\n", gpu_naive_time);
    // Copy the result of the naive GPU kernel back to host.
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu[b], d_C[b], M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // --------------------------------------------------------------------------------------------
    // 3. Run cuBLAS Batched GEMM and Time It
    // --------------------------------------------------------------------------------------------
    // Reset device output matrices to zero.
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMemset(d_C[b], 0, M * N * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_C_array, d_C, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaEvent_t start_cublas, stop_cublas;
    CUDA_CHECK(cudaEventCreate(&start_cublas));
    CUDA_CHECK(cudaEventCreate(&stop_cublas));
    CUDA_CHECK(cudaEventRecord(start_cublas, 0));
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        (const float**)d_A_array, M,
        (const float**)d_B_array, K,
        &beta,
        d_C_array, M,
        batchCount
    ));
    CUDA_CHECK(cudaEventRecord(stop_cublas, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_cublas));
    float cublas_time;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_time, start_cublas, stop_cublas));
    printf("cuBLAS Batched GEMM time: %f ms\n", cublas_time);
    // Copy the cuBLAS result back to host.
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMemcpy(h_C_cublas[b], d_C[b], M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // --------------------------------------------------------------------------------------------
    // 4. Run Optimized GPU Kernel (Shared Memory, Tiling, Loop Unrolling) and Time It
    // --------------------------------------------------------------------------------------------
    // Reset device output matrices to zero.
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMemset(d_C[b], 0, M * N * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_C_array, d_C, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    cudaEvent_t start_opt, stop_opt;
    CUDA_CHECK(cudaEventCreate(&start_opt));
    CUDA_CHECK(cudaEventCreate(&stop_opt));
    CUDA_CHECK(cudaEventRecord(start_opt, 0));
    optimizedGemmKernel<<<gridDim, blockDim>>>( (const float **)d_A_array, (const float **)d_B_array, d_C_array, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop_opt, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_opt));
    float gpu_opt_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_opt_time, start_opt, stop_opt));
    printf("Optimized GPU Kernel GEMM time: %f ms\n", gpu_opt_time);
    // Copy the optimized kernel result back to host.
    for (int b = 0; b < batchCount; b++) {
        CUDA_CHECK(cudaMemcpy(h_C_optimized[b], d_C[b], M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // --------------------------------------------------------------------------------------------
    // Verify Correctness (Mandatory)
    // Compare CPU results with those from the naive GPU kernel, cuBLAS, and optimized kernel.
    // --------------------------------------------------------------------------------------------
    const float tol = 1e-5;
    for (int b = 0; b < batchCount; b++) {
        int size = M * N;
        float diff_gpu = verifyMatrix(h_C_cpu[b], h_C_gpu[b], size);
        float diff_cublas = verifyMatrix(h_C_cpu[b], h_C_cublas[b], size);
        float diff_opt   = verifyMatrix(h_C_cpu[b], h_C_optimized[b], size);
        if (diff_gpu > tol || diff_cublas > tol || diff_opt > tol) {
            printf("Verification FAILED on batch %d: max diff GPU = %f, cuBLAS = %f, optimized = %f\n", 
                   b, diff_gpu, diff_cublas, diff_opt);
            exit(EXIT_FAILURE);
        }
    }
    printf("Verification PASSED: All results match within tolerance %f.\n", tol);

    // --------------------------------------------------------------------------------------------
    // Generate Markdown Report ("bmm_report.md") containing only the runtime numbers.
    // --------------------------------------------------------------------------------------------
    std::ofstream report("bmm_report.md");
    if (!report.is_open()) {
        printf("Failed to open report file for writing.\n");
        exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << "# Batched Matrix Multiplication Runtimes\n\n";
    ss << "- **CPU GEMM Time:** " << cpu_time << " ms\n";
    ss << "- **Naive GPU Kernel GEMM Time:** " << gpu_naive_time << " ms\n";
    ss << "- **Optimized GPU Kernel GEMM Time:** " << gpu_opt_time << " ms\n";
    ss << "- **cuBLAS Batched GEMM Time:** " << cublas_time << " ms\n";
    report << ss.str();
    report.close();
    printf("Markdown report generated: bmm_report.md\n");

    // --------------------------------------------------------------------------------------------
    // Cleanup: Free host and device memory, destroy handles and CUDA events.
    // --------------------------------------------------------------------------------------------
    for (int b = 0; b < batchCount; b++) {
        free(h_A[b]); free(h_B[b]); free(h_C_cpu[b]); free(h_C_gpu[b]); free(h_C_cublas[b]); free(h_C_optimized[b]);
        CUDA_CHECK(cudaFree(d_A[b])); CUDA_CHECK(cudaFree(d_B[b])); CUDA_CHECK(cudaFree(d_C[b]));
    }
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu); free(h_C_cublas); free(h_C_optimized);
    free(d_A); free(d_B); free(d_C);
    CUDA_CHECK(cudaFree(d_A_array)); CUDA_CHECK(cudaFree(d_B_array)); CUDA_CHECK(cudaFree(d_C_array));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start_gpu)); CUDA_CHECK(cudaEventDestroy(stop_gpu));
    CUDA_CHECK(cudaEventDestroy(start_cublas)); CUDA_CHECK(cudaEventDestroy(stop_cublas));
    CUDA_CHECK(cudaEventDestroy(start_opt)); CUDA_CHECK(cudaEventDestroy(stop_opt));

    return 0;
}
