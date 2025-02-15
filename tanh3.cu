#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>

#define TILE_SIZE 16  // Block size for tiling (Step 4: Memory Optimization)

// Macro to check CUDA API errors
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// ---------------------------------------------------------------------
// Unified tanh function for both host and device.
// This is our elementary operation (activation) that we’ll optimize.
// ---------------------------------------------------------------------
__host__ __device__ float tanh_function(float x) {
    return (expf(2.0f * x) - 1.0f) / (expf(2.0f * x) + 1.0f);
}

// ---------------------------------------------------------------------
// Step 2: Baseline CPU Implementation for Correctness & Timing.
// ---------------------------------------------------------------------
void tanh_cpu(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        output[i] = tanh_function(input[i]);
    }
}

// ---------------------------------------------------------------------
// Step 2: Naïve GPU Kernel (Baseline)
// Each thread processes one element without any memory optimizations.
// ---------------------------------------------------------------------
__global__ void tanh_naive(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    
    if (row < rows && col < cols) {
        output[idx] = tanh_function(input[idx]);
    }
}

// ---------------------------------------------------------------------
// Advanced Optimized Kernel (Final Version)
// This kernel implements shared memory tiling to reduce global memory
// accesses and is structured for later inclusion of warp-level
// optimizations (Steps 4, 5, and 7).
//
// The kernel does the following:
// 1. Loads a tile of input data into shared memory (with padding to avoid bank conflicts).
// 2. Computes the tanh activation on data in shared memory.
// 3. Writes the result back to global memory.
// ---------------------------------------------------------------------
__global__ void tanh_optimized_final(float* input, float* output, int rows, int cols) {
    // Shared memory tile with extra column to avoid bank conflicts.
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Compute global indices.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    // Step 1: Load input data into shared memory (if within bounds).
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[index];
    }
    __syncthreads();

    // Step 2: Compute the tanh activation in shared memory.
    // (This is a good spot for potential warp-level optimizations if needed.)
    if (row < rows && col < cols) {
        float val = tile[threadIdx.y][threadIdx.x];
        float result = tanh_function(val);
        tile[threadIdx.y][threadIdx.x] = result;
    }
    __syncthreads();

    // Step 3: Write the results back to global memory.
    if (row < rows && col < cols) {
        output[index] = tile[threadIdx.y][threadIdx.x];
    }
}

// ---------------------------------------------------------------------
// Main function demonstrating the full development workflow.
// It sets up the CPU and GPU versions, uses CUDA streams to allow
// asynchronous memory copies (Step 6), and includes result verification
// and documentation (writing to a markdown file).
// ---------------------------------------------------------------------
int main() {
    // Problem dimensions.
    int rows = 1024;
    int cols = 1024;
    size_t size = rows * cols * sizeof(float);

    // Allocate host memory.
    float* h_input = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);

    // Initialize input data (random values in [-1,1]).
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = (rand() % 2000 - 1000) / 1000.0f;
    }

    // -----------------------------------------------------------------
    // Step 2: Run and time the CPU version.
    // -----------------------------------------------------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    tanh_cpu(h_input, h_output_cpu, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("CPU execution time: %.5f ms\n", cpu_time);

    // -----------------------------------------------------------------
    // Allocate device memory.
    // -----------------------------------------------------------------
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // -----------------------------------------------------------------
    // Step 6: Create a CUDA stream for asynchronous execution.
    // This allows overlapping of memory transfers with computation.
    // -----------------------------------------------------------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy input data from host to device.
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream));

    // Set up kernel launch configuration.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    // -----------------------------------------------------------------
    // Step 3: Launch the naïve kernel and profile it.
    // -----------------------------------------------------------------
    cudaEvent_t start_naive, stop_naive;
    CUDA_CHECK(cudaEventCreate(&start_naive));
    CUDA_CHECK(cudaEventCreate(&stop_naive));
    CUDA_CHECK(cudaEventRecord(start_naive, stream));
    
    tanh_naive<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, rows, cols);
    
    CUDA_CHECK(cudaEventRecord(stop_naive, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_naive));
    
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start_naive, stop_naive));
    printf("Naïve GPU execution time: %.5f ms\n", naive_time);

    // -----------------------------------------------------------------
    // Step 4 & 7: Launch the optimized kernel.
    // This kernel includes shared memory tiling and is structured to allow
    // for additional optimizations (e.g., warp-level primitives, kernel fusion).
    // -----------------------------------------------------------------
    cudaEvent_t start_opt, stop_opt;
    CUDA_CHECK(cudaEventCreate(&start_opt));
    CUDA_CHECK(cudaEventCreate(&stop_opt));
    CUDA_CHECK(cudaEventRecord(start_opt, stream));
    
    tanh_optimized_final<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, rows, cols);
    
    CUDA_CHECK(cudaEventRecord(stop_opt, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_opt));
    
    float optimized_time;
    CUDA_CHECK(cudaEventElapsedTime(&optimized_time, start_opt, stop_opt));
    printf("Optimized GPU execution time: %.5f ms\n", optimized_time);

    // Asynchronously copy the GPU results back to host.
    CUDA_CHECK(cudaMemcpyAsync(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // -----------------------------------------------------------------
    // Step 11: Final Testing & Verification.
    // Compare the GPU results with the CPU version to ensure correctness.
    // -----------------------------------------------------------------
    bool correct = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_output_cpu[i], h_output_gpu[i]);
            break;
        }
    }
    printf("Results verification: %s\n", correct ? "PASS" : "FAIL");

    // -----------------------------------------------------------------
    // Theoretical Performance Calculations (Step 1)
    // -----------------------------------------------------------------
    // We assume that each tanh operation requires roughly 5 FLOPs:
    // 2 for expf(2*x), 1 subtraction, 1 addition, and 1 division.
    const float ops_per_tanh = 5.0f;
    double total_ops = static_cast<double>(rows) * cols * ops_per_tanh;
    // For an RTX 3050, assume a peak FP32 performance of 9.0 TFLOPs:
    double gpu_peak_flops = 9.0e12; // 9.0 TFLOPs in FLOPs per second.
    double theoretical_time_sec = total_ops / gpu_peak_flops;
    double theoretical_time_ms = theoretical_time_sec * 1e3;
    
    // -----------------------------------------------------------------
    // Step 11: Writing detailed results to a markdown file.
    // This includes our measured timings as well as theoretical limits.
    // -----------------------------------------------------------------
    std::ofstream result_file("results.md", std::ios::app);
    result_file << "\n## Tanh Activation Performance Benchmark\n\n";
    result_file << "### Kernel Execution Times\n";
    result_file << "| Rows | Cols | CPU Time (ms) | Naïve GPU Time (ms) | Optimized GPU Time (ms) | Theoretical Min Time (ms) | Speedup (CPU/Optimized) |\n";
    result_file << "|------|------|---------------|---------------------|-------------------------|---------------------------|-------------------------|\n";
    result_file << "| " << rows << " | " << cols << " | " << cpu_time 
                << " | " << naive_time 
                << " | " << optimized_time 
                << " | " << theoretical_time_ms 
                << " | " << (cpu_time / optimized_time) << " |\n";
    result_file << "\n**Theoretical Details:**\n\n";
    result_file << "- Assumed FLOPs per tanh operation: " << ops_per_tanh << "\n";
    result_file << "- Total operations: " << total_ops << " FLOPs\n";
    result_file << "- RTX 3050 assumed peak performance: 9.0 TFLOPs (9.0e12 FLOPs/sec)\n";
    result_file << "- Theoretical minimum execution time: " << theoretical_time_ms << " ms\n";
    result_file.close();

    // -----------------------------------------------------------------
    // Cleanup: Destroy CUDA events, stream, and free allocated memory.
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start_naive));
    CUDA_CHECK(cudaEventDestroy(stop_naive));
    CUDA_CHECK(cudaEventDestroy(start_opt));
    CUDA_CHECK(cudaEventDestroy(stop_opt));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}
