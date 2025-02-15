#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cudnn.h>      // For cuDNN activation
#include <cstdio>       // For popen
#include <string>
#include <cstdlib>

// Tiling block size (for shared memory optimization)
#define TILE_SIZE 16

// Macro to check CUDA API errors
#define CUDA_CHECK(err) { \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// ---------------------------------------------------------------------
// Unified tanh function for both host and device.
// ---------------------------------------------------------------------
__host__ __device__ float tanh_function(float x) {
    return (expf(2.0f * x) - 1.0f) / (expf(2.0f * x) + 1.0f);
}

// ---------------------------------------------------------------------
// CPU version of tanh activation.
// ---------------------------------------------------------------------
void tanh_cpu(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        output[i] = tanh_function(input[i]);
    }
}

// ---------------------------------------------------------------------
// Naïve GPU Kernel (Baseline)
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
// Optimized GPU Kernel using Shared Memory Tiling.
// ---------------------------------------------------------------------
__global__ void tanh_optimized_final(float* input, float* output, int rows, int cols) {
    // Shared memory tile with extra column to help avoid bank conflicts.
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[index];
    }
    __syncthreads();

    if (row < rows && col < cols) {
        float val = tile[threadIdx.y][threadIdx.x];
        tile[threadIdx.y][threadIdx.x] = tanh_function(val);
    }
    __syncthreads();

    if (row < rows && col < cols) {
        output[index] = tile[threadIdx.y][threadIdx.x];
    }
}

// ---------------------------------------------------------------------
// Benchmark cuDNN Tanh Activation using cudnnActivationForward.
// The input is interpreted as a 4D tensor with shape (1,1,rows,cols).
// ---------------------------------------------------------------------
float benchmark_cudnn_tanh(float* d_input, float* d_output, int rows, int cols, cudaStream_t stream) {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t inDesc, outDesc;
    cudnnActivationDescriptor_t actDesc;

    // Create and set tensor descriptors.
    cudnnCreate(&handle);
    cudnnSetStream(handle, stream);
    cudnnCreateTensorDescriptor(&inDesc);
    cudnnCreateTensorDescriptor(&outDesc);
    cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols);
    cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols);

    // Create and set activation descriptor for tanh.
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0f);


    // Setup timing using CUDA events.
    // cudaEvent_t start_cudnn, stop_cudnn;
    // CUDA_CHECK(cudaEventCreate(&start_cudnn));
    // CUDA_CHECK(cudaEventCreate(&stop_cudnn));
    // CUDA_CHECK(cudaEventRecord(start_cudnn, stream));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Start timing using std::chrono
    cudaDeviceSynchronize(); // Ensure stream is empty before timing
    auto cudnn_start = std::chrono::high_resolution_clock::now();

    // Add a loop for cuDNN warm-up
    for (int i = 0; i < 10; i++) {
        cudnnActivationForward(handle, actDesc, &alpha, inDesc, d_input, &beta, outDesc, d_output);
    }
    cudaDeviceSynchronize();  // Ensure warm-up completes before timing

    auto cudnn_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cudnn_duration = cudnn_end - cudnn_start;
    float cudnn_time = cudnn_duration.count();

    // Cleanup cuDNN resources.
    // CUDA_CHECK(cudaEventDestroy(start_cudnn));
    // CUDA_CHECK(cudaEventDestroy(stop_cudnn));
    cudnnDestroyActivationDescriptor(actDesc);
    cudnnDestroyTensorDescriptor(inDesc);
    cudnnDestroyTensorDescriptor(outDesc);
    cudnnDestroy(handle);

    return cudnn_time;
}

// ---------------------------------------------------------------------
// Helper function to run an external command and capture its output.
// The command should output a single number (the runtime in ms).
// ---------------------------------------------------------------------
// double get_runtime_from_command(const std::string& cmd) {
//     FILE* pipe = popen(cmd.c_str(), "r");
//     if (!pipe) {
//         return -1;
//     }
//     char buffer[128];
//     std::string result;
//     while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
//         result += buffer;
//     }
//     pclose(pipe);
//     try {
//         return std::stod(result);
//     } catch(...) {
//         return -1;
//     }
// }

// ---------------------------------------------------------------------
// Main function.
// ---------------------------------------------------------------------
int main() {
    // Set environment variables to avoid conflicts with PyTorch and TensorFlow
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true", 1);
    setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128", 1);

    printf("Environment variables set for isolation.\n");
    // Problem dimensions.
    int rows = 1024;
    int cols = 1024;
    size_t size = rows * cols * sizeof(float);

    // Allocate host memory.
    float* h_input      = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);

    // Initialize input data (values in [-1,1]).
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = (rand() % 2000 - 1000) / 1000.0f;
    }

    // -----------------------------------------------------------------
    // CPU Benchmark
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
    // Create a CUDA stream for asynchronous execution.
    // -----------------------------------------------------------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Copy input data from host to device asynchronously.
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream));

    // Kernel launch configuration.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    // -----------------------------------------------------------------
    // Naïve GPU Kernel Benchmark
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
    // Optimized GPU Kernel Benchmark
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

    // -----------------------------------------------------------------
    // cuDNN Tanh Benchmark
    // -----------------------------------------------------------------
    printf("cuDNN Version: %d.%d\n", CUDNN_MAJOR, CUDNN_MINOR);

    float cudnn_time = benchmark_cudnn_tanh(d_input, d_output, rows, cols, stream);
    printf("cuDNN Tanh execution time: %.5f ms\n", cudnn_time);

    // -----------------------------------------------------------------
    // Asynchronously copy GPU results back to host (for verification).
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // -----------------------------------------------------------------
    // Verify results against CPU output.
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
    // Theoretical Performance Calculation.
    // -----------------------------------------------------------------
    // Assume each tanh operation uses roughly 5 FLOPs.
    const float ops_per_tanh = 5.0f;
    double total_ops = static_cast<double>(rows) * cols * ops_per_tanh;
    // For an RTX 3050, assume a peak FP32 performance of 9.0 TFLOPs.
    double gpu_peak_flops = 9.0e12; // 9.0e12 FLOPs/s.
    double theoretical_time_sec = total_ops / gpu_peak_flops;
    double theoretical_time_ms = theoretical_time_sec * 1e3;

    // -----------------------------------------------------------------
    // Benchmark external implementations (PyTorch & TensorFlow).
    // These functions call external Python scripts that must output a single
    // runtime value (in ms) to stdout.
    // -----------------------------------------------------------------
    // double pytorch_time   = get_runtime_from_command("python pytorch_tanh.py");
    // double tensorflow_time = get_runtime_from_command("python tensorflow_tanh.py");

    // -----------------------------------------------------------------
    // Write all results to a Markdown file.
    // -----------------------------------------------------------------
    std::ofstream result_file("results.md", std::ios::app);
    result_file << "\n## Tanh Activation Performance Benchmark\n\n";
    result_file << "### Kernel Execution Times\n";
    result_file << "| Rows | Cols | CPU Time (ms) | Naïve GPU Time (ms) | Optimized GPU Time (ms) | cuDNN Time (ms) | Theoretical Min Time (ms) | Speedup (CPU/Optimized) |\n";
    result_file << "|------|------|---------------|---------------------|-------------------------|-----------------|---------------------------|-------------------------|\n";
    result_file << "| " << rows << " | " << cols << " | " << cpu_time 
                << " | " << naive_time 
                << " | " << optimized_time 
                << " | " << cudnn_time
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
