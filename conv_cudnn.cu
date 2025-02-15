/**
 * conv_cudnn.cu
 *
 * This program benchmarks a simplified 2D convolution for a single channel with no
 * padding and a stride of 1. Three implementations are compared:
 *
 *   1. CPU Convolution: A simple nested-loop implementation.
 *   2. Naïve GPU Convolution: A custom CUDA kernel (each thread computes one output element).
 *   3. cuDNN Convolution: Uses cuDNN's optimized convolution routine.
 *
 * For an input image of size H x W and a filter of size R x S, the output dimensions are:
 *       outH = H - R + 1,    outW = W - S + 1.
 *
 * A theoretical minimum runtime is computed based on the total number of floating‑point
 * operations (2 FLOPs per multiply–accumulate) divided by an assumed peak of 9 TFLOPs.
 *
 * Results are written to "results.md" in Markdown format.
 *
 * To compile using the provided Makefile, run:
 *     make
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>

//-----------------------------------------------------------------------------
// Error-checking macros for CUDA and cuDNN calls.
#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " - " << cudaGetErrorString(err) << std::endl;          \
            exit(err);                                                           \
        }                                                                        \
    }

#define CHECK_CUDNN(call)                                                        \
    {                                                                            \
        cudnnStatus_t status = call;                                             \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudnnGetErrorString(status) << std::endl;       \
            exit(status);                                                        \
        }                                                                        \
    }

//-----------------------------------------------------------------------------
// CPU Convolution Implementation (Single Channel, No Padding, Stride=1)
//
// Input:  image of dimensions H x W stored in a 1D array (row-major)
// Filter: kernel of dimensions R x S stored in a 1D array (row-major)
// Output: convolved output of dimensions outH x outW, where outH = H - R + 1, outW = W - S + 1.
void conv2d_cpu(const float* input, const float* filter, float* output,
                int H, int W, int R, int S,
                int outH, int outW)
{
    for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
            float sum = 0.0f;
            for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                    int ih = oh + r;
                    int iw = ow + s;
                    int input_idx = ih * W + iw;
                    int filter_idx = r * S + s;
                    sum += input[input_idx] * filter[filter_idx];
                }
            }
            output[oh * outW + ow] = sum;
        }
    }
}

//-----------------------------------------------------------------------------
// Naïve GPU Convolution Kernel (Single Channel, No Padding, Stride=1)
//
// Each thread computes one output element.
__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ filter,
                                    float* output,
                                    int H, int W, int R, int S,
                                    int outH, int outW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (oh < outH && ow < outW) {
        float sum = 0.0f;
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int ih = oh + r;
                int iw = ow + s;
                int input_idx = ih * W + iw;
                int filter_idx = r * S + s;
                sum += input[input_idx] * filter[filter_idx];
            }
        }
        output[oh * outW + ow] = sum;
    }
}

//-----------------------------------------------------------------------------
// Main function
int main()
{
    // 1. Define convolution parameters.
    // Input image: H x W; Filter: R x S; No padding, stride=1.
    const int H = 1024;  // Input height
    const int W = 1024;  // Input width
    const int R = 3;    // Kernel height
    const int S = 3;    // Kernel width
    const int outH = H - R + 1;
    const int outW = W - S + 1;

    // 2. Allocate host memory.
    const int inputSize = H * W;
    const int filterSize = R * S;
    const int outputSize = outH * outW;
    float* h_input         = new float[inputSize];
    float* h_filter        = new float[filterSize];
    float* h_output_cpu    = new float[outputSize];
    float* h_output_gpu    = new float[outputSize];
    float* h_output_cudnn  = new float[outputSize];

    // Initialize input and filter with random values.
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < filterSize; i++) {
        h_filter[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. CPU Convolution Benchmark.
    std::cout << "Running CPU convolution..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv2d_cpu(h_input, h_filter, h_output_cpu, H, W, R, S, outH, outW);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU convolution time: " << cpu_time_ms << " ms" << std::endl;

    // 4. Naïve GPU Convolution Benchmark.
    std::cout << "Running naïve GPU convolution..." << std::endl;
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter, filterSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filterSize * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start_naive, stop_naive;
    CHECK_CUDA(cudaEventCreate(&start_naive));
    CHECK_CUDA(cudaEventCreate(&stop_naive));
    CHECK_CUDA(cudaEventRecord(start_naive));

    dim3 block(32, 32);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    conv2d_naive_kernel<<<grid, block>>>(d_input, d_filter, d_output, H, W, R, S, outH, outW);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_naive));
    CHECK_CUDA(cudaEventSynchronize(stop_naive));
    float gpu_naive_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_naive_time_ms, start_naive, stop_naive));
    std::cout << "Naïve GPU convolution time: " << gpu_naive_time_ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. cuDNN Convolution Benchmark.
    std::cout << "Running cuDNN convolution..." << std::endl;
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors in NCHW format.
    cudnnTensorDescriptor_t in_desc, out_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    // Input is (1, 1, H, W)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, H, W));
    // Output is (1, 1, outH, outW)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, outH, outW));

    // Create filter descriptor.
    cudnnFilterDescriptor_t filt_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filt_desc));
    // Filter is (out_channels=1, in_channels=1, R, S)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, R, S));

    // Create convolution descriptor: no padding, stride=1, dilation=1.
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                0, 0,  // padding height, width
                                                1, 1,  // stride height, width
                                                1, 1,  // dilation height, width
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Choose convolution algorithm.
    cudnnConvolutionFwdAlgo_t algo;
    #ifdef CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
        // Newer cuDNN versions define this macro and function.
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn,
            in_desc,
            filt_desc,
            conv_desc,
            out_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,  // no memory limit
            &algo));
    #else
        // For older cuDNN versions, use the default implicit GEMM algorithm.
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    #endif

    // Determine workspace size.
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        algo,
        &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));

    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start_cudnn, stop_cudnn;
    CHECK_CUDA(cudaEventCreate(&start_cudnn));
    CHECK_CUDA(cudaEventCreate(&stop_cudnn));
    CHECK_CUDA(cudaEventRecord(start_cudnn));

    // Run cuDNN convolution (single iteration).
    CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        in_desc,
                                        d_input,
                                        filt_desc,
                                        d_filter,
                                        conv_desc,
                                        algo,
                                        d_workspace,
                                        workspace_bytes,
                                        &beta,
                                        out_desc,
                                        d_output));

    CHECK_CUDA(cudaEventRecord(stop_cudnn));
    CHECK_CUDA(cudaEventSynchronize(stop_cudnn));
    float cudnn_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cudnn_time_ms, start_cudnn, stop_cudnn));
    std::cout << "cuDNN convolution time: " << cudnn_time_ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Compute Theoretical Minimum Time.
    // Each output element uses (R*S) MACs at 2 FLOPs per MAC.
    const double total_ops = 2.0 * outputSize * (R * S);
    const double GPU_PEAK_FLOPS = 9e12; // 9 TFLOPs for RTX 3050
    double theoretical_time_ms = (total_ops / GPU_PEAK_FLOPS) * 1000.0;

    // 7. Write the results to a Markdown file.
    std::ofstream md_file("results.md");
    if (!md_file.is_open()) {
        std::cerr << "Error opening results.md for writing." << std::endl;
        exit(EXIT_FAILURE);
    }
    md_file << "# Conv2D (Single Channel, No Padding, Stride=1) Runtime Comparison\n\n";
    md_file << "| Method               | Time (ms) |\n";
    md_file << "|----------------------|-----------|\n";
    md_file << "| CPU                  | " << cpu_time_ms << " |\n";
    md_file << "| Naïve GPU Kernel     | " << gpu_naive_time_ms << " |\n";
    md_file << "| cuDNN                | " << cudnn_time_ms << " |\n";
    md_file << "| Theoretical Minimum  | " << theoretical_time_ms << " |\n";
    md_file.close();
    std::cout << "Results written to results.md" << std::endl;

    // 8. Cleanup.
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    if (d_workspace)
        CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaEventDestroy(start_naive));
    CHECK_CUDA(cudaEventDestroy(stop_naive));
    CHECK_CUDA(cudaEventDestroy(start_cudnn));
    CHECK_CUDA(cudaEventDestroy(stop_cudnn));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filt_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    delete[] h_input;
    delete[] h_filter;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_output_cudnn;

    return 0;
}
