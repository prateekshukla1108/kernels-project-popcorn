/*
 * Max Pooling Example with Markdown Report Generation (CPU vs GPU)
 *
 * This program demonstrates a basic max–pooling operation (forward pass) on a 
 * single–channel image using both a CPU and a naïve GPU (CUDA) implementation.
 *
 * The max pooling is implemented with a 2x2 window and a stride of 2.
 * Each output pixel is computed as the maximum value in the corresponding 
 * 2x2 region of the input image.
 *
 * The program includes:
 *   - A CPU function: maxPool2D_cpu()
 *   - A GPU kernel: maxPool2D_gpu()
 *   - Timing measurements for both implementations
 *   - A Markdown report generation that summarizes the performance
 *
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cfloat>
#include <cuda_runtime.h>

// Define the input image dimensions (using 1024x1024 for this example)
#define IMAGE_WIDTH  1024
#define IMAGE_HEIGHT 1024

// Define pooling parameters: a 2x2 pooling window with stride 2.
#define POOL_WIDTH  2
#define POOL_HEIGHT 2
#define STRIDE      2

// ---------------------------------------------------------------------
// CPU Implementation of Max Pooling
// ---------------------------------------------------------------------
/*
 * maxPool2D_cpu
 *
 * Performs 2D max pooling on a single–channel image.
 *
 * Parameters:
 *   input       - pointer to the input image array (size: width * height)
 *   output      - pointer to the output image array (size: outWidth * outHeight)
 *   width       - width of the input image
 *   height      - height of the input image
 *   poolWidth   - width of the pooling window
 *   poolHeight  - height of the pooling window
 *   stride      - stride of the pooling operation
 *
 * The function computes the output dimensions as:
 *   outWidth  = floor((width - poolWidth) / stride) + 1
 *   outHeight = floor((height - poolHeight) / stride) + 1
 *
 * For each output pixel, the pooling window is applied and the maximum value
 * within that window is computed.
 */
void maxPool2D_cpu(const float* input, float* output, int width, int height,
                   int poolWidth, int poolHeight, int stride) {
    // Compute the output dimensions.
    int outWidth  = (width  - poolWidth) / stride + 1;
    int outHeight = (height - poolHeight) / stride + 1;
    
    // Loop over every output pixel.
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            float maxVal = -FLT_MAX; // Initialize to smallest possible float.
            // For each pooling window element.
            for (int py = 0; py < poolHeight; py++) {
                for (int px = 0; px < poolWidth; px++) {
                    // Compute the corresponding input coordinate.
                    int ix = x * stride + px;
                    int iy = y * stride + py;
                    // Since we assume valid pooling (no padding), bounds check is not required.
                    float currVal = input[iy * width + ix];
                    if (currVal > maxVal)
                        maxVal = currVal;
                }
            }
            // Store the maximum value for this output pixel.
            output[y * outWidth + x] = maxVal;
        }
    }
}

// ---------------------------------------------------------------------
// GPU (CUDA) Implementation of Max Pooling
// ---------------------------------------------------------------------
__global__ void maxPool2D_gpu(const float* input, float* output, int width, int height,
                              int poolWidth, int poolHeight, int stride) {
    // Calculate output coordinates (each thread computes one output pixel).
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the output dimensions.
    int outWidth  = (width  - poolWidth) / stride + 1;
    int outHeight = (height - poolHeight) / stride + 1;

    if (out_x < outWidth && out_y < outHeight) {
        float maxVal = -FLT_MAX; // Initialize to a very small value.
        // Loop over the pooling window.
        for (int py = 0; py < poolHeight; py++) {
            for (int px = 0; px < poolWidth; px++) {
                int ix = out_x * stride + px;
                int iy = out_y * stride + py;
                float currVal = input[iy * width + ix];
                if (currVal > maxVal)
                    maxVal = currVal;
            }
        }
        // Write the maximum value to the output.
        output[out_y * outWidth + out_x] = maxVal;
    }
}

// ---------------------------------------------------------------------
// Main Function: Setup, Execution, and Markdown Report Generation
// ---------------------------------------------------------------------
int main() {
    // Set input image dimensions.
    const int width  = IMAGE_WIDTH;
    const int height = IMAGE_HEIGHT;
    const int imageSize = width * height;
    
    // Define pooling parameters.
    const int poolWidth = POOL_WIDTH;
    const int poolHeight = POOL_HEIGHT;
    const int stride = STRIDE;
    
    // Compute output dimensions.
    int outWidth  = (width  - poolWidth) / stride + 1;
    int outHeight = (height - poolHeight) / stride + 1;
    int outSize = outWidth * outHeight;

    // Allocate host memory for input and output images.
    float* h_input       = new float[imageSize];
    float* h_output_cpu  = new float[outSize];
    float* h_output_gpu  = new float[outSize];

    // Initialize the input image with random values in [0, 1].
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ----------------------------
    // CPU Max Pooling and Timing
    // ----------------------------
    std::cout << "Running CPU max pooling..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    maxPool2D_cpu(h_input, h_output_cpu, width, height, poolWidth, poolHeight, stride);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    double cpuTimeMs = cpu_duration.count();
    std::cout << "CPU max pooling took " << cpuTimeMs << " ms" << std::endl;

    // ------------------------------------
    // GPU Max Pooling: Memory Allocation & Data Transfer
    // ------------------------------------
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input,  imageSize * sizeof(float));
    cudaMalloc((void**)&d_output, outSize * sizeof(float));

    cudaMemcpy(d_input, h_input, imageSize * sizeof(float), cudaMemcpyHostToDevice);

    // ----------------------------
    // Launching the GPU Kernel for Max Pooling
    // ----------------------------
    dim3 blockSize(16,16);
    dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x,
                  (outHeight + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running GPU max pooling (naïve)..." << std::endl;
    cudaEventRecord(start);
    maxPool2D_gpu<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                             poolWidth, poolHeight, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTimeMs = 0.0f;
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    std::cout << "GPU max pooling (naïve) took " << gpuTimeMs << " ms" << std::endl;

    // Copy the GPU result back to host.
    cudaMemcpy(h_output_gpu, d_output, outSize * sizeof(float), cudaMemcpyDeviceToHost);

    // ----------------------------
    // Validate and Compare CPU and GPU Results
    // ----------------------------
    int errorCount = 0;
    for (int i = 0; i < outSize; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                std::cout << "Mismatch at index " << i << ": CPU = " 
                          << h_output_cpu[i] << ", GPU = " << h_output_gpu[i] << std::endl;
            }
        }
    }
    if (errorCount == 0)
        std::cout << "CPU and GPU max pooling results match." << std::endl;
    else
        std::cout << "Total mismatches: " << errorCount << std::endl;

    // ----------------------------
    // Theoretical Minimum Time Calculation
    // ----------------------------
    // For each output pixel in max pooling, we perform (poolWidth * poolHeight - 1) comparisons.
    // For a 2x2 window, that's 3 comparisons per output pixel.
    double totalComparisons = static_cast<double>(outSize) * 3.0;
    // Assume a hypothetical GPU peak rate for comparisons, e.g., 9.0e12 ops/sec (same as before).
    double theoreticalTimeMs = (totalComparisons / 9.0e12) * 1000.0;

    // ----------------------------
    // Speedup Calculation
    // ----------------------------
    double speedup = (gpuTimeMs > 0) ? (cpuTimeMs / gpuTimeMs) : 0.0;

    // ----------------------------
    // Markdown Report Generation for Max Pooling Results
    // ----------------------------
    std::ofstream mdFile("results_maxpool.md");
    if (!mdFile) {
        std::cerr << "Error opening results_maxpool.md for writing." << std::endl;
    } else {
        mdFile << "# Max Pooling Kernel Results\n\n";
        mdFile << "This document summarizes the performance results for a max pooling (forward pass) operation on a " 
               << width << "×" << height << " single–channel image using a " 
               << poolWidth << "×" << poolHeight << " pooling window with stride " << stride << ".\n\n";
        mdFile << "### Kernel Execution Times\n\n";
        mdFile << "| Rows | Cols | CPU Time (ms) | Naïve GPU Time (ms) | Theoretical Min Time (ms) | Speedup (CPU/Naïve) |\n";
        mdFile << "|------|------|---------------|---------------------|---------------------------|---------------------|\n";
        mdFile << "| " << outHeight << " | " << outWidth << " | " 
               << cpuTimeMs << " | " << gpuTimeMs 
               << " | " << theoreticalTimeMs << " | " << speedup << " |\n\n";
        mdFile << "### Theoretical Details\n\n";
        mdFile << "- **Comparisons per Output Pixel:** For a " << poolWidth << "×" << poolHeight << " window, "
               << "(poolWidth*poolHeight - 1) = " << (poolWidth*poolHeight - 1) << " comparisons are performed per output pixel.\n";
        mdFile << "- **Total Comparisons:** " << totalComparisons << " comparisons.\n";
        mdFile << "- **Assumed GPU Peak Performance:** 9.0e12 ops/sec.\n";
        mdFile << "- **Theoretical Minimum Execution Time:** " << theoreticalTimeMs << " ms.\n\n";
        mdFile << "### Explanation of the Max Pooling Operation\n\n";
        mdFile << "In max pooling, each output pixel is computed by applying a pooling window over the corresponding "
               << "region in the input image and taking the maximum value within that window. The pooling window slides "
               << "across the image with a specified stride. This operation helps in reducing the spatial dimensions and "
               << "extracting the most prominent features.\n";
        mdFile.close();
        std::cout << "Markdown report generated as results_maxpool.md" << std::endl;
    }

    // ----------------------------
    // Cleanup: Free Allocated Memory and Destroy CUDA Events
    // ----------------------------
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
