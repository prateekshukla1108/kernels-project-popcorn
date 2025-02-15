/*
 *
 * This program demonstrates a basic 2D convolution (forward pass) on a single–channel
 * image using both a CPU and a naïve GPU (CUDA) implementation. After running the
 * convolution, it writes a Markdown file ("results.md") that contains the performance
 * results and theoretical minimum execution time based on an assumed RTX 3050 GPU.
 *
 * The convolution uses a 3x3 kernel with zero-padding.
 *
 * explanation for the offsets:
 * - Each output pixel is computed by placing the center of the kernel over the pixel.
 * - kHalfWidth and kHalfHeight (computed as kernel_width/2 and kernel_height/2)
 *   are used to shift the kernel indices so that the kernel is centered.
 * - Without these offsets, the kernel would be misaligned (e.g., its top-left would be
 *   at the pixel), resulting in an incorrect weighted sum.
 * 
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

// Define the image dimensions (using 1024x1024 for this example)
#define IMAGE_WIDTH  1024
#define IMAGE_HEIGHT 1024

// Define the convolution kernel (3x3 kernel)
#define KERNEL_WIDTH  3
#define KERNEL_HEIGHT 3

// ---------------------------------------------------------------------
// CPU Implementation of 2D Convolution
// ---------------------------------------------------------------------
/*
 * conv2D_cpu
 *
 * Performs a 2D convolution on a single–channel image.
 *
 * Parameters:
 *   input   - pointer to the input image array (size: width * height)
 *   output  - pointer to the output image array (size: width * height)
 *   width   - width of the image
 *   height  - height of the image
 *   kernel  - pointer to the convolution kernel array (size: kWidth * kHeight)
 *   kWidth  - width of the convolution kernel
 *   kHeight - height of the convolution kernel
 *
 * The function applies zero-padding: if the kernel extends beyond the image
 * boundaries, those positions are treated as zero.
 *
 * Explanation for kernel offsets:
 *   kHalfWidth and kHalfHeight (kernel_width/2 and kernel_height/2) are used to
 *   center the kernel on the current pixel. This ensures that the kernel's center
 *   aligns with the pixel being processed. Without these offsets, the convolution
 *   would incorrectly apply the kernel (e.g., with its top-left corner on the pixel),
 *   leading to misalignment of weights and erroneous output.
 */
void conv2D_cpu(const float* input, float* output, int width, int height,
                const float* kernel, int kWidth, int kHeight) {
    int kHalfWidth  = kWidth / 2;
    int kHalfHeight = kHeight / 2;

    // Iterate over every pixel in the output image.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            // Loop over each kernel element.
            for (int ky = 0; ky < kHeight; ky++) {
                for (int kx = 0; kx < kWidth; kx++) {
                    // Calculate corresponding input coordinates, using offsets to center the kernel.
                    int ix = x + kx - kHalfWidth;
                    int iy = y + ky - kHalfHeight;
                    // If within image boundaries, accumulate the convolution sum.
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += input[iy * width + ix] * kernel[ky * kWidth + kx];
                    }
                    // Else: zero-padding is assumed.
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// ---------------------------------------------------------------------
// GPU (CUDA) Implementation of 2D Convolution
// ---------------------------------------------------------------------
__global__ void conv2D_gpu(const float* input, float* output, int width, int height,
                             const float* kernel, int kWidth, int kHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (x < width && y < height) {
        float sum = 0.0f;
        int kHalfWidth  = kWidth / 2;
        int kHalfHeight = kHeight / 2;
        for (int ky = 0; ky < kHeight; ky++) {
            for (int kx = 0; kx < kWidth; kx++) {
                int ix = x + kx - kHalfWidth;
                int iy = y + ky - kHalfHeight;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * kernel[ky * kWidth + kx];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

// ---------------------------------------------------------------------
// Main Function: Setup, Execution, and Markdown Report Generation
// ---------------------------------------------------------------------
int main() {
    // Set image dimensions.
    const int width = IMAGE_WIDTH;
    const int height = IMAGE_HEIGHT;
    const int imageSize = width * height;
    const int kernelSize = KERNEL_WIDTH * KERNEL_HEIGHT;

    // Allocate host memory for input and output images.
    float* h_input      = new float[imageSize];
    float* h_output_cpu = new float[imageSize];
    float* h_output_gpu = new float[imageSize];

    // Define and initialize a simple 3x3 convolution kernel (e.g., a sharpening kernel).
    float h_kernel[KERNEL_WIDTH * KERNEL_HEIGHT] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    // Initialize the input image with random values in [0, 1].
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ----------------------------
    // CPU Convolution and Timing
    // ----------------------------
    std::cout << "Running CPU convolution..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv2D_cpu(h_input, h_output_cpu, width, height, h_kernel, KERNEL_WIDTH, KERNEL_HEIGHT);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    double cpuTimeMs = cpu_duration.count();
    std::cout << "CPU convolution took " << cpuTimeMs << " ms" << std::endl;

    // ------------------------------------
    // GPU Convolution: Memory Allocation & Data Transfer
    // ------------------------------------
    float *d_input, *d_output, *d_kernel;
    cudaMalloc((void**)&d_input,  imageSize * sizeof(float));
    cudaMalloc((void**)&d_output, imageSize * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * sizeof(float));

    cudaMemcpy(d_input,  h_input,  imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // ----------------------------
    // Launching the GPU Kernel
    // ----------------------------
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running GPU convolution (naïve)..." << std::endl;
    cudaEventRecord(start);
    conv2D_gpu<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                          d_kernel, KERNEL_WIDTH, KERNEL_HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTimeMs = 0.0f;
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    std::cout << "GPU convolution (naïve) took " << gpuTimeMs << " ms" << std::endl;

    // Copy the GPU result back to host.
    cudaMemcpy(h_output_gpu, d_output, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    // ----------------------------
    // Validate and Compare Results
    // ----------------------------
    int errorCount = 0;
    for (int i = 0; i < imageSize; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 10) {
                std::cout << "Mismatch at index " << i << ": CPU = " 
                          << h_output_cpu[i] << ", GPU = " << h_output_gpu[i] << std::endl;
            }
        }
    }
    if (errorCount == 0)
        std::cout << "Results match between CPU and GPU implementations." << std::endl;
    else
        std::cout << "Total mismatches: " << errorCount << std::endl;

    // ----------------------------
    // Theoretical Minimum Time Calculation
    // ----------------------------
    // For a 3x3 kernel, each output pixel does 9 multiplications and 9 additions = 18 FLOPs.
    double totalFLOPs = static_cast<double>(width) * height * 18.0;
    // RTX 3050 assumed peak: 9.0e12 FLOPs/sec
    double theoreticalTimeMs = (totalFLOPs / 9.0e12) * 1000.0;

    // ----------------------------
    // Speedup Calculation
    // ----------------------------
    // Since we don't have an optimized GPU kernel, we calculate speedup as CPU time / Naïve GPU time.
    double speedup = (gpuTimeMs > 0) ? (cpuTimeMs / gpuTimeMs) : 0.0;

    // ----------------------------
    // Markdown Report Generation
    // ----------------------------
    std::ofstream mdFile("results_conv.md");
    if (!mdFile) {
        std::cerr << "Error opening results.md for writing." << std::endl;
    } else {
        mdFile << "# 2D Convolution Kernel Results\n\n";
        mdFile << "### Kernel Execution Times\n\n";
        mdFile << "| Rows | Cols | CPU Time (ms) | Naïve GPU Time (ms) | Theoretical Min Time (ms) | Speedup (CPU/Naïve) |\n";
        mdFile << "|------|------|---------------|---------------------|---------------------------|---------------------|\n";
        mdFile << "| " << height << " | " << width << " | " 
               << cpuTimeMs << " | " << gpuTimeMs 
               << " | " << theoreticalTimeMs 
               << " | " << speedup << " |\n\n";
        mdFile << "### Theoretical Details\n\n";
        mdFile << "- **FLOPs per Output Pixel:** For a 3×3 convolution, each pixel involves 9 multiplications and 9 additions, totalling **18 FLOPs**.\n";
        mdFile << "- **Total Operations:** " << totalFLOPs << " FLOPs.\n";
        mdFile << "- **Assumed GPU Peak Performance:** 9.0 TFLOPs (9.0e12 FLOPs/sec).\n";
        mdFile << "- **Theoretical Minimum Execution Time:** " << theoreticalTimeMs << " ms.\n\n";
        mdFile << "### Explanation for Kernel Offsets\n\n";
        mdFile << "The variables `kHalfWidth` and `kHalfHeight` represent half of the kernel's dimensions. They are used to offset the "
               << "kernel indices so that the kernel is centered over the current pixel. Without these offsets, the kernel would not "
               << "be properly aligned (e.g., its top–left corner would be placed on the pixel), leading to an incorrect weighted "
               << "sum during the convolution.\n";
        mdFile.close();
        std::cout << "Markdown report generated as results.md" << std::endl;
    }

    // ----------------------------
    // Cleanup: Free Memory and Destroy CUDA Events
    // ----------------------------
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
