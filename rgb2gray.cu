// Include necessary header files
#include "common.h"   // Common utilities (assumed to be available)
#include "timer.h"    // Timer utilities for performance measurement

#include <cuda_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"    // stb_image for reading images
#include "stb_image_write.h" // stb_image_write for writing images

// CUDA Kernel: Converts an RGB image to grayscale
__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    // Compute row and column indices based on thread and block indices
    // blockIdx is the index of the block in the grid   
    // threadIdx is the index of the thread in the block
    // blockDim is the number of threads per block
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;


    // Check if within image boundaries
    if (row < height && col < width) {
        unsigned int index = row * width + col; // Convert 2D indices to a 1D index
        
        // Compute grayscale value using weighted sum method
        // The formula uses standard luminance conversion:
        // Y = 0.3*R + 0.59*G + 0.11*B (perceptual weights from NTSC standards)
        gray[index] = static_cast<unsigned char>(red[index] * 0.3f + green[index] * 0.59f + blue[index] * 0.11f);
    }
}

// Function to manage memory allocation, data transfer, and kernel execution
void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    Timer timer;  // Timer object for measuring execution time

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy input data from host to device
    startTime(&timer);
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Data transfer to GPU");

    // Define thread block and grid sizes for 2D image
    dim3 blockSize(32, 32);  // 32x32 threads per block
    // Choosing 32x32 because it aligns with CUDA warp size (32 threads per warp), maximizing efficiency.
    // GPUs execute threads in groups of 32 (warps), so having block sizes as multiples of 32 ensures no wasted computation.
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel
    startTime(&timer);
    rgb2gray_kernel<<<gridSize, blockSize>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel execution time");

    // Copy result back from device to host
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Data transfer to CPU");

    // Free GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}

int main() {
    // Load the image using stb_image
    int width, height, channels;
    unsigned char *img = stbi_load("image.png", &width, &height, &channels, 3); // Force 3 channels (RGB)
    if (!img) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    
    // Allocate memory for host RGB and grayscale images
    unsigned char *red = new unsigned char[width * height];
    unsigned char *green = new unsigned char[width * height];
    unsigned char *blue = new unsigned char[width * height];
    unsigned char *gray = new unsigned char[width * height];
    
    // Extract RGB channels from image data
    for (unsigned int i = 0; i < width * height; i++) {
        red[i] = img[i * 3];     // Red channel
        green[i] = img[i * 3 + 1]; // Green channel
        blue[i] = img[i * 3 + 2];  // Blue channel
    }
    
    // Call the function to process the image on GPU
    rgb2gray_gpu(red, green, blue, gray, width, height);
    
    // Save grayscale image using stb_image_write
    stbi_write_png("grayscale.png", width, height, 1, gray, width);
    
    // Free host memory
    delete[] red;
    delete[] green;
    delete[] blue;
    delete[] gray;
    stbi_image_free(img);

    return 0;
}
