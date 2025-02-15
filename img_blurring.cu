#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "timer.h"

// CUDA kernel to apply a 3x3 blur filter to an image
__global__ void my_kernel(unsigned char *d_img, unsigned char *d_out_img, int width, int height, int channels) {
    // Compute the row and column index for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the image boundaries
    if (row < height && col < width) {
        // Process all color channels (R, G, B)
        for (int c = 0; c < channels; c++) {
            // Compute the 1D index for this pixel in the image array
            int index = (row * width + col) * channels + c;
            
            // Initialize sum and count for averaging neighboring pixels
            float sum = 0.0f;
            int count = 0;

            // Iterate over the 3x3 neighborhood of the pixel
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int new_row = row + i;
                    int new_col = col + j;

                    // Ensure the neighboring pixel is within image boundaries
                    if (new_row >= 0 && new_row < height && new_col >= 0 && new_col < width) {
                        int neighbor_index = (new_row * width + new_col) * channels + c;
                        sum += d_img[neighbor_index];
                        count++;
                    }
                }
            }

            // Compute the average value and store it in the output image using floating-point division
            d_out_img[index] = static_cast<unsigned char>(sum / static_cast<float>(count));
        }
    }
}

// Function to apply a blur filter to an image using CUDA
void img_blur(unsigned char *h_img, int width, int height, int channels) {
    Timer timer;
    startTime(&timer);

    // Allocate memory for input and output images on the GPU
    unsigned char *d_img, *d_out_img;
    cudaMalloc((void**)&d_img, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_out_img, width * height * channels * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Time taken to allocate memory: %f seconds\n", timer.duration.count());

    // Copy the input image data from the host to the GPU memory
    startTime(&timer);
    cudaMemcpy(d_img, h_img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Time taken to copy memory from host to device: %f seconds\n", timer.duration.count());

    // Define the size of thread blocks (16x16 threads per block)
    dim3 my_block(16, 16);

    // Compute the number of blocks needed to cover the entire image
    dim3 my_grid((width + my_block.x - 1) / my_block.x, 
                 (height + my_block.y - 1) / my_block.y);

    // Launch the CUDA kernel to apply the blur filter
    startTime(&timer);
    my_kernel<<<my_grid, my_block>>>(d_img, d_out_img, width, height, channels);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Time taken to call the kernel: %f seconds\n", timer.duration.count());

    // Allocate memory for storing the output image on the host
    unsigned char *h_out_img = (unsigned char*)malloc(width * height * channels);

    // Copy the processed image data from the GPU back to the host memory
    startTime(&timer);
    cudaMemcpy(h_out_img, d_out_img, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Time taken to copy memory from device to host: %f seconds\n", timer.duration.count());

    // Free GPU memory after processing
    cudaFree(d_img);
    cudaFree(d_out_img);

    // Save the blurred image as a PNG file.
    // Use the same channel count and compute the stride as width * channels.
    stbi_write_png("blurred_image.png", width, height, channels, h_out_img, width * channels);
    free(h_out_img);
}

// Main function: loads an image, applies the blur filter, and saves the output
int main(int argc, char **argv) {
    int width, height, orig_channels;

    // Force the image to have 3 channels (RGB). Note that stbi_load writes the original
    // channel count into orig_channels. We force our working channel count to 3.
    unsigned char *h_img = stbi_load("image.png", &width, &height, &orig_channels, 3);
    if (!h_img) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    // Since we forced 3 channels, use 3 in the rest of the code.
    int channels = 3;

    // Print image dimensions
    std::cout << "Image dimensions: " << width << " x " << height << std::endl;

    // Call the function to process the image on the GPU
    img_blur(h_img, width, height, channels);

    // Free the image memory after processing
    stbi_image_free(h_img);

    return 0;
}
