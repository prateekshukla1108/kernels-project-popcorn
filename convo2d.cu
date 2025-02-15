#include <iostream> 
#include <stdio.h>  
#include <cuda_runtime.h> 

// Define constants for image and filter sizes
#define IMAGE_SIZE 5
#define FILTER_SIZE 3
#define STRIDE 1

// CUDA kernel function for 2D convolution
__global__ void conv2d(float* image, float* filter, float* output, int image_size, int filter_size, int stride) {
    // Calculate the row and column indices for the current thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // Calculate the size of the output image
    int output_size = (image_size - filter_size) / stride + 1;

    // Synchronize threads within a block (important for correct calculations)
    __syncthreads();

    // Check if the current thread is within the valid output range
    if (row < image_size - filter_size + 1 && col < image_size - filter_size + 1) {
        float sum = 0.0; // Initialize the sum for the convolution calculation

        // Iterate over the filter rows
        for (int i = 0; i < filter_size; i++) {
            // Iterate over the filter columns
            for (int j = 0; j < filter_size; j++) {
                // Perform the convolution calculation: multiply image pixel by filter value and add to sum
                sum += image[(row + i) * (image_size) + col + j] * filter[(i)*filter_size + j];
            }
        }
        // Store the result in the output array
        output[row * output_size + col] = sum;
    }
}

// Function to verify the convolution result on the host (CPU)
void verifyConvolution(float* h_image, float* h_filter, float* h_output, int image_size, int filter_size, int stride) {
    int output_size = (image_size - filter_size) / stride + 1; // Calculate output size
    float* expected_output = new float[output_size * output_size]; // Allocate memory for expected output

    // Calculate the expected output on the host (CPU)
    for (int row = 0; row < output_size; ++row) {
        for (int col = 0; col < output_size; ++col) {
            float sum = 0.0; // Initialize sum for each output pixel
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    sum += h_image[(row + i) * image_size + col + j] * h_filter[i * filter_size + j];
                }
            }
            expected_output[row * output_size + col] = sum; // Store expected output
        }
    }

    bool match = true; // Flag to track if the results match
    // Compare the calculated output with the expected output
    for (int i = 0; i < output_size * output_size; ++i) {
        if (h_output[i] != expected_output[i]) { // Check for mismatches
            std::cerr << "Mismatch at index " << i << ": Expected " << expected_output[i] << ", Got " << h_output[i] << std::endl;
            match = false; // Set flag to false if mismatch found
        }
    }

    // Print verification result
    if (match) {
        std::cout << "Convolution verified successfully!\n";
    } else {
        std::cerr << "Convolution verification failed.\n";
    }

    delete[] expected_output; // Free allocated memory for expected output
}

int main() {
    // Input image (5x5)
    float h_image[IMAGE_SIZE * IMAGE_SIZE] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // Filter (3x3)
    float h_filter[FILTER_SIZE * FILTER_SIZE] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    int output_size = (IMAGE_SIZE - FILTER_SIZE) / STRIDE + 1; // Calculate output size
    float* h_output = new float[output_size * output_size]; // Allocate memory for output on host

    for (int i = 0; i < output_size * output_size; ++i) {
        h_output[i] = 0.0f; // Initialize output array to 0
    }


    // Device pointers (for data on the GPU)
    float *d_image, *d_filter, *d_output;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_image, IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * output_size * sizeof(float));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_image, h_image, IMAGE_SIZE * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes for the kernel launch
    dim3 blockSize(16, 16); // Block size (threads per block)
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y); // Grid size (blocks per grid)

    // Launch the convolution kernel on the GPU
    conv2d<<<gridSize, blockSize>>>(d_image, d_filter, d_output, IMAGE_SIZE, FILTER_SIZE, STRIDE);

    // Copy the result back from device (GPU) to host (CPU)
    cudaMemcpy(h_output, d_output, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result by comparing with host-side calculation
    verifyConvolution(h_image, h_filter, h_output, IMAGE_SIZE, FILTER_SIZE, STRIDE);

    std::cout << "Convolution Output:\n";
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            std::cout << h_output[i * output_size + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);

    // Free memory allocated on the host
    delete[] h_output;

    return 0;
}
