#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define KERNEL_SIZE 3
#define Filter_Radius 1

__constant__ float kernel_gpu[KERNEL_SIZE][KERNEL_SIZE];

__global__ void conv_constant_kernel(float* input, float* output, int height, int width) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float temp = 0.0f;
    if (row < height && col<width) {
        
        for (int i=0; i<KERNEL_SIZE;i++){
            for (int j=0;j<KERNEL_SIZE;j++){
                int inRow = row - Filter_Radius +i;
                int inCol = col - Filter_Radius + j;
                if (inRow >=0 && inRow <height && inCol >=0 && inCol < width) {
                    temp += input[inRow*width + inCol] * kernel_gpu[i][j];
                }
            }
        }
    output[row*width + col] = temp;


    }
}

void conv_constant(float *input, int height, int width, float *kernel, float *output){
    float *input_gpu, *output_gpu;
    float size = height * width * sizeof(float);
    // Allocate the memory in GPU
    cudaMalloc((void**) &input_gpu, size);
    cudaMalloc((void **)&output_gpu, size);

    //copy contents from CPU to GPU

    cudaMemcpy(input_gpu, input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_gpu, kernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));

    //Launch the kernel

    dim3 numThreads (32, 32);
    dim3 numBlocks ((width + numThreads.x - 1)/numThreads.x, (height + numThreads.y-1)/numThreads.y);

    conv_constant_kernel<<<numBlocks, numThreads>>>(input_gpu, output_gpu, height, width);

    // Copy contents of the output back to GPU
    
    cudaMemcpy(output, output_gpu, size, cudaMemcpyDeviceToHost);
    
    // free the space
    cudaFree(output_gpu);
    cudaFree(input_gpu);


}
// Function to perform 2D convolution on CPU
void convolution2DCPU(float* input, float* output, int width, int height, float* kernel) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float value = 0.0f;
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    int inputRow = row - KERNEL_SIZE / 2 + i;
                    int inputCol = col - KERNEL_SIZE / 2 + j;

                    if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width) {
                        value += input[inputRow * width + inputCol] * kernel[i * KERNEL_SIZE + j];
                    }
                }
            }
            output[row * width + col] = value;
        }
    }
}

// Function to print a 1D array as a 2D matrix
void printArray(float* arr, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << arr[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to compare two matrices
bool compareMatrices(float *c, float *c_gpu, int m, int n, float tolerance = 1e-4) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(c[i * n + j] - c_gpu[i * n + j]) > tolerance) {
                cout << "Mismatch at (" << i << ", " << j << "): CPU = " << c[i * n + j] << ", GPU = " << c_gpu[i * n + j] << " " << c[i * n + j] - c_gpu[i * n + j] << endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    int height = 320000;
    int width = 3200;
    int kernelHeight = 2 * Filter_Radius + 1;
    int kernelSize = kernelHeight * kernelHeight;

    float *input = (float*)malloc(height * width * sizeof(float));
    float *kernel = (float*)malloc(kernelSize * sizeof(float));
    float *output = (float*)malloc(height * width * sizeof(float));

    // Initialize input and kernel with random values
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i * width + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i < kernelHeight; i++) {
        for (int j = 0; j < kernelHeight; j++) {
            kernel[i * kernelHeight + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Perform convolution on CPU
    clock_t start, stop;
    start = clock();
    convolution2DCPU(input, output, width, height, kernel);
    stop = clock();
    double cpu_time_used = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "CPU Time : " << cpu_time_used * 1000 << " ms" << endl;


    // Perform convolution on GPU
    float *output_gpu = (float*)malloc(height * width * sizeof(float));
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    conv_constant(input, height, width, kernel, output_gpu);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_used;
    cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
    cout << "GPU Time : " << gpu_time_used << " ms" << endl;

    // Compare CPU and GPU outputs
    if (compareMatrices(output, output_gpu, height, width)) {
        std::cout << "Matrices match!" << std::endl;
    } else {
        std::cout << "Matrices do not match!" << std::endl;
    }

    // Free memory
    free(input);
    free(kernel);
    free(output);
    free(output_gpu);

    return 0;
}