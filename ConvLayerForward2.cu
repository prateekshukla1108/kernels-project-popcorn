#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__constant__ float d_kernel[3][3];  

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define PADDING 1  // (KERNEL_SIZE - 1) / 2
#define sharedWidth (TILE_SIZE + 2 * PADDING)



__global__ void ConvForward(float *d_input, float *d_output, int width, int height,
                           int poolSize, int stride, int outputW, int outputH) {
    
    __shared__ float sharedMem[sharedWidth][sharedWidth][3];  
    __shared__ float convResults[TILE_SIZE][TILE_SIZE][3];    

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    
    // Global input position with padding offset
    const int gx = blockIdx.x * TILE_SIZE + tx - PADDING; 
    const int gy = blockIdx.y * TILE_SIZE + ty - PADDING;
    
    // 1. Load input tile into shared memory
    if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
        sharedMem[ty][tx][tz] = d_input[(gy * width + gx) * 3 + tz];
    } else {
        sharedMem[ty][tx][tz] = 0.0f;  // Zero-padding
    }
    __syncthreads();

    // 2. Compute convolution only for valid threads
    if (tx >= 1 && tx < TILE_SIZE+1 && ty >= 1 && ty < TILE_SIZE+1) {
        float result = 0.0f;
        
        // Clear 3x3 convolution using 2D kernel access
        for(int ky = 0; ky < 3; ky++) {      // Kernel row
                for(int kx = 0; kx < 3; kx++) {  // Kernel column
                result += sharedMem[ty-1+ky][tx-1+kx][tz] // ty-1 because we want the topleft of the kernel
                        * d_kernel[ky][kx];  
            }
        }
        convResults[ty-1][tx-1][tz] = fmaxf(0.0f, result); // ReLU
    }
    __syncthreads();

    // tx % stride == 0 && ty % stride == 0, could be used for more effeciency
    if (tx < outputW && ty < outputH) {
        const int poolX = tx / stride;
        const int poolY = ty / stride;
        float max_val = -INFINITY;

        // Pool within conv results
        for(int py = 0; py < poolSize; py++) {      
                for(int px = 0; px < poolSize; px++) {
                if (poolY*stride + py < TILE_SIZE && poolX*stride + px < TILE_SIZE) {
                    max_val = fmaxf(max_val,
                                  convResults[poolY*stride + py][poolX*stride + px][tz]);
                }
            }
        }

        // Global output position
        const int outX = blockIdx.x * (TILE_SIZE/stride) + poolX;
        const int outY = blockIdx.y * (TILE_SIZE/stride) + poolY;
        
        if (outX < outputW && outY < outputH) {
            d_output[(outY * outputW + outX) * 3 + tz] = max_val;
        }
    }
}



int main(){

    string imgPath;
    cout << "Enter image path: ";
    cin >> imgPath;


    Mat image = imread(imgPath, IMREAD_COLOR); //should be clear
    if (image.empty()) {
    cerr << "Error: Could not load image!" << endl;
    return -1;
    }
    Mat float_img;
    image.convertTo(float_img, CV_32FC3);

    int width = image.cols; //neat
    int height = image.rows;

    cout << "Input Width: " << width << endl;
    cout << "Input Height: " << height << endl;

    int imgSize = width * height * 3 * sizeof(float); //dont forget to account for 3 channels


    int poolSize;
    int stride;

    cout << "Enter PoolSize: " << endl;
    cin >> poolSize;
    cout << "Enter stride: " << endl;
    cin >> stride;

    int outputW = ((width - poolSize)/stride) + 1;
    int outputH = ((height - poolSize)/stride) + 1;

    cout << "Output Width: " << outputW << endl;
    cout << "Output Height: " << outputH << endl;

    if (outputW <= 0 || outputH <= 0) {
    std::cerr << "Error: Invalid poolSize/stride for image dimensions.\n";
    return -1; // Exit early
    }

    int k = 3;        //filter , 3 for 3x3 filter, 5 for 5x5 filter etc
    int k_n = k * k; //number of elements in filter
    int k_size = k_n * sizeof(float); // size of filter
    int padding = (k - 1) / 2;


    // dynamically assign kernel the normal way
// Host code
    float h_kernel[3][3] = {{0,-1,0}, {-1,5,-1}, {0,-1,0}};

    Mat h_output(outputH, outputW, CV_32FC3); //output image for later

    int outImgSize = outputH * outputW * 3 * sizeof(float); // same channels here

    float *d_input, *d_conv_out, *d_relu_out, *d_output;
    

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, outImgSize);

    cudaMemcpy(d_input, float_img.ptr<float>(), imgSize, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_kernel, h_kernel, k_size);

    // cudaMemcpy(d_kernel, h_kernel, k_size, cudaMemcpyHostToDevice);

    // int sharedWidth = TILE_SIZE + 2 * PADDING;

    dim3 blockSize(sharedWidth, sharedWidth, 3);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);

    ConvForward<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, poolSize, stride, outputW, outputH
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // copy output back to host memory
    cudaMemcpy(h_output.ptr<float>(), d_output, outImgSize, cudaMemcpyDeviceToHost);

    // convert and save image
    Mat pool_output_8u;

    h_output.convertTo(pool_output_8u, CV_8U, 1.0);

    imwrite("ConvF_output_img.png", pool_output_8u);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    // delete[] h_kernel;
    // cudaFree(d_conv_out);
    // cudaFree(d_relu_out);
    cudaFree(d_output);

    return 0;
}
