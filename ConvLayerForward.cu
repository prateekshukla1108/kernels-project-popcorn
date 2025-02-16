#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
// #include <cmath> // For sqrt

using namespace std;
using namespace cv;

__constant__ float d_kernel[9];

__global__ void Conv2DKernel(float *d_input, float *d_output, int width, int height, int k){
    int x = blockIdx.x * blockDim.x + threadIdx.x; //get thread index
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int padding = (k - 1) / 2;
    if (x < width && y < height && z < 3){ // check boundary
        float result = 0.0f;
        for (int i = -padding; i <= padding; i++) { // loop neighbouring pixels
            for (int j = -padding; j <= padding; j++) {
                int nx = x + i;
                int ny = y + j;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) { // check boundary if valid
                    int fIdx = (ny * width + nx) * 3 + z;
                    float pixel = d_input[fIdx];
                    float k_value = d_kernel[(i + padding) * k + (j + padding)];
                    result += pixel * k_value;
            }
        }
    }   
            // if (x < width && y < height && z<3)
            int oIdx = (y * width + x) * 3 + z;
            d_output[oIdx] = result; //store result
    }
}

__global__ void ReLUKernel(float *d_input, float *d_output, int Width, int Height){

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    

    if (x < Width && y < Height && z < 3){
    int i = (y * Width + x) * 3 + z;

    d_output[i] = fmaxf(0.0f, d_input[i]);
    }
}

// define the MaxPool kernel
__global__ void MaxPoolKernel(
    float *d_input, float *d_output,
    int inputW, int inputH, int poolSize,
    int stride, int outputW, int outputH)
    {
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int cIdx = blockIdx.z;

    if (xIdx < outputW && yIdx < outputH){

        float maxVal = -INFINITY; // because in some cases the activations can get below zero

        for (int py=0; py<poolSize; py++){
            for(int px=0; px<poolSize; px++){
                int inX = xIdx * stride + px;
                int inY = yIdx * stride + py;
                if (inX<inputW && inY<inputH){
                    int idx = (inY * inputW + inX) * 3 + cIdx;
                    float val = d_input[idx];
                    if (val > maxVal) maxVal = val;
                }
            }
        }
        int outIdx = (yIdx * outputW + xIdx) * 3 + cIdx;
        d_output[outIdx] = maxVal;
        //d_output[yIdx * outputW + xIdx] = maxVal;
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

    // dynamically assign kernel the normal way
    float *h_kernel = new float[k_n]{0, -1, 0, -1, 5, -1, 0, -1, 0};

    Mat convh_output(height, width, CV_32FC3); //output for Conv
    Mat reluh_output(height, width, CV_32FC3); //output for Relu
    Mat h_output(outputH, outputW, CV_32FC3); //output image for later

    int outImgSize = outputH * outputW * 3 * sizeof(float); // same channels here

    float *d_input, *d_conv_out, *d_relu_out, *d_output;
    

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_conv_out, imgSize);
    cudaMalloc(&d_relu_out, imgSize);
    // cudaMalloc(&d_kernel, k_size);
    cudaMalloc(&d_output, outImgSize);

    cudaMemcpy(d_input, float_img.ptr<float>(), imgSize, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_kernel, h_kernel, k_size);

    // cudaMemcpy(d_kernel, h_kernel, k_size, cudaMemcpyHostToDevice);


    dim3 blockSize(16, 16, 1);  // 16x16 threads per block
    dim3 ConvGrid((width + blockSize.x -1)/ blockSize.x,
                  (height + blockSize.y -1)/blockSize.y,
                   3);

    dim3 Poolgrid(
        (outputW + blockSize.x - 1)/ blockSize.x,
        (outputH + blockSize.y - 1)/ blockSize.y,
        3
    );
    Conv2DKernel<<<ConvGrid, blockSize>>>(d_input, d_conv_out, width, height, k);
    ReLUKernel<<<ConvGrid, blockSize>>>(d_conv_out, d_relu_out, width, height);
    MaxPoolKernel<<<Poolgrid, blockSize>>>(d_relu_out, d_output, width, height, poolSize, stride, outputW, outputH);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // copy output back to host memory
    cudaMemcpy(convh_output.ptr<float>(), d_conv_out, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output.ptr<float>(), d_output, outImgSize, cudaMemcpyDeviceToHost);

    // convert output to 8-bit unsigned integer
    Mat conv_output_8u, relu_output_8u, pool_output_8u;

    convh_output.convertTo(conv_output_8u, CV_8U, 1.0);
    h_output.convertTo(pool_output_8u, CV_8U, 1.0);

    // Save images
    imwrite("conv_output_img.png", conv_output_8u);
    imwrite("pool_output_img.png", pool_output_8u);

    // free device memory
    cudaFree(d_input);
    // cudaFree(d_kernel);
    delete[] h_kernel;
    cudaFree(d_conv_out);
    cudaFree(d_relu_out);
    cudaFree(d_output);

    return 0;
}
