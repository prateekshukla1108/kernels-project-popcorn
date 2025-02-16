#include <cuda_runtime.h>
#include <stdio.h>
#include <string>                // For std::string
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// define the MaxPool kernel
__global__ void MaxPoolKernel(
    float *d_input, float *d_output,
    int inputW, int inputH, int poolSize,
    int stride, int outputW, int outputH)
{
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int cIdx = blockIdx.z * blockDim.z + threadIdx.z;

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

    Mat h_output(outputH, outputW, CV_32FC3); //output image for later

    int outImgSize = outputH * outputW * 3 * sizeof(float); // same channels here

    float *d_input, *d_output;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, outImgSize);

    cudaMemcpy(d_input, float_img.ptr<float>(), imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);  // 16x16 threads per block
    dim3 gridSize(
        (outputW + blockSize.x - 1)/ blockSize.x,
        (outputH + blockSize.y - 1)/ blockSize.y,
        3
    );
    MaxPoolKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, poolSize, stride, outputW, outputH);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // copy output back to host memory
    cudaMemcpy(h_output.ptr<float>(), d_output, outImgSize, cudaMemcpyDeviceToHost);

    // convert output to 8-bit unsigned integer
    Mat output_8u;

    h_output.convertTo(output_8u, CV_8U, 1.0);

    // save output image
    imwrite("cuda_output_img.png", output_8u);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
