#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void Conv2DKernel(float *d_input, float *d_output, int width, int height, float *d_kernel, int k){
    int x = blockIdx.x * blockDim.x + threadIdx.x; //get thread index
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height){ // check boundary
        float result = 0.0f;

        for (int i = -1; i<= 1; i++){ // loop neighbouring pixels
            for(int j=-1; j<=1; j++){
                int nx = x + i;
                int ny = y+j;

                if(nx>=0 && nx<width && ny>=0 && ny<height){ //check boundary if valid do
                    float pixel = d_input[ny * width + nx];
                    float k_value = d_kernel[(i+1) * k + (j+1)];
                    result += pixel * k_value;

                }
            }
        }
            d_output[y * width + x] = result; //store result
    }
}

int main(){
    
    string imgPath = "/content/lena.jpeg"; //path to ur image

    Mat image = imread(imgPath, IMREAD_GRAYSCALE); //should be clear
    Mat float_img; 
    image.convertTo(float_img, CV_32F);

    int width = image.cols; //neat
    int height = image.rows;
    int img_size = width * height * sizeof(float); //dont forget  *sizeof(float)

    Mat h_output(height, width, CV_32F); //output image for later

    float *d_input, *d_output, *d_kernel; //device ptr inits

    int k = 3;        //filter , 3 for 3x3 filter, 5 for 5x5 filter etc
    int k_n = k * k; //number of elements in filter
    int k_size = k_n * sizeof(float); // size of filter

    // dynamically assign kernel the normal way
    float *h_kernel = new float[k_n]{0, -1, 0, -1, 5, -1, 0, -1, 0};
    
    // or use the cuda way to chose your own (3x3) KERNEL
    // float *h_kernel;
    // cudaMallocHost(&h_kernel, k_size);

    // Initialize the kernel values
    // h_kernel[0] = 0; h_kernel[1] = -1; h_kernel[2] = 0;
    // h_kernel[3] = -1; h_kernel[4] = 5; h_kernel[5] = -1;
    // h_kernel[6] = 0; h_kernel[7] = -1; h_kernel[8] = 0
    
    // allocate device memory
    cudaMalloc(&d_input, img_size); 
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel, k_size);

    // copy host memory to device memory
    cudaMemcpy(d_input, float_img.ptr<float>(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, k_size, cudaMemcpyHostToDevice);

    // define grid and block size
    dim3 blocksize(16, 16);
    dim3 gridsize((width + blocksize.x -1)/ blocksize.x, (height + blocksize.y -1)/blocksize.y);

    // launch kernel
    Conv2DKernel<<<gridsize, blocksize>>>(d_input, d_output, width, height, d_kernel, k);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // copy output back to host memory
    cudaMemcpy(h_output.ptr<float>(), d_output, img_size, cudaMemcpyDeviceToHost);

    // convert output to 8-bit unsigned integer
    Mat output_8u;
    
    h_output.convertTo(output_8u, CV_8U, 1.0, 128);

    // save output image
    imwrite("cuda_output_img.png", output_8u);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] h_kernel;

    return 0;
}
