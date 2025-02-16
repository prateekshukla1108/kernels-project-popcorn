#include <iostream>
#include <cuda_runtime.h>

using namespace std; 

// filter size will be (2 * FILTER_PIXELS + 1)
# define FILTER_PIXELS 1

const int TEST_IMAGE_HT = 50;
const int TEST_IMAGE_WD = 50;

__global__
void blur_greyscale(unsigned char* img_in, unsigned char* img_out, int height, int width){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int idx = row * width + col;

    // if thread operates on image pixels
    if (row < height && col < width){
        int pixel_sum = 0;
        int pixel_count = 0;
        for (int i=-FILTER_PIXELS; i < FILTER_PIXELS+1; ++i){
            for (int j=-FILTER_PIXELS; j < FILTER_PIXELS+1; ++j){
                int n_row = row + i;
                int n_col = col + j;
                if (n_row > -1 && n_col > -1 && n_row < height && n_col < width){
                    pixel_sum += img_in[n_row * width + n_col];
                    ++pixel_count;
                }
            }
        }
        img_out[idx] = (unsigned char)pixel_sum/pixel_count;
    }
}


//Blur a greyscale image
unsigned char* image_blur(unsigned char* image, int image_height, int image_width){
    int size_image = image_height*image_width;

    unsigned char *d_img, *d_blurimg, *blur_img;

    blur_img = new unsigned char[size_image];

    //allocateMemory on device
    cudaMalloc((void**) &d_img, size_image);
    cudaMalloc((void**) &d_blurimg, size_image);

    //copy image to gpu
    cudaMemcpy(d_img, image, size_image, cudaMemcpyHostToDevice);

    const int n_thread_x = 32;
    const int n_thread_y = 32;
    const int n_blocks_x = (image_width + n_thread_x - 1) / n_thread_x;
    const int n_blocks_y = (image_height + n_thread_y - 1) / n_thread_y;
    dim3 block_dimens(n_thread_x, n_thread_y, 1);
    dim3 grid_dimens(n_blocks_x, n_blocks_y, 1);

    blur_greyscale<<<grid_dimens, block_dimens>>>(d_img, d_blurimg, image_height, image_width);


    //copy blurred image to host
    cudaMemcpy(blur_img, d_blurimg, size_image, cudaMemcpyDeviceToHost);

    //Free gpu memory
    cudaFree(d_img);
    cudaFree(d_blurimg);

    return blur_img;
}

void test_gray_scale_filter(){
    unsigned char* img = new unsigned char[5*5]();
    img[2 * 5 + 2] = 255;

    unsigned char* op = image_blur(img, 5, 5);

     for (int i=0; i< 5; ++i){
        for (int j=0; j < 5; ++j){
            cout << int(op[i * 5 + j]) << " ";
        }
        cout << "\n";
    }
}

int main(){
    test_gray_scale_filter();
}