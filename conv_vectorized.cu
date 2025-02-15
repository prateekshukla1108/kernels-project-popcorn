#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void conv2d_vectorized(
    float* input,
    float* filter,
    float* output,
    const int image_size,
    const int filter_size,
    const int stride,
    const int batch_size
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    int output_size = (image_size - filter_size) / stride + 1;

    if (tx < output_size && ty < output_size && batch < batch_size) {
        int in_row = ty * stride;
        int in_col = tx * stride;

        float result = 0.0f;

        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                int input_row = in_row + i;
                int input_col = in_col + j;

                if (input_row < image_size && input_col < image_size) {
                    int input_index = batch * image_size * image_size + input_row * image_size + input_col;
                    int filter_index = i * filter_size + j;
                    result += input[input_index] * filter[filter_index];
                }
            }
        }

        int out_idx = batch * (output_size * output_size) + ty * output_size + tx;
        output[out_idx] = result;
    }
}

void launch_conv2d_vectorized(
    float* d_input,
    float* d_filter,
    float* d_output,
    int image_size,
    int filter_size,
    int stride,
    int batch_size
) {
    int output_size = (image_size - filter_size) / stride + 1;

    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(
        (output_size + block_dim.x - 1) / block_dim.x,
        (output_size + block_dim.y - 1) / block_dim.y,
        batch_size
    );

    conv2d_vectorized<<<grid_dim, block_dim>>>(
        d_input, d_filter, d_output, image_size, filter_size, stride, batch_size
    );

    cudaDeviceSynchronize();
}

// CPU convolution for verification
std::vector<float> cpu_conv2d(const std::vector<float>& input, const std::vector<float>& filter, int image_size, int filter_size, int stride, int batch_size) {
    int output_size = (image_size - filter_size) / stride + 1;
    std::vector<float> output(output_size * output_size * batch_size);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int y = 0; y < output_size; ++y) {
            for (int x = 0; x < output_size; ++x) {
                float result = 0.0f;
                for (int i = 0; i < filter_size; ++i) {
                    for (int j = 0; j < filter_size; ++j) {
                        int input_row = y * stride + i;
                        int input_col = x * stride + j;
                        if (input_row < image_size && input_col < image_size) {
                            int input_index = batch * image_size * image_size + input_row * image_size + input_col;
                            int filter_index = i * filter_size + j;
                            result += input[input_index] * filter[filter_index];
                        }
                    }
                }
                output[batch * output_size * output_size + y * output_size + x] = result;
            }
        }
    }
    return output;
}


int main() {
    int image_size = 256;
    int filter_size = 3;
    int stride = 1;
    int batch_size = 1;

    int output_size = (image_size - filter_size) / stride + 1;

    std::vector<float> h_input(image_size * image_size * batch_size);
    std::vector<float> h_filter(filter_size * filter_size);
    std::vector<float> h_output_gpu(output_size * output_size * batch_size);
    std::vector<float> h_output_cpu;

    for (int i = 0; i < h_input.size(); ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < h_filter.size(); ++i) {
        h_filter[i] = 1.0f;
    }

    float* d_input;
    float* d_filter;
    float* d_output;

    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_filter, h_filter.size() * sizeof(float));
    cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), h_filter.size() * sizeof(float), cudaMemcpyHostToDevice);

    launch_conv2d_vectorized(d_input, d_filter, d_output, image_size, filter_size, stride, batch_size);

    cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    h_output_cpu = cpu_conv2d(h_input, h_filter, image_size, filter_size, stride, batch_size);


    bool correct = true;
    float tolerance = 1e-5f;  

    for (size_t i = 0; i < h_output_gpu.size(); ++i) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU = " << h_output_gpu[i] << ", CPU = " << h_output_cpu[i] << std::endl;
            correct = false;
            break; 
        }
    }

    if (correct) {
        std::cout << "Verification successful!" << std::endl;
    } else {
        std::cout << "Verification failed." << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
