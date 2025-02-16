#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA Kernel for Online Softmax
__global__ void online_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float max_val = -FLT_MAX;
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            if (max_val == -FLT_MAX){
                max_val = input[row*cols + i];
            }
            if (input[row * cols + i] > max_val && max_val != -FLT_MAX) {
                float correction_factor = max_val - input[row*cols + i];
                sum_exp *= expf(correction_factor);
                max_val = input[row*cols + i];
            }


            sum_exp += expf(input[row * cols + i] - max_val);
        }

        for (int i = 0; i < cols; i++) {
            output[row * cols + i] = expf(input[row * cols + i] - max_val) / sum_exp;
        }
    }
}

// Host function to launch the kernel
void online_softmax_launcher(const torch::Tensor& input, torch::Tensor& output) {
    const auto rows = input.size(0);
    const auto cols = input.size(1);

    const dim3 threads_per_block(256);
    const dim3 num_blocks((rows + threads_per_block.x - 1) / threads_per_block.x);

    online_softmax_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("online_softmax", &online_softmax_launcher, "Online Softmax CUDA");
}
