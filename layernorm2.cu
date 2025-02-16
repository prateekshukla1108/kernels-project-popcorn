#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <torch/extension.h>

// CUDA kernel for LayerNorm
__global__ void layerNormKernel(const float* input, float* output, const float* gamma, const float* beta, int feature_size) {
    int batch_idx = blockIdx.x; // Each block corresponds to a single batch element

    // Shared memory to compute mean and variance for a single batch
    extern __shared__ float shared_data[];

    float* shared_mean = shared_data;
    float* shared_variance = shared_data + 1;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        *shared_mean = 0.0f;
        *shared_variance = 0.0f;
    }
    __syncthreads();

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        sum += input[batch_idx * feature_size + i];
    }
    atomicAdd(shared_mean, sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        *shared_mean /= feature_size;
    }
    __syncthreads();

    // Step 2: Compute variance
    float variance_sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float diff = input[batch_idx * feature_size + i] - *shared_mean;
        variance_sum += diff * diff;
    }
    atomicAdd(shared_variance, variance_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        *shared_variance = sqrtf(*shared_variance / feature_size + 1e-5f); // Add epsilon for numerical stability
    }
    __syncthreads();

    // Step 3: Normalize and apply gamma and beta
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        int idx = batch_idx * feature_size + i;
        output[idx] = gamma[i] * ((input[idx] - *shared_mean) / *shared_variance) + beta[i];
    }
}

// Host function to call LayerNorm kernel
void layerNorm(const float* input, float* output, const float* gamma, const float* beta, int batch_size, int feature_size) {
    float *d_input, *d_output, *d_gamma, *d_beta;

    size_t input_size = batch_size * feature_size * sizeof(float);
    size_t param_size = feature_size * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, param_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int shared_memory_size = 2 * sizeof(float); // For mean and variance

    layerNormKernel<<<batch_size, block_size, shared_memory_size>>>(d_input, d_output, d_gamma, d_beta, feature_size);

    cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layerNorm, "Layer Normalization CUDA");
}

int main() {
    const int batch_size = 2;
    const int feature_size = 4;

    float input[batch_size * feature_size] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float gamma[feature_size] = {1.0, 1.0, 1.0, 1.0};
    float beta[feature_size] = {0.0, 0.0, 0.0, 0.0};
    float output[batch_size * feature_size];

    layerNorm(input, output, gamma, beta, batch_size, feature_size);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_size; ++j) {
            printf("%.4f ", output[i * feature_size + j]);
        }
        printf("\n");
    }

    return 0;
}
