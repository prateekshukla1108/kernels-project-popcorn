__global__ void gelu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

void launch_gelu_kernel(float* d_input, float* d_output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
}
