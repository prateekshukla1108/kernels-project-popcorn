#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>



// CUDA kernel to compute QK^T (Query-Key dot product)
__global__ void qkDotKernel(float *Q, float *K, float *QK, int d) {
    int row = blockIdx.x;  // Query index
    int col = threadIdx.x; // Key index

    if (col < SEQ_LEN) {
        float sum = 0.0;
        for (int i = 0; i < d; i++) {
            sum += Q[row * d + i] * K[col * d + i];  // Dot product
        }
        QK[row * SEQ_LEN + col] = sum / sqrtf(d);  // Scale by sqrt(d_k)
    }
}

__global__ void softmaxKernel(float *QK, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    float sum = 0.0;
    for (int i = 0; i < SEQ_LEN; i++) {
        sum += expf(QK[idx * SEQ_LEN + i]);
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        QK[idx * SEQ_LEN + i] = expf(QK[idx * SEQ_LEN + i]) / sum;  // Softmax
    }
}

__global__ void attentionKernel(float *QK, float *V, float *output, int d) {
    int row = blockIdx.x;  // Token index
    int col = threadIdx.x; // Embedding index

    if (col < d) {
        float sum = 0.0;
        for (int i = 0; i < SEQ_LEN; i++) {
            sum += QK[row * SEQ_LEN + i] * V[i * d + col];  // Weighted sum
        }
        output[row * d + col] = sum;
    }
}



