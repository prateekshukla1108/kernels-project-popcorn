#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h> 

#define SEQ_LEN 128
#define DIM 64
#define BLOCK_SIZE 16
#define WARP_SIZE 32



void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

void verifyAttn(float *host_Q, float *host_K, float *host_V, float *host_output, int seq_len, int dim) {
    float *qk_matrix = (float*)malloc(seq_len * seq_len * sizeof(float));


    for (int row = 0; row < seq_len; row++) {
        for (int col = 0; col < seq_len; col++) {
            float score = 0.0f;
            for (int k = 0; k < dim; k++) {
                score += host_Q[row * dim + k] * host_K[col * dim + k];
            }
            qk_matrix[row * seq_len + col] = score / sqrtf((float)dim);
        }
    }


    for (int row = 0; row < seq_len; row++) {
        float max_val = -INFINITY;
        for (int col = 0; col < seq_len; col++) {
            max_val = fmaxf(max_val, qk_matrix[row * seq_len + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            qk_matrix[row * seq_len + col] = expf(qk_matrix[row * seq_len + col] - max_val);
            sum += qk_matrix[row * seq_len + col];
        }

        for (int d = 0; d < dim; d++) {
            float weighted_sum = 0.0f;
            for (int col = 0; col < seq_len; col++) {
                float attention_weight = qk_matrix[row * seq_len + col] / sum;
                weighted_sum += attention_weight * host_V[col * dim + d];
            }
            host_output[row * dim + d] = weighted_sum;
        }
    }

    free(qk_matrix);
}


__global__ void QK(const float* Q, const float* K, float* QK, const int seq_len, const int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int k = 0; k < dim; ++k) {
            sum += Q[row * dim + k] * K[col * dim + k];
        }
        QK[row * seq_len + col] = sum / sqrtf((float)dim);
    }
}


__global__ void SoftmaxAndMul(const float* QK, const float* V, float* output, const int seq_len, const int dim) {
    __shared__ float shared_qk[BLOCK_SIZE];
    __shared__ float shared_max_block[BLOCK_SIZE]; 
    __shared__ float shared_sum_block[BLOCK_SIZE]; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int localThreadId = threadIdx.x; 

    if (row < seq_len) {
        float max_val = -INFINITY;
        for (int col = localThreadId; col < seq_len; col += blockDim.x) {
            if (col < seq_len)
                max_val = fmaxf(max_val, QK[row * seq_len + col]);
        }
        shared_max_block[localThreadId] = max_val; 
        __syncthreads();

        
        if (localThreadId < WARP_SIZE) { 
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) { 
                if (localThreadId + offset < blockDim.x) {
                    shared_max_block[localThreadId] = fmaxf(shared_max_block[localThreadId], shared_max_block[localThreadId + offset]);
                }
                __syncthreads();
            }
        }
        __syncthreads();
        float row_max = shared_max_block[0]; 


        float sum = 0.0f;
        for (int col = localThreadId; col < seq_len; col += blockDim.x) {
            if (col < seq_len)
                shared_qk[localThreadId] = expf(QK[row * seq_len + col] - row_max);
            else
                shared_qk[localThreadId] = 0.0f;
            sum += shared_qk[localThreadId];
        }
        shared_sum_block[localThreadId] = sum; 
        __syncthreads();

        
        if (localThreadId < WARP_SIZE) {
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) { 
                if (localThreadId + offset < blockDim.x) {
                    shared_sum_block[localThreadId] += shared_sum_block[localThreadId + offset];
                }
                __syncthreads();
            }
        }
        __syncthreads();
        float row_sum = shared_sum_block[0]; 



        // if (row == 0 && localThreadId < 4) {
        //     printf("Row %d, Thread %d: row_max = %f, row_sum = %f\n", row, localThreadId, row_max, row_sum);
        // }

        for (int d = localThreadId; d < dim; d += blockDim.x) {
            if (d < dim) {
                float weighted_sum = 0.0f;
                for (int col = 0; col < seq_len; ++col) {
                    float attention_weight = expf(QK[row * seq_len + col] - row_max) / row_sum;
                    weighted_sum += attention_weight * V[col * dim + d];
                }
                output[row * dim + d] = weighted_sum;
            }
        }
    }
}

int main() {


    cudaSetDevice(0);

    int total_qk = SEQ_LEN * SEQ_LEN;
    int total_val = SEQ_LEN * DIM;
    size_t qk_size = total_qk * sizeof(float);
    size_t val_size = total_val * sizeof(float);

    float *host_Q, *host_K, *host_V, *host_output, *host_output_cpu;
    float *device_Q, *device_K, *device_V, *device_QK, *device_output;


    checkCudaError(cudaMallocHost(&host_Q, val_size), "cudaMallocHost Q");
    checkCudaError(cudaMallocHost(&host_K, val_size), "cudaMallocHost K");
    checkCudaError(cudaMallocHost(&host_V, val_size), "cudaMallocHost V");
    checkCudaError(cudaMallocHost(&host_output, val_size), "cudaMallocHost output");
    host_output_cpu = (float*)malloc(val_size);

    for (int i = 0; i < total_val; i++) {
        host_Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        host_K[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        host_V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }


    checkCudaError(cudaMalloc(&device_Q, val_size), "cudaMalloc Q");
    checkCudaError(cudaMalloc(&device_K, val_size), "cudaMalloc K");
    checkCudaError(cudaMalloc(&device_V, val_size), "cudaMalloc V");
    checkCudaError(cudaMalloc(&device_QK, qk_size), "cudaMalloc QK");
    checkCudaError(cudaMalloc(&device_output, val_size), "cudaMalloc output");


    checkCudaError(cudaMemcpy(device_Q, host_Q, val_size, cudaMemcpyHostToDevice), "cudaMemcpy Q");
    checkCudaError(cudaMemcpy(device_K, host_K, val_size, cudaMemcpyHostToDevice), "cudaMemcpy K");
    checkCudaError(cudaMemcpy(device_V, host_V, val_size, cudaMemcpyHostToDevice), "cudaMemcpy V");

    dim3 blockDim_qk(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim_qk((SEQ_LEN + blockDim_qk.x - 1) / blockDim_qk.x, (SEQ_LEN + blockDim_qk.y - 1) / blockDim_qk.y);
    dim3 blockDim_sm(BLOCK_SIZE, 1); 
    dim3 gridDim_sm( (DIM + blockDim_sm.x - 1) / blockDim_sm.x, SEQ_LEN);


    QK<<<gridDim_qk, blockDim_qk>>>(device_Q, device_K, device_QK, SEQ_LEN, DIM);
    checkCudaError(cudaGetLastError(), "QK kernel");


    float *host_QK_gpu = (float*)malloc(qk_size);
    checkCudaError(cudaMemcpy(host_QK_gpu, device_QK, qk_size, cudaMemcpyDeviceToHost), "cudaMemcpy device_QK to host");


    float *host_qk_matrix_cpu = (float*)malloc(SEQ_LEN * SEQ_LEN * sizeof(float));
    for (int row = 0; row < SEQ_LEN; row++) {
        for (int col = 0; col < SEQ_LEN; col++) {
            float score = 0.0f;
            for (int k = 0; k < DIM; k++) {
                score += host_Q[row * DIM + k] * host_K[col * DIM + k];
            }
            host_qk_matrix_cpu[row * SEQ_LEN + col] = score / sqrtf((float)DIM);
        }
    }


    // printf("\n--- Comparing QK Matrices (GPU vs CPU) ---\n");
    // for (int row = 0; row < 4; ++row) {
    //     printf("Row %d: GPU: ", row);
    //     for (int col = 0; col < 4; ++col) {
    //         printf("%f ", host_QK_gpu[row * SEQ_LEN + col]);
    //     }
    //     printf("  CPU: ");
    //     for (int col = 0; col < 4; ++col) {
    //         printf("%f ", host_qk_matrix_cpu[row * SEQ_LEN + col]);
    //     }
    //     printf("\n");
    // }


    SoftmaxAndMul<<<gridDim_sm, blockDim_sm>>>(device_QK, device_V, device_output, SEQ_LEN, DIM);
    checkCudaError(cudaGetLastError(), "SoftmaxAndMul kernel");


    checkCudaError(cudaMemcpy(host_output, device_output, val_size, cudaMemcpyDeviceToHost), "cudaMemcpy output");

    verifyAttn(host_Q, host_K, host_V, host_output_cpu, SEQ_LEN, DIM);

    float max_diff = 0.0f;
    for (int i = 0; i < total_val; i++) {
        float diff = fabs(host_output[i] - host_output_cpu[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    printf("Maximum difference between CPU and GPU results: %e\n", max_diff);

    cudaFreeHost(host_Q);
    cudaFreeHost(host_K);
    cudaFreeHost(host_V);
    cudaFreeHost(host_output);
    free(host_output_cpu);
    free(host_QK_gpu); 
    free(host_qk_matrix_cpu); 

    cudaFree(device_Q);
    cudaFree(device_K);
    cudaFree(device_V);
    cudaFree(device_QK);
    cudaFree(device_output);

    return 0;
}
