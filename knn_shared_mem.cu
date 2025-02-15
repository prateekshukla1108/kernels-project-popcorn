#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define NUM_POINTS 1024
#define DIMENSIONS 3
#define NEAREST_NEIGHBORS 5
#define THREADS_PER_BLOCK 256
#define MAX_CLASSES 10

__global__ void knn_optimized(float *train_data, float *test_data, int *labels, int *predictions, 
                              int train_size, int test_size, int feature_size, int k) {
    extern __shared__ char shared_mem[];
    float* shared_train = (float*)shared_mem;
    int* shared_labels = (int*)&shared_train[train_size * feature_size];
    
    int tid = threadIdx.x;
    int test_idx = blockIdx.x * blockDim.x + tid;
    if (test_idx >= test_size) return;

    
    for (int i = tid; i < train_size * feature_size; i += blockDim.x) {
        shared_train[i] = train_data[i];
    }
    
    for (int i = tid; i < train_size; i += blockDim.x) {
        shared_labels[i] = labels[i];
    }
    __syncthreads();


    float top_dist[NEAREST_NEIGHBORS];
    int top_labels[NEAREST_NEIGHBORS];
    for (int i = 0; i < k; ++i) {
        top_dist[i] = FLT_MAX;
        top_labels[i] = -1;
    }


    for (int i = 0; i < train_size; ++i) {
        float dist_sq = 0.0f;
        #pragma unroll
        for (int f = 0; f < feature_size; ++f) {
            float diff = test_data[test_idx * feature_size + f] - shared_train[i * feature_size + f];
            dist_sq += diff * diff;
        }

        if (dist_sq < top_dist[k-1]) {
            top_dist[k-1] = dist_sq;
            top_labels[k-1] = shared_labels[i];
            

            for (int j = k-1; j > 0 && top_dist[j] < top_dist[j-1]; --j) {
                float d_tmp = top_dist[j];
                top_dist[j] = top_dist[j-1];
                top_dist[j-1] = d_tmp;
                
                int l_tmp = top_labels[j];
                top_labels[j] = top_labels[j-1];
                top_labels[j-1] = l_tmp;
            }
        }
    }


    int votes[MAX_CLASSES] = {0};
    for (int i = 0; i < k; ++i) {
        if (top_labels[i] >= 0 && top_labels[i] < MAX_CLASSES) {
            votes[top_labels[i]]++;
        }
    }

    int best_label = 0;
    for (int i = 1; i < MAX_CLASSES; ++i) {
        if (votes[i] > votes[best_label]) best_label = i;
    }
    predictions[test_idx] = best_label;
}
bool verify(float* host_train_data, float* host_test_data, int* host_labels, int* host_predictions_gpu, int train_size, int test_size, int feature_size, int k) {
    int* host_predictions_cpu = new int[test_size];
    for (int i = 0; i < test_size; ++i) {
        float distances[train_size];
        int neighbor_labels[train_size];

        for (int j = 0; j < train_size; ++j) {
            float dist = 0.0f;
            for (int f = 0; f < feature_size; ++f) {
                float diff = host_test_data[i * feature_size + f] - host_train_data[j * feature_size + f];
                dist += diff * diff;
            }
            distances[j] = sqrtf(dist);
            neighbor_labels[j] = host_labels[j];
        }

        for (int m = 0; m < k; ++m) {
            for (int n = m + 1; n < train_size; ++n) {
                if (distances[n] < distances[m]) {
                    float temp_dist = distances[m];
                    distances[m] = distances[n];
                    distances[n] = temp_dist;

                    int temp_label = neighbor_labels[m];
                    neighbor_labels[m] = neighbor_labels[n];
                    neighbor_labels[n] = temp_label;
                }
            }
        }
        int votes[10] = {0};
        for (int v = 0; v < k; ++v) {
            votes[neighbor_labels[v]]++;
        }

        int best_label = 0;
        for (int v = 1; v < 10; ++v) {
            if (votes[v] > votes[best_label]) {
                best_label = v;
            }
        }
        host_predictions_cpu[i] = best_label;
    }

    bool match = true;
    for (int i = 0; i < test_size; ++i) {
        if (host_predictions_cpu[i] != host_predictions_gpu[i]) {
            printf("Verification failed at index %d: CPU: %d, GPU: %d\n", i, host_predictions_cpu[i], host_predictions_gpu[i]);
            match = false;
            break;
        }
    }
    delete[] host_predictions_cpu;
    return match;
}
void launchKNN(float *host_train, float *host_test, int *host_labels, int *host_predictions, 
               int train_size, int test_size, int feature_size, int k) {
    float *d_train, *d_test;
    int *d_labels, *d_predictions;

    cudaMalloc(&d_train, train_size * feature_size * sizeof(float));
    cudaMalloc(&d_test, test_size * feature_size * sizeof(float));
    cudaMalloc(&d_labels, train_size * sizeof(int));
    cudaMalloc(&d_predictions, test_size * sizeof(int));

    cudaMemcpy(d_train, host_train, train_size * feature_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, host_test, test_size * feature_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, host_labels, train_size * sizeof(int), cudaMemcpyHostToDevice);

    size_t shared_mem_size = train_size * feature_size * sizeof(float) + train_size * sizeof(int);
    int blocks = (test_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    knn_optimized<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_train, d_test, d_labels, d_predictions, 
                                                                  train_size, test_size, feature_size, k);

    cudaMemcpy(host_predictions, d_predictions, test_size * sizeof(int), cudaMemcpyDeviceToHost);

    int blocks = (test_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    knn_global_memory<<<blocks, THREADS_PER_BLOCK>>>(device_train_data, device_test_data, device_labels, device_predictions, train_size, test_size, feature_size, k);

    cudaMemcpy(host_predictions, device_predictions, test_size * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(host_train_data, host_test_data, host_labels, host_predictions, train_size, test_size, feature_size, k)) {
        printf("Verification passed!\n");
    } else {
        printf("Verification failed!\n");
    }

    
    cudaFree(d_train);
    cudaFree(d_test);
    cudaFree(d_labels);
    cudaFree(d_predictions);
}

int main() {
    srand(time(NULL));

    int train_size = NUM_POINTS;
    int test_size = NUM_POINTS;
    int feature_size = DIMENSIONS;
    int k = NEAREST_NEIGHBORS;

    float* host_train_data = (float*)malloc(train_size * feature_size * sizeof(float));
    float* host_test_data = (float*)malloc(test_size * feature_size * sizeof(float));
    int* host_labels = (int*)malloc(train_size * sizeof(int));
    int* host_predictions = (int*)malloc(test_size * sizeof(int));

    for (int i = 0; i < train_size * feature_size; ++i) {
        host_train_data[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < test_size * feature_size; ++i) {
        host_test_data[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < train_size; ++i) {
        host_labels[i] = rand() % 10;
    }

    launchKNN(host_train_data, host_test_data, host_labels, host_predictions, train_size, test_size, feature_size, k);

    for (int i = 0; i < test_size; ++i) {
        printf("Prediction for test point %d: %d\n", i, host_predictions[i]);
    }

    free(host_train_data);
    free(host_test_data);
    free(host_labels);
    free(host_predictions);

    return 0;
}
