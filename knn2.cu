#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_POINTS 1024
#define DIMENSIONS 3
#define NEAREST_NEIGHBORS 5

__global__ void computeDistances(float *dataset, float *query, float *distances, int numPoints, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float dist = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            float diff = dataset[idx * dimensions + i] - query[i];
            dist += diff * diff;
        }
        distances[idx] = sqrtf(dist);
    }
}

bool verify(float* host_dataset, float* host_query, float* host_distances_cpu, float* host_distances_gpu, int numPoints, int dimensions) {
    for (int i = 0; i < numPoints; ++i) {
        float dist_cpu = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            dist_cpu += powf(host_dataset[i * dimensions + j] - host_query[j], 2.0f);
        }
        dist_cpu = sqrtf(dist_cpu);

        if (fabsf(dist_cpu - host_distances_gpu[i]) > 1e-5) {
            printf("Verification failed at index %d: CPU: %f, GPU: %f\n", i, dist_cpu, host_distances_gpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float host_dataset[NUM_POINTS * DIMENSIONS], host_query[DIMENSIONS], host_distances[NUM_POINTS];
    float *device_dataset, *device_query, *device_distances;

    for (int i = 0; i < NUM_POINTS * DIMENSIONS; i++) host_dataset[i] = static_cast<float>(rand() % 100) / 10.0f;
    for (int i = 0; i < DIMENSIONS; i++) host_query[i] = static_cast<float>(rand() % 100) / 10.0f;

    cudaMalloc(&device_dataset, NUM_POINTS * DIMENSIONS * sizeof(float));
    cudaMalloc(&device_query, DIMENSIONS * sizeof(float));
    cudaMalloc(&device_distances, NUM_POINTS * sizeof(float));

    cudaMemcpy(device_dataset, host_dataset, NUM_POINTS * DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_query, host_query, DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_POINTS + threadsPerBlock - 1) / threadsPerBlock;
    computeDistances<<<blocksPerGrid, threadsPerBlock>>>(device_dataset, device_query, device_distances, NUM_POINTS, DIMENSIONS);

    cudaMemcpy(host_distances, device_distances, NUM_POINTS * sizeof(float), cudaMemcpyDeviceToHost);

    float host_distances_cpu[NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        float dist = 0.0f;
        for (int j = 0; j < DIMENSIONS; j++) {
            dist += powf(host_dataset[i * DIMENSIONS + j] - host_query[j], 2.0f);
        }
        host_distances_cpu[i] = sqrtf(dist);
    }

    if (verify(host_dataset, host_query, host_distances_cpu, host_distances, NUM_POINTS, DIMENSIONS)) {
        printf("Verification passed!\n");
    } else {
        printf("Verification failed!\n");
        return 1;
    }

    int nearest_neighbor_indices[NEAREST_NEIGHBORS];
    for (int i = 0; i < NEAREST_NEIGHBORS; i++) nearest_neighbor_indices[i] = i;

    for (int i = 0; i < NEAREST_NEIGHBORS; i++) {
        for (int j = i + 1; j < NUM_POINTS; j++) {
            if (host_distances[j] < host_distances[nearest_neighbor_indices[i]]) {
                nearest_neighbor_indices[i] = j;
            }
        }
    }

    printf("Query point: ");
    for (int i = 0; i < DIMENSIONS; i++) printf("%f ", host_query[i]);
    printf("\nK-Nearest Neighbors:\n");
    for (int i = 0; i < NEAREST_NEIGHBORS; i++) {
        printf("Point %d (Distance: %f)\n", nearest_neighbor_indices[i], host_distances[nearest_neighbor_indices[i]]);
    }

    cudaFree(device_dataset);
    cudaFree(device_query);
    cudaFree(device_distances);

    return 0;
}
