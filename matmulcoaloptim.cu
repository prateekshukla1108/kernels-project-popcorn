#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <string.h>

// Matrix dimensions 
const int M = 4096;
const int K = 4096;
const int N = 4096;

//dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);
//dim3 three_dblock(32, 32, 1);
//dim3 one_dblock(32*32);

// Update block and grid dimensions
const int BLOCKSIZE = 32;
dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
dim3 one_dblock(BLOCKSIZE * BLOCKSIZE);  // 1024 threads per block
dim3 three_dblock(BLOCKSIZE, BLOCKSIZE, 1);

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Add error checking macro for CUPTI
#define CHECK_CUPTI(err) \
    do { \
        CUptiResult _err = (err); \
        if (_err != CUPTI_SUCCESS) { \
            const char *errstr; \
            cuptiGetResultString(_err, &errstr); \
            fprintf(stderr, "CUPTI error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
            exit(1); \
        } \
    } while(0)

// Update to use performance API metrics
const char* METRIC_NAMES[] = {
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",     // SM utilization
    "dram__bytes.sum.per_second",                             // Memory bandwidth
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum"       // FMA instructions
};
const int NUM_METRICS = sizeof(METRIC_NAMES)/sizeof(METRIC_NAMES[0]);



bool verifyResults(float* C_gpu, const float* C_cpu, int M, int N) {
    const float epsilon = 1e-2;
    for(int i = 0; i<M*N; i++) {
        if(abs(C_gpu[i] - C_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

void initializeMatrices(float* A, float* B, int M, int K, int N) {
    // Corrected parameter order in initialization
    // initialize A to random 
    for (int i=0; i<M*K; i++){
        A[i] = rand() / (float)RAND_MAX;
    }

    // initialize B to identity
    for (int i=0; i<K*N; i++){
        if (i % (N+1) == 0){
            B[i] = 1;
        } else {
            B[i] = 0;
        }
    }
}

// Performance metrics structure
struct PerformanceMetrics {
    float kernel_time;      // ms
    float gflops;          // GFLOPS achieved
    float sm_efficiency;    // Occupancy (0-1)
    float dram_throughput; // GB/s
    float arithmetic_intensity; // FLOPS/byte
    float bandwidth_utilization; // Percentage of theoretical
    float compute_utilization;   // Percentage of theoretical
    float memory_bound_ratio;    // >1 means memory bound, <1 means compute bound
    bool correct;
};

// CUPTI initialization function
void initCupti() {
    CUpti_SubscriberHandle subscriber;
    CHECK_CUPTI(cuptiSubscribe(&subscriber, 
        [](void* userdata, CUpti_CallbackDomain domain, 
           CUpti_CallbackId cbid, const void* cbdata) {}, 
        nullptr));
}

__global__ void matmul_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
    int col_c = blockDim.x * blockIdx.x + threadIdx.x;
    int row_c = blockDim.y * blockIdx.y + threadIdx.y;

    if (col_c < N && row_c < M){
        float accu = 0.0f;
        for (int sum_index = 0; sum_index < K; sum_index+=1){
            accu += A[row_c * K + sum_index] * B[sum_index * N + col_c];
        }
        C[row_c * N + col_c] = accu;
    }
}

__global__ void matmul_coal(int M, int N, int K, const float *A, 
                           const float* B, float *C) {
    const int BLOCKSIZE = 32;
    
    // Change indexing to minimize jumps between iterations
    const int tid = threadIdx.x;
    const int row = blockIdx.y * BLOCKSIZE + tid / BLOCKSIZE;
    const int col = blockIdx.x * BLOCKSIZE + tid % BLOCKSIZE;
    
    /*
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Memory access pattern for first warp:\n");
        for (int k = 0; k < 3; k++) {
            printf("\nIteration %d:\n", k);
            for (int t = 0; t < 32; t++) {
                printf("Thread %d: A[%d], B[%d]\n", 
                    t, 
                    (blockIdx.y * BLOCKSIZE + t / BLOCKSIZE) * K + k,
                    t + k * BLOCKSIZE  // Key change: consecutive access within BLOCKSIZE
                );
            }
        }
    }
    */
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Process elements in BLOCKSIZE chunks to maintain coalescing
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col + k * N];
        }
        
        C[row * N + col] = sum;
    }
}

 // Start of Selection
__global__ void matmul_coal_optim(int M, int N, int K, const float *A, 
                                   const float* B, float *C) {
    const int BLOCKSIZE = 32;
    
    // Use a 1D thread index to derive 2D coordinates within the tile
    const int tid = threadIdx.x;
    const int localRow = tid >> 5;              // Equivalent to tid / 32
    const int localCol = tid & (BLOCKSIZE - 1);   // Equivalent to tid % 32
    const int globalRow = blockIdx.y * BLOCKSIZE + localRow;
    const int globalCol = blockIdx.x * BLOCKSIZE + localCol;
    
    // Allocate shared memory for A and B tiles
    __shared__ float sharedA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float sharedB[BLOCKSIZE][BLOCKSIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles in the K dimension
    // Loop over the tiles in the K dimension.
    // This loop divides the matrix multiplication task along the K dimension into smaller tiles,
    // each of size BLOCKSIZE. For example, if K = 100 and BLOCKSIZE = 32, then:
    //   (100 + 32 - 1) / 32 = 131 / 32, which equals 4 tiles.
    // This corresponds to:
    //   Tile 0: covers indices 0 to 31,
    //   Tile 1: covers indices 32 to 63,
    //   Tile 2: covers indices 64 to 95,
    //   Tile 3: covers indices 96 to 99 (with any extra elements padded with zeros).
    for (int tile = 0; tile < (K + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {
        // Calculate the column index for A and row index for B in this tile
        int tiledA_col = tile * BLOCKSIZE + localCol;
        int tiledB_row = tile * BLOCKSIZE + localRow;
        
        // Load element into shared memory from A if within bounds; otherwise zero-pad
        if (globalRow < M && tiledA_col < K) {
            sharedA[localRow][localCol] = A[globalRow * K + tiledA_col];
        } else {
            sharedA[localRow][localCol] = 0.0f;
        }
        
        // Load element into shared memory from B if within bounds; otherwise zero-pad
        if (tiledB_row < K && globalCol < N) {
            sharedB[localRow][localCol] = B[tiledB_row * N + globalCol];
        } else {
            sharedB[localRow][localCol] = 0.0f;
        }
        
        __syncthreads(); // Ensure the tile is completely loaded
        
        // Compute the partial product for this tile
        for (int k = 0; k < BLOCKSIZE; k++) {
            sum += sharedA[localRow][k] * sharedB[k][localCol];
        }
        
        __syncthreads(); // Ensure all threads complete computation before next tile load
    }
    
    // Write the result back to global memory if within the valid output matrix range
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}

typedef void (*KernelFunction)(int, int, int, const float*, const float*, float*);

float calculateOccupancy(int blockSize) {
    int device;
    cudaGetDevice(&device);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate theoretical and actual occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        matmul_naive,
        blockSize,
        0  // shared memory size
    );
    
    // Print detailed occupancy information
    printf("\n=== Occupancy Details ===\n");
    printf("Threads per block: %d\n", blockSize);
    printf("Max blocks per SM: %d\n", maxActiveBlocks);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Theoretical max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("Active warps per SM: %d\n", (maxActiveBlocks * blockSize) / 32);
    
    // Calculate actual occupancy
    float activeWarps = (float)(maxActiveBlocks * blockSize) / 32.0f;
    float maxWarps = (float)prop.maxThreadsPerMultiProcessor / 32.0f;
    float occupancy = activeWarps / maxWarps;
    
    // Print grid information
    printf("\n=== Grid Details ===\n");
    printf("Grid dimensions: %d x %d\n", (N + 32 - 1) / 32, (M + 32 - 1) / 32);
    printf("Total blocks: %d\n", ((N + 32 - 1) / 32) * ((M + 32 - 1) / 32));
    printf("Blocks per SM: %.2f\n", (float)(((N + 32 - 1) / 32) * ((M + 32 - 1) / 32)) / prop.multiProcessorCount);
    
    return occupancy;
}



const char* getCacheConfigString(cudaFuncCache cacheConfig) {
    switch(cacheConfig) {
        case cudaFuncCachePreferNone: return "No Preference";
        case cudaFuncCachePreferShared: return "Prefer Shared Memory";
        case cudaFuncCachePreferL1: return "Prefer L1 Cache";
        case cudaFuncCachePreferEqual: return "Equal L1 and Shared";
        default: return "Unknown";
    }
}

PerformanceMetrics runKernel(KernelFunction kernel, dim3 grid, dim3 block, const char* name, 
                            const float* h_A, const float* d_A, 
                            const float* d_B, float* d_C) 
{
    PerformanceMetrics metrics = {0};
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // Get device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Get current context and device
    CUcontext context;
    CUdevice cuDevice;
    cudaFuncCache cacheConfig;
    cuCtxGetCurrent(&context);
    cuCtxGetDevice(&cuDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // Run kernel
    kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);
    cudaDeviceGetCacheConfig(&cacheConfig);

    // Calculate basic metrics
    float operations = 2.0f * M * N * K;  // multiply-add counts as 2 operations
    size_t bytes_read = M * K * sizeof(float) + K * N * sizeof(float);
    size_t bytes_written = M * N * sizeof(float);
    float total_bytes = bytes_read + bytes_written;
    float seconds = metrics.kernel_time / 1000.0f;

    // Calculate GFLOPS
    metrics.gflops = (operations / 1e9) / seconds;

    // Calculate memory throughput
    metrics.dram_throughput = total_bytes / (seconds * 1e9);  // GB/s

    // Calculate arithmetic intensity (FLOPS/byte)
    metrics.arithmetic_intensity = operations / total_bytes;

    // Calculate theoretical peak performance
    float clock_rate = prop.clockRate * 1e3;  // Convert kHz to Hz
    float theoretical_gflops = (prop.multiProcessorCount * prop.warpSize * 2 * clock_rate) / 1e9;
    
    // Calculate theoretical memory bandwidth (GB/s)
    // Memory clock is in kHz, bus width in bits
    float theoretical_bandwidth = (float)(prop.memoryClockRate * 1000.0) *  // Convert to Hz
                                (prop.memoryBusWidth / 8.0) *               // Convert bits to bytes
                                2.0 /                                       // DDR factor
                                1.0e9;                                     // Convert to GB/s

    // Calculate utilization percentages
    metrics.bandwidth_utilization = (metrics.dram_throughput / theoretical_bandwidth) * 100.0f;
    metrics.compute_utilization = (metrics.gflops / theoretical_gflops) * 100.0f;

    // Calculate if kernel is memory or compute bound
    float time_compute = operations / (theoretical_gflops * 1e9);
    float time_memory = total_bytes / (theoretical_bandwidth * 1e9);
    metrics.memory_bound_ratio = time_memory / time_compute;

    // Calculate occupancy
    //metrics.sm_efficiency = calculateOccupancy(block.x * block.y);

    // Copy results back and verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    metrics.correct = verifyResults(h_C, h_A, M, N);

    // Print detailed performance analysis
    printf("\n=== Performance Analysis (%s) ===\n", name);
    printf("Basic Metrics:\n");
    printf("  Kernel Time: %.3f ms\n", metrics.kernel_time);
    printf("  GFLOPS: %.2f\n", metrics.gflops);
    printf("  Memory Throughput: %.2f GB/s\n", metrics.dram_throughput);
    //printf("  SM Occupancy: %.2f%%\n", metrics.sm_efficiency * 100.0f);
    
    printf("\nAdvanced Metrics:\n");
    //printf("  Arithmetic Intensity: %.2f FLOPS/byte\n", metrics.arithmetic_intensity);
    //printf("  Theoretical Peak: %.2f GFLOPS\n", theoretical_gflops);
    //printf("  Theoretical Bandwidth: %.2f GB/s\n", theoretical_bandwidth);
    printf("  Compute Utilization: %.2f%%\n", metrics.compute_utilization);
    printf("  Memory Bandwidth Utilization: %.2f%%\n", metrics.bandwidth_utilization);
    printf("  Memory/Compute Bound Ratio: %.2f %s\n", 
           metrics.memory_bound_ratio,
           metrics.memory_bound_ratio > 1.0f ? "(Memory Bound)" : "(Compute Bound)");
    
    printf("\nCorrectness: %s\n", metrics.correct ? "PASS" : "FAIL");
    //printf("\nCache Configuration: %s\n", getCacheConfigString(cacheConfig));

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C);

    return metrics;
}



int main() {
    // Initialize CUDA driver API
    cuInit(0);
    
    // define host pointers 
    float *h_A, *h_B, *h_C;
    // define device pointers 
    float *d_A, *d_B, *d_C;

    
    //initialise matrices sizes 
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    // allocate memory on the CPU
    h_A = (float*)malloc(size_a);
    h_B = (float*)malloc(size_b);
    h_C = (float*)malloc(size_c);
    
    // allocate memory on GPU
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_a));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_b));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_c));

    initializeMatrices(h_A, h_B, M, K, N);
    // send data to GPU 
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    printf("Going to run naive kernel\n");  // Add newline for proper output
    PerformanceMetrics metrics = runKernel(matmul_naive, grid, three_dblock, "matmul_naive", h_A, d_A, d_B, d_C);
    printf("Done running naive kernel\n");
    printf("Going to run coalesced kernel\n");
    metrics = runKernel(matmul_coal, grid, one_dblock, "matmul_coal", h_A, d_A, d_B, d_C);
    printf("Done running coalesced kernel\n");
    metrics = runKernel(matmul_coal_optim, grid, one_dblock, "matmul_coal_optim", h_A, d_A, d_B, d_C);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
