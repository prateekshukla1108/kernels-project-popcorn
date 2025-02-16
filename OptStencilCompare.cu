#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>

using namespace std;

#define TILE_SIZE 8
#define PADDING 1
#define SHARED_SIZE (TILE_SIZE + 2 * PADDING)
#define COARSE_FACTOR 2  

__global__ void Stencil3D_Original(float *d_input, float *d_output, int Z) {
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gx = blockIdx.x * TILE_SIZE + tx - PADDING;
    int gy = blockIdx.y * TILE_SIZE + ty - PADDING;
    int gz = blockIdx.z * TILE_SIZE + tz - PADDING;

    __shared__ float tile[SHARED_SIZE][SHARED_SIZE][SHARED_SIZE];

    if (gx >= 0 && gx < Z && gy >= 0 && gy < Z && gz >= 0 && gz < Z) {
        tile[tz][ty][tx] = d_input[(gz * Z + gy) * Z + gx];
    } else {
        tile[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    if (gz >= 1 && gz < Z-1 && gy >= 1 && gy < Z-1 && gx >= 1 && gx < Z-1) {
        if (tz >= 1 && tz < SHARED_SIZE-1 && 
            ty >= 1 && ty < SHARED_SIZE-1 && 
            tx >= 1 && tx < SHARED_SIZE-1) {
            
            float sum = 
                tile[tz][ty][tx] +        
                tile[tz-1][ty][tx] +      
                tile[tz+1][ty][tx] +      
                tile[tz][ty][tx-1] +      
                tile[tz][ty][tx+1] +      
                tile[tz][ty-1][tx] +      
                tile[tz][ty+1][tx];       
            
            d_output[(gz * Z + gy) * Z + gx] = sum;
        }
    }
}

// Optimized kernel with 3 arrays, coarsening, and register tiling
__global__ void Stencil3D_Optimized(float *d_input, float *d_output, int Z) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_SIZE + tx - PADDING;
    int gy = blockIdx.y * TILE_SIZE + ty - PADDING;
    int gz_start = 1 + blockIdx.z * COARSE_FACTOR;

    // Shared memory for current plane only
    __shared__ float inCurr_s[SHARED_SIZE][SHARED_SIZE];
    
    // Register storage for previous and next planes
    float inPrev[COARSE_FACTOR], inNext[COARSE_FACTOR];

    for(int c = 0; c < COARSE_FACTOR; c++) {
        int gz = gz_start + c;
        
        // Load previous plane (register)
        if (gz - 1 >= 0 && gy >= 0 && gy < Z && gx >= 0 && gx < Z)
            inPrev[c] = d_input[((gz-1) * Z + gy) * Z + gx];
        else
            inPrev[c] = 0.0f;

        // Load current plane (shared memory)
        if (gz >= 0 && gz < Z && gy >= 0 && gy < Z && gx >= 0 && gx < Z)
            inCurr_s[ty][tx] = d_input[(gz * Z + gy) * Z + gx];
        else
            inCurr_s[ty][tx] = 0.0f;


        // Load next plane (register)
        if(gz+1 >= 0 && gz+1 < Z && gy >= 0 && gy < Z && gx >= 0 && gx < Z)
            inNext[c] = d_input[((gz+1) * Z + gy) * Z + gx];
        else
            inNext[c] = 0.0f;

        __syncthreads();

        // Compute stencil for coarsened elements
        if(gz >= 1 && gz < Z-1 && gy >= 1 && gy < Z-1 && gx >= 1 && gx < Z-1) {
            if(ty >= 1 && ty < SHARED_SIZE-1 && tx >= 1 && tx < SHARED_SIZE-1) {
                float sum = 
                    inCurr_s[ty][tx] +          // center
                    inCurr_s[ty-1][tx] +        // front
                    inCurr_s[ty+1][tx] +        // back
                    inCurr_s[ty][tx-1] +        // left
                    inCurr_s[ty][tx+1] +        // right
                    inPrev[c] +                 // previous z
                    inNext[c];                  // next z

                d_output[(gz * Z + gy) * Z + gx] = sum;
            }
        }
        __syncthreads();
    }
}

int main() {
    const int Z = 32;
    const int numVoxels = Z * Z * Z;
    const size_t volumeSize = numVoxels * sizeof(float);
    
    // Allocate and initialize host memory
    float *h_input = new float[numVoxels];
    float *h_output_orig = new float[numVoxels];
    float *h_output_opt = new float[numVoxels];
    
    for(int i = 0; i < numVoxels; ++i) {
        h_input[i] = static_cast<float>(rand() % 10 + 1);
    }

    // device memory
    float *d_input, *d_output_orig, *d_output_opt;
    cudaMalloc(&d_input, volumeSize);
    cudaMalloc(&d_output_orig, volumeSize);
    cudaMalloc(&d_output_opt, volumeSize);
    cudaMemcpy(d_input, h_input, volumeSize, cudaMemcpyHostToDevice);
    cudaMemset(d_output_orig, 0, volumeSize);
    cudaMemset(d_output_opt, 0, volumeSize);    


    dim3 blockOrig(SHARED_SIZE, SHARED_SIZE, SHARED_SIZE);
    dim3 gridOrig(
        (Z + TILE_SIZE - 1) / TILE_SIZE,
        (Z + TILE_SIZE - 1) / TILE_SIZE,
        (Z + TILE_SIZE - 1) / TILE_SIZE
    );

    dim3 blockOpt(SHARED_SIZE, SHARED_SIZE);
    dim3 gridOpt(
        (Z + TILE_SIZE - 1) / TILE_SIZE,
        (Z + TILE_SIZE - 1) / TILE_SIZE,
        ( (Z - 2) + COARSE_FACTOR - 1 ) / COARSE_FACTOR
);


    // Timing variables
    cudaEvent_t start, stop;
    float elapsed_orig, elapsed_opt;

    // Run original kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Stencil3D_Original<<<gridOrig, blockOrig>>>(d_input, d_output_orig, Z);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_orig, start, stop);
    
    cudaEventRecord(start);
    Stencil3D_Optimized<<<gridOpt, blockOpt>>>(d_input, d_output_opt, Z);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_opt, start, stop);

    cudaMemcpy(h_output_orig, d_output_orig, volumeSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_opt, d_output_opt, volumeSize, cudaMemcpyDeviceToHost);

    // Verify results
    bool valid = true;
        for(int i = 0; i < numVoxels; i++) {
            if(abs(h_output_orig[i] - h_output_opt[i]) > 1e-5) {
                valid = false;
        break;
    }
}


    cout << "Original kernel time: " << elapsed_orig << " ms\n";
    cout << "Optimized kernel time: " << elapsed_opt << " ms\n";
    cout << "Speedup: " << elapsed_orig/elapsed_opt << "x\n";
    cout << "Results valid: " << (valid ? "Yes" : "No") << endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output_orig);
    cudaFree(d_output_opt);
    delete[] h_input;
    delete[] h_output_orig;
    delete[] h_output_opt;

    return 0;
}
