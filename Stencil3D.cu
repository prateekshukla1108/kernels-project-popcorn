#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_SIZE 8
#define PADDING 1
#define SHARED_SIZE (TILE_SIZE + 2 * PADDING)

__global__ void Stencil3DKernel(float *d_input, float *d_output, int Z){

    // local idx
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    // global idx
    int gx = blockIdx.x * TILE_SIZE + tx - PADDING;
    int gy = blockIdx.y * TILE_SIZE + ty - PADDING;
    int gz = blockIdx.z * TILE_SIZE + tz - PADDING;

    // shared memory
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE][SHARED_SIZE];

    if (gx >= 0 && gx < Z && gy >= 0 && gy < Z && gz >= 0 && gz < Z) {

    tile[tz][ty][tx] = d_input[(gz * Z + gy) * Z + gx];
    } else {
    tile[tz][ty][tx] = 0.0f;

    }
    __syncthreads();

    if (gz >= 1 && gz<Z-1 && gy >= 1 && gy < Z-1 && gx >= 1 && gx < Z-1) {

        // Onty compute for global points that are not on the boundary.
       if (tz >= 1 && tz<(TILE_SIZE + 1) && ty >= 1 && ty < (TILE_SIZE + 1) && tx >= 1 && tx < (TILE_SIZE + 1)) {

            // Access the 7 neighbors: center, plus one in each cardinal direction.
            float center = tile[tz][ty][tx];
            float up     = tile[tz - 1][ty][tx];
            float down   = tile[tz + 1][ty][tx];
            float left   = tile[tz][ty][tx - 1];
            float right  = tile[tz][ty][tx + 1];
            float front  = tile[tz][ty - 1][tx];
            float back   = tile[tz][ty + 1][tx];

            float result = center + up + down + left + right + front + back;

            // Write the result back to global memory.
            d_output[(gz * Z + gy) * Z + gx] = result;
        }
    }
}

int main() {

    const int Z = 64;
    const int numVoxels = Z * Z * Z;
    const size_t volumeSize = numVoxels * sizeof(float);
    srand(time(0)); 

    // Allocate host memory and initialize the volume (for example, all ones).
    float *h_input  = new float[numVoxels];
    float *h_output = new float[numVoxels];
    for (int i = 0; i < numVoxels; ++i) {
        h_input[i] = static_cast<float>(rand() % 10 + 1);  // random numbers 1 to 10
    }
//     for (int z = 0; z < Z; z++) {
//     cout << "Input Slice z = " << z << ":\n";
//     for (int y = 0; y < Z; y++) {
//         for (int x = 0; x < Z; x++) {
//             cout << h_input[(z * Z + y) * Z + x] << " ";
//         }
//         cout << "\n";
//     }
//     cout << "\n";
// }

    // Allocate device memory.
    float *d_input, *d_output;
    cudaMalloc(&d_input, volumeSize);
    cudaMalloc(&d_output, volumeSize);

    // Copy input data to device.
    cudaMemcpy(d_input, h_input, volumeSize, cudaMemcpyHostToDevice);

    // Calculate grid dimensions.
    dim3 blockDim(SHARED_SIZE, SHARED_SIZE, SHARED_SIZE);  // (TILE_SIZE+2)^3 threads per block.
    dim3 gridDim((Z + TILE_SIZE - 1) / TILE_SIZE,
                 (Z + TILE_SIZE - 1) / TILE_SIZE,
                 (Z + TILE_SIZE - 1) / TILE_SIZE);
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch your kernel
    Stencil3DKernel<<<gridDim, blockDim>>>(d_input, d_output, Z);

    // Record the stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU operation time: " << elapsedTime << " ms" << endl;

    // Destroy the events when done
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_output, volumeSize, cudaMemcpyDeviceToHost);

    // for (int z = 0; z < Z; z++) {
//     cout << "Output Slice z = " << z << ":\n";
//     for (int y = 0; y < Z; y++) {
//         for (int x = 0; x < Z; x++) {
//             cout << h_output[(z * Z + y) * Z + x] << " ";
//         }
//         cout << "\n";
//     }
//     cout << "\n";
// }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
