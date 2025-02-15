// vector_addition.cu
//
// This program demonstrates vector addition on the GPU using two approaches:
//  1. Using cuBLAS's SAXPY routine (which computes y = alpha * x + y)
//  2. Using a custom CUDA kernel (vectorAddKernel)
// 
// The code compares the execution times of both methods.
//
// Basic constructs and their advantages:
//
// 1. CUDA_CALL macro:
//    - Purpose: Wraps CUDA runtime API calls to check for errors immediately.
//    - Advantage: Automatically verifies that each CUDA call completes successfully.
//      If a call fails, it prints the file name, line number, and error message and then exits.
//    - Benefit: Simplifies error handling and ensures that issues are caught early,
//      avoiding silent errors.
//
// 2. CUBLAS_CALL macro:
//    - Purpose: Wraps cuBLAS library calls and checks their return status.
//    - Advantage: cuBLAS functions return a status code, but do not print errors on failure.
//      This macro handles that by reporting the error location and exiting if the call fails.
//    - Benefit: Ensures robust error checking when using the highly optimized cuBLAS routines.
//
// 3. SAXPY Routine (cublasSaxpy):
//    - SAXPY stands for "Single-precision A * X Plus Y".
//    - It performs the operation: y = alpha * x + y.
//      By setting alpha = 1.0, we effectively compute vector addition: y = x + y.
//    - Advantage: cuBLAS's implementation is highly tuned for NVIDIA GPUs, leveraging
//      specialized optimizations and hardware features (like vectorized instructions and
//      even tensor cores on newer GPUs).
//    - Benefit: Instead of writing your own vector-add kernel, using cuBLAS's SAXPY can
//      yield better performance on large arrays or in applications where such routines
//      are heavily optimized.
//    - Comparison: A custom CUDA kernel can be simple and have low overhead but may lack
//      the additional optimizations available in cuBLAS.
// 
// 4. Custom CUDA Kernel (vectorAddKernel):
//    - Provides a straightforward implementation of vector addition.
//    - Advantage: Offers more control over the kernel code and has minimal API overhead.
//    - Benefit: Useful for simple or specialized operations, though it might not match the
//      performance of an optimized library call for standard tasks.
//
// The code also uses CUDA events to time the execution of both methods for a fair comparison.

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --------------------------------------------------------------------------
// Macro: CUDA_CALL
// This macro wraps any CUDA runtime API call to check for errors.
// It does the following:
//   - Executes the CUDA API call.
//   - Checks if the returned error code is cudaSuccess.
//   - If an error is detected, it prints an error message with file and line details
//     and then exits the program.
//
// This is advantageous because it centralizes error checking and prevents the need to
// manually check the return status of every CUDA call, reducing boilerplate code.
#define CUDA_CALL(call)                                                        \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " at line "            \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// --------------------------------------------------------------------------
// Macro: CUBLAS_CALL
// This macro wraps cuBLAS API calls in the same fashion as CUDA_CALL.
// It executes the cuBLAS function, checks whether the return value equals
// CUBLAS_STATUS_SUCCESS, and if not, prints an error message with file and line details.
// 
// This is important because cuBLAS routines are optimized but do not provide
// their own error messages; this macro ensures that any problems are caught.
#define CUBLAS_CALL(call)                                                      \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "CUBLAS error in " << __FILE__ << " at line "          \
                      << __LINE__ << std::endl;                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// --------------------------------------------------------------------------
// Custom CUDA kernel for vector addition.
// Each thread computes one element of the resulting vector:
//   y[i] = x[i] + y[i]
__global__ void vectorAddKernel(const float *x, float *y, int n) {
    // Compute the global thread index.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Ensure we don't access beyond the vector boundaries.
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    // Number of elements (size of vectors)
    const int n = 1 << 20;  // Approximately 1 million elements.
    size_t size = n * sizeof(float);

    // Allocate host memory (CPU) for vectors x and y.
    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    // This buffer will be used to copy back results from the GPU for verification.
    float *h_y_result = (float*)malloc(size);

    // Initialize host arrays:
    // Fill vector x with 1.0f and vector y with 2.0f.
    // The expected result after addition is a vector with all elements equal to 3.0f.
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // --------------------------------------------------------------------------
    // Allocate device (GPU) memory for vectors x and y.
    float *d_x, *d_y;
    CUDA_CALL(cudaMalloc((void**)&d_x, size));
    CUDA_CALL(cudaMalloc((void**)&d_y, size));

    // Copy the host data to device memory.
    CUDA_CALL(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------
    // Create CUDA events for timing the GPU operations.
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // --------------------------------------------------------------------------
    // ****************** Using cuBLAS (SAXPY) for Vector Addition ********************
    //
    // The cuBLAS library includes the SAXPY routine. SAXPY stands for:
    //    Single-Precision A * X Plus Y.
    // It computes: y = alpha * x + y.
    // By setting alpha to 1.0, the operation becomes y = x + y, which is the vector addition.
    //
    // Advantages of using SAXPY via cuBLAS:
    //   - It is highly optimized for NVIDIA GPUs.
    //   - It leverages advanced hardware features and vectorized instructions.
    //   - It abstracts low-level optimization details, making code more concise and reliable.
    //   - However, it may have a slight overhead due to library management.
    //
    // Create a cuBLAS handle to initialize the library context.
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    // Synchronize the device before timing to ensure that previous work is complete.
    CUDA_CALL(cudaDeviceSynchronize());

    // Record the start time.
    CUDA_CALL(cudaEventRecord(start, 0));
    
    float alpha = 1.0f;  // Scalar value for SAXPY.
    // Call cuBLAS SAXPY routine:
    //   - 'n' is the number of elements.
    //   - '&alpha' is a pointer to the scalar multiplier.
    //   - 'd_x' is the device pointer to vector x.
    //   - The stride is 1, meaning the data is contiguous.
    //   - 'd_y' is the device pointer to vector y.
    // This call performs the addition: d_y = 1.0 * d_x + d_y.
    CUBLAS_CALL(cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1));
    
    // Record the stop time and wait for the operation to finish.
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));

    float msec_cublas = 0;
    CUDA_CALL(cudaEventElapsedTime(&msec_cublas, start, stop));
    std::cout << "Time taken by cuBLAS (SAXPY) vector addition: " 
              << msec_cublas << " ms" << std::endl;

    // Copy the result from the device back to the host for later verification.
    CUDA_CALL(cudaMemcpy(h_y_result, d_y, size, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------------------------
    // Reset the device vector d_y to its original values (2.0f) for fair kernel timing.
    CUDA_CALL(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------
    // ***************** Using a Custom CUDA Kernel for Vector Addition ********
    //
    // Instead of using cuBLAS, we can write a custom kernel for vector addition.
    // While this approach is more direct and may have lower API overhead,
    // it typically lacks the extensive optimizations provided by cuBLAS.
    //
    // Determine grid and block dimensions:
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Synchronize before launching the kernel.
    CUDA_CALL(cudaDeviceSynchronize());

    // Record the start time.
    CUDA_CALL(cudaEventRecord(start, 0));

    // Launch the custom kernel. Each thread computes one element.
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);

    // Record the stop event and synchronize.
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));

    float msec_kernel = 0;
    CUDA_CALL(cudaEventElapsedTime(&msec_kernel, start, stop));
    std::cout << "Time taken by custom CUDA kernel vector addition: " 
              << msec_kernel << " ms" << std::endl;

    // --------------------------------------------------------------------------
    // Verification: Check that each element of the result vector equals 3.0f.
    CUDA_CALL(cudaMemcpy(h_y_result, d_y, size, cudaMemcpyDeviceToHost));
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_y_result[i] - 3.0f) > 1e-5) {
            std::cerr << "Verification error at index " << i 
                      << ": " << h_y_result[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition result is verified successfully." << std::endl;
    } else {
        std::cerr << "Vector addition result verification failed." << std::endl;
    }

    // --------------------------------------------------------------------------
    // Cleanup: Release all allocated resources.
    CUBLAS_CALL(cublasDestroy(handle)); // Destroy the cuBLAS handle.
    CUDA_CALL(cudaFree(d_x));           // Free device memory for vector x.
    CUDA_CALL(cudaFree(d_y));           // Free device memory for vector y.
    free(h_x);  // Free host memory.
    free(h_y);
    free(h_y_result);
    CUDA_CALL(cudaEventDestroy(start)); // Destroy CUDA events.
    CUDA_CALL(cudaEventDestroy(stop));

    return 0;
}
