#include <iostream>
#include <cuda_runtime.h>

// define the  add kernel 
__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i<n)
{
    C[i] = A[i] + B[i];
}
}

// wraper function to call the function correctly
void vecAdd(float* A, float* B, float* C, int n){
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vecAddKernel<<<gridSize, blockSize>>>(A, B, C, n);
}

int main(){

// number of elements in vector
// const int n = 5;
int n;
std::cout << "Enter number of elements for vectors " << std::endl;
std::cin >> n;
// size of vector in bytes
int size = n *sizeof(float);

// init and allocate host input vectors
float *h_A, *h_B, *h_C;
cudaMallocHost(&h_A, size);
cudaMallocHost(&h_B, size);
cudaMallocHost(&h_C, size);

// get vectors from user
std::cout << "Enter " << n << " values for vector A (space-separated): ";
for(int i = 0; i < n; i++) {
     std::cin >> h_A[i];
}

std::cout << "Enter " << n << " values for vector B (space-separated): ";
for(int i = 0; i < n; i++) {
    std::cin >> h_B[i];
}

//init and allocate device memory
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

// copy from host to device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// kernel call
vecAdd(d_A, d_B, d_C, n);

//copy back to host
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

for(int i = 0; i < n; i++) {
  std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
}

// free memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

cudaFreeHost(h_A);
cudaFreeHost(h_B);
cudaFreeHost(h_C);
return 0;
}
