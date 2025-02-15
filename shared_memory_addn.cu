#include <iostream>
#define BLOCK_SIZE 256
#define TOTAL_PTS 1024
__global__ void shared_addn( float * a, float * b, float * c,int n){
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];

    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    if(indices < n){
        shared_a[threadIdx.x] = a[indices];
        shared_b[threadIdx.x] = b[indices];
    }
    __syncthreads();

    if(indices<n){
        c[indices] = shared_a[threadIdx.x] + shared_b[threadIdx.x];
    }
}
int main(){
    float*host_a;
    float*host_b;
    float*host_c;
    float*device_a;
    float*device_b;
    float*device_c;
    int memory_size = sizeof(float)*TOTAL_PTS;

    host_a = (float*)malloc(memory_size);
    host_b = (float*)malloc(memory_size);
    host_c = (float*)malloc(memory_size);
    for(int i = 0;i<TOTAL_PTS;i++){
        host_a[i] = i;
        host_b[i] = i*2;
    }
    
    cudaMalloc(&device_a,memory_size);
    cudaMalloc(&device_b,memory_size);
    cudaMalloc(&device_c,memory_size);

    cudaMemcpy(device_a,host_a,memory_size,cudaMemcpyHostToDevice);
    cudaMemcpy(device_b,host_b,memory_size,cudaMemcpyHostToDevice);

    int blocks = (TOTAL_PTS + BLOCK_SIZE -1)/BLOCK_SIZE;
    shared_addn<<<blocks,BLOCK_SIZE>>>(device_a,device_b,device_c,TOTAL_PTS);
    cudaMemcpy(host_c,device_c,memory_size,cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int j=0;j<TOTAL_PTS;j++){
        if(host_c[j] != host_b[j] + host_a[j]){
            correct = false;
        }
    }
    if (correct) {
        std::cout << "Result is correct!\n";
    } else {
        std::cout << "Result is incorrect!\n";
    }

    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return 0;
}
