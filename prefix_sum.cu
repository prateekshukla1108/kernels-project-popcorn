#include <stdlib.h>
#include <iostream>
#include <assert.h>
using namespace std;

__global__ void prefix_sum(int *nums, int n)
{
    int i = threadIdx.x;
    for (int stride = 1; stride <= n; stride *= 2)
    {
        int temp = 0;
        if ((i - stride) >= 0)
            temp = nums[i - stride];
        __syncthreads();
        nums[i] += temp;
        __syncthreads();
    }
}

int main(void)
{
    int n = 1024;
    int *h_nums, *h_output;
    h_nums = (int *)malloc(n * sizeof(int));
    h_output = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        h_nums[i] = rand() % 1000;
    }
    int *d_nums;
    cudaMalloc((void **)&d_nums, n * sizeof(int));
    cudaMemcpy(d_nums, h_nums, n * sizeof(int), cudaMemcpyHostToDevice);
    prefix_sum<<<1, n>>>(d_nums, n);
    cudaMemcpy(h_output, d_nums, n * sizeof(int), cudaMemcpyDeviceToHost);
    int temp = 0;
    for (int i = 0; i < n; i++)
    {
        temp += h_nums[i];
        assert(temp == h_output[i]);
    }
    return 0;
}