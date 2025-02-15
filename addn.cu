#include <stdio.h>

__global__ void addn(float *a, float*b, float *c, int n){
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices < n){
        c[indices] = a[indices] + b[indices];
    }
}

int main(){
    float *host_a,*host_b,*host_c;
    float *device_a,*device_b,*device_c;
    int total_nums;
    printf("Enter the number of value points : \n");
    scanf("%d",&total_nums);
    host_a = (float *)malloc(total_nums*sizeof(float));
    host_b = (float *)malloc(total_nums*sizeof(float));
    host_c = (float *)malloc(total_nums*sizeof(float));
    cudaMalloc(&device_a,sizeof(float)*total_nums);
    cudaMalloc(&device_b,sizeof(float)*total_nums);
    cudaMalloc(&device_c,sizeof(float)*total_nums);
    for(int i = 0;i<total_nums;i++){
        host_a[i] = i;
        host_b[i] = i*2;
    }
    cudaMemcpy(device_a,host_a,total_nums*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(device_b,host_b,total_nums*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(device_c,host_c,total_nums*sizeof(float),cudaMemcpyHostToDevice);
    int thd_pblock = 512;
    int num_blocks = (total_nums + thd_pblock -1 )/thd_pblock;
    addn<<<num_blocks,thd_pblock>>>(device_a,device_b,device_c,total_nums);
    cudaMemcpy(host_c,device_c,sizeof(float)*total_nums,cudaMemcpyDeviceToHost);
    for(int j = 0;j<total_nums;j++){
        if(host_c[j] != host_a[j]+host_b[j]){
            printf("The value does not match at %d such that the c is %f while a and b are %f and %f",j,host_c[j],host_a[j],host_b[j]);
            return 1;        
        }
    }
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    printf("The values are added correctly!!!");
    printf("\n-----------Self patts------------\n")
    return 0;
}
