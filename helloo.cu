#include <stdio.h>

__global__ void print_from_gpu(void) { // it will execute of gpu(device code)
    printf("Hello World! from thread [%d,%d] \
    From device\n", threadIdx.x,blockIdx.x);
}
int main(void) {
    printf("Hello World from host!\n");
    print_from_gpu<<<2,1>>>(); // call device functions
    cudaDeviceSynchronize();//halt further executation untill device code finished executing
return 0;
}