#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <torch/types.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define PI 3.1415

float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

/*
Reduction functions on device. These will be inline:
The compiler will replace the call with the code instead of calling the function (overhead)
*/
/*
Utility warp level sum reduction with shuffle instructions
*/

__device__ __forceinline__ float warpReduceSum(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__device__ __forceinline__ float warpReduceMax(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

/*
This kernel uses flash attention algorithm to compute multi-head attention.
Q, K, and V are 4D tensors of shape (batch_size, n_heads, seq_len, embed_dim).
Additional inputs are Tr and Tc which are the tiles each block computes.
The arrays l and m are to save the norm and maximum for the ith tile.
SRAM will have size M and Br = ceil(M / 4d) and Bc = min(ceil(M / 4d), d)
where M is the size of the SRAM.
*/
template <const int Br, const int Bc>
__global__ void flash_attn_2_kernel(float* Q, float* K, float* V, int N, int d, int Tr, int Tc, float scale, float* L, float* O) {
    int tx = threadIdx.x;  // Br * Bc threads

    int bx = blockIdx.x;  // Batch index
    int by = blockIdx.y;  // Head index

    // tip to calculate offset:
    // count how many elements to skip in the array to reach an index
    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_off = (bx * gridDim.y * N) + (by * N);

    // TODO: remove too much shared memory usage
    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * d;
    float* Sij = Vj + Bc * d;
    float* Oi = Sij + Br * Bc;
    float* li = Oi + Br * d;
    float* mi = li + Br;
    float* mi_new = mi + Br;

    int loads_per_thread_Brd = CEIL_DIV(d, Bc);
    int loads_per_thread_Bcd = CEIL_DIV(d, Br);

    for (int i = 0; i < Tr; i++) {
        // load Qi and Oi from HBM into SMEM
        for (int e = 0; e < loads_per_thread_Brd; e++) {
            int idx = e * (Br * Bc) + tx;
            int row = idx / d;
            if (idx < Br * d && i * Br + row < N) {
                int col = idx % d;
                Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
                Oi[row * d + col] = 0.0f;
            }
        }

        int s_row = tx / Bc;
        int s_col = tx % Bc;

        int global_row = (i * Br) + s_row;

        // init li and mi for each row
        if (s_col == 0) {
            li[s_row] = 0.f;
            mi[s_row] = -INFINITY;
            mi_new[s_row] = -INFINITY;
        }
        __syncthreads();

        for (int j = 0; j < Tc; j++) {
            // load Kj and Vj into SMEM from HBM
            for (int e = 0; e < loads_per_thread_Bcd; e++) {
                int idx = e * (Br * Bc) + tx;
                int row = idx / d;
                if (idx < Bc * d && j * Bc + row < N) {
                    int col = idx % d;
                    Kj[row * d + col] = K[qkv_off + (j * Bc + row) * d + col];
                    Vj[row * d + col] = V[qkv_off + (j * Bc + row) * d + col];
                }
            }
            __syncthreads();

            // compute S = Qi * Kj^T where shape of S: (Br, Bc)
            // TODO: reduce shared memory bank conflicts
            float acc = 0.f;
            for (int k = 0; k < d; k++)
                acc += Qi[s_row * d + k] * Kj[s_col * d + k];

            acc *= scale;
            Sij[s_row * Bc + s_col] = acc;

            // rowmax(S) and rowsum(S) (only one thread per row)
            // computes both in a single pass
            if (s_col == 0) {
                mi[s_row] = mi_new[s_row];
                float row_m = -INFINITY, row_l = 0.f;
                for (int c = 0; c < Bc; c++) {
                    float val = Sij[s_row * Bc + c];
                    if (val > row_m) {
                        row_m = val;
                    }
                }
                float maxval = max(mi[s_row], row_m);

                float kahan_comp = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    float exp_val = expf(Sij[s_row * Bc + c] - maxval);
                    Sij[s_row * Bc + c] = exp_val;

                    float y = exp_val - kahan_comp;  // subtract the compensation from the new value
                    float t = row_l + y;             // add the compensated value to the sum
                    kahan_comp = (t - row_l) - y;    // compute the new compensation.
                    row_l = t;
                }

                mi_new[s_row] = maxval;
                li[s_row] = expf(mi[s_row] - maxval) * li[s_row] + row_l;
            }
            __syncthreads();

            // compute Sij * Vj and do a roll-forward update to O
            // Sij (Br, Bc) and Vj (Bc, d) and we have Br * Bc threads
            // a thread may compute more than one element's dot product
            for (int col = s_col; col < d; col += Bc) {
                float acc = 0.f;
                float kahan_comp = 0.f;
                for (int c = 0; c < Bc; c++) {
                    float term = Sij[s_row * Bc + c] * Vj[c * d + col];

                    float y = term - kahan_comp;
                    float t = acc + y;
                    kahan_comp = (t - acc) - y;
                    acc = t;
                }

                Oi[s_row * d + col] = (mi[s_row] == -INFINITY || mi_new[s_row] == -INFINITY) ? acc : expf(mi[s_row] - mi_new[s_row]) * Oi[s_row * d + col] + acc;
            }
        }

        for (int col = s_col; col < d; col += Bc) {
            if (global_row < N)
                O[qkv_off + global_row * d + col] = (1 / li[s_row]) * Oi[s_row * d + col];
        }

        if (s_col == 0) {
            L[lm_off + (i * Br) + s_row] = mi_new[s_row] + logf(li[s_row]);
        }
        __syncthreads();
    }
}

// Comment the below function to compile and run this file as executable
torch::Tensor fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 16;
    const int Br = 16;

    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    int Tc = ceil((float)N / Bc);
    int Tr = ceil((float)N / Br);
    float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});
    torch::Device device(torch::kCUDA);
    L = L.to(device);

    const int smem_size = ((Br * Bc) + (2 * Br * d) + (2 * Bc * d) + (3 * Br)) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

    dim3 grid_size(B, nh);     // batch_size x num_heads
    dim3 block_size(Br * Bc);  // Br * Bc threads per block

    flash_attn_2_kernel<Br, Bc><<<grid_size, block_size, smem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tr, Tc, softmax_scale,
        L.data_ptr<float>(), O.data_ptr<float>());
    return O;
}

int main() {
    int batch_size = 16;
    int n_head = 8;
    int seq_len = 512;
    int head_embd = 64;

    int qkv_size = batch_size * n_head * seq_len * head_embd;
    int lm_size = batch_size * n_head * seq_len;

    float *Qh, *Kh, *Vh, *Oh, *Lh;
    Qh = (float*)malloc(qkv_size * sizeof(float));
    Kh = (float*)malloc(qkv_size * sizeof(float));
    Vh = (float*)malloc(qkv_size * sizeof(float));
    Oh = (float*)malloc(qkv_size * sizeof(float));
    Lh = (float*)malloc(lm_size * sizeof(float));

    for (int i = 0; i < qkv_size; i++) {
        Qh[i] = 1.0;  // random_normal_clamped(-1, 1);
        Kh[i] = 2.0;  // random_normal_clamped(-1, 1);
        Vh[i] = 3.0;  // random_normal_clamped(-1, 1);
        Oh[i] = 0.0f;
    }
    for (int i = 0; i < lm_size; i++) {
        Lh[i] = 0.0f;
    }

    const int Br = 16, Bc = 16;
    int Tc = ceil((float)seq_len / Bc);
    int Tr = ceil((float)seq_len / Br);
    float softmax_scale = 1.0 / sqrt(head_embd);

    const int smem_size = ((Br * Bc) + (2 * Br * head_embd) + (2 * Bc * head_embd) + (3 * Br)) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

    dim3 grid_dim(batch_size, n_head);  // batch_size x num_heads
    dim3 block_dim(Br * Bc);            // Br * Bc threads per block

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    float *Q, *K, *V, *O, *L;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&O, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&L, lm_size * sizeof(float)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Q, Qh, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, Kh, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, Vh, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(O, Oh, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L, Lh, lm_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    flash_attn_2_kernel<Br, Bc><<<grid_dim, block_dim, smem_size>>>(
        Q, K, V, seq_len, head_embd, Tr, Tc, softmax_scale, L, O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Flash-Attention 1 kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Oh, O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nFirst and Last value in Output:\n");
    printf("%f and %f\n", Oh[0], Oh[qkv_size - 1]);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
    cudaFree(L);
    free(Qh);
    free(Kh);
    free(Vh);
    free(Oh);
    free(Lh);

    return 0;
}