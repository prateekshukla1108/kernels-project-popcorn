#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h" // Include necessary header for macro checks and function declaration

// CUDA Kernel for Trilinear Interpolation Forward Pass
// Each thread computes one output value
template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> feat_interp)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < feats.size(0) && f < feats.size(2)) {
        const scalar_t u = (points[n][0] + 1) / 2;
        const scalar_t v = (points[n][1] + 1) / 2;
        const scalar_t w = (points[n][2] + 1) / 2;

        const scalar_t a = (1 - v) * (1 - w);
        const scalar_t b = v * (1 - w);
        const scalar_t c = v * w;
        const scalar_t d = 1 - a - b - c;

        feat_interp[n][f][0] = (1 - u) * (a * feats[n][0][f] +
                                              b * feats[n][1][f] +
                                              c * feats[n][2][f] +
                                              d * feats[n][3][f]) +
                                      u * (a * feats[n][4][f] +
                                           b * feats[n][5][f] +
                                           c * feats[n][6][f] +
                                           d * feats[n][7][f]);
    }
}

// CUDA Kernel for Trilinear Interpolation Backward Pass
template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> dL_dfeats)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < feats.size(0) && f < feats.size(2)) {
        const scalar_t u = (points[n][0] + 1) / 2;
        const scalar_t v = (points[n][1] + 1) / 2;
        const scalar_t w = (points[n][2] + 1) / 2;

        const scalar_t a = (1 - v) * (1 - w);
        const scalar_t b = v * (1 - w);
        const scalar_t c = v * w;
        const scalar_t d = 1 - a - b - c;

        scalar_t grad = dL_dfeat_interp[n][f];
        dL_dfeats[n][0][f] = grad * (1 - u) * a;
        dL_dfeats[n][1][f] = grad * (1 - u) * b;
        dL_dfeats[n][2][f] = grad * (1 - u) * c;
        dL_dfeats[n][3][f] = grad * (1 - u) * d;
        dL_dfeats[n][4][f] = grad * u * a;
        dL_dfeats[n][5][f] = grad * u * b;
        dL_dfeats[n][6][f] = grad * u * c;
        dL_dfeats[n][7][f] = grad * u * d;
    }
}

// Wrapper function for launching the CUDA forward kernel
torch::Tensor trilinear_fwd_cuda(torch::Tensor feats, torch::Tensor points)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    
    const int N = feats.size(0);
    const int F = feats.size(2);
    
    torch::Tensor feat_interp = torch::zeros({N, F, 1}, feats.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fwd_cuda", ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            feat_interp.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));
    
    return feat_interp;
}

// Wrapper function for launching the CUDA backward kernel
torch::Tensor trilinear_interpolation_bw(torch::Tensor dL_dfeat_interp, torch::Tensor feats, torch::Tensor points)
{
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    
    torch::Tensor dL_dfeats = torch::zeros_like(feats);
    
    const dim3 threads(16, 16);
    const dim3 blocks((feats.size(0) + threads.x - 1) / threads.x, (feats.size(2) + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_interpolation_bw", ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));
    
    return dL_dfeats;
}
