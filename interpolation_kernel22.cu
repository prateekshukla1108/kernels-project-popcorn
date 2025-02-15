torch::Tensor trilinear_fwd_cuda(torch::Tensor feats, torch::Tensor points)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    const int N = feats.size(0);
    const int F = feats.size(2);

    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fwd_cuda", ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return feat_interp;
}

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feat_interp)
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

        feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] +
                                      b * feats[n][1][f] +
                                      c * feats[n][2][f] +
                                      d * feats[n][3][f]) +
                              u * (a * feats[n][4][f] +
                                   b * feats[n][5][f] +
                                   c * feats[n][6][f] +
                                   d * feats[n][7][f]);
    }
} 