// Custom CUDA histogram kernels for the level-wise GPU grower.
//
// The torch implementation builds each level's (3, K, F, B) histogram with three
// scatter_add_ calls over an explicitly materialized (n*F) int64 index tensor
// (~40 MB at the root) plus expanded value tensors -- a lot of memory traffic and
// several kernel launches for what is fundamentally one pass of atomic adds. This
// kernel does that pass directly: one launch, fused over the grad/hess/count
// channels, no index/value materialization. The atomic-add work is the same as
// torch's, but the overhead around it (index build, three passes, big temporaries)
// is gone -- which is exactly what made the torch path GPU-bandwidth-heavy.
//
// hist_build: dense histogram, one thread per row. Each row adds its (g, h, w)
// into bin binned[row, f] of its node slot, for every feature f.
//
// hist_build_priv: a privatized variant for the root / wide levels. Each block
// keeps a small per-feature-tile histogram for a SINGLE node slot in shared
// memory and flushes it once, cutting global-atomic contention on hot bins. It is
// only valid when every row in the launch maps to the same slot (K==1), i.e. the
// root histogram, the single largest build in a tree.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// One thread per row; loop over features doing fused atomic adds into global mem.
__global__ void hist_kernel(const uint8_t* __restrict__ binned,
                            const int64_t* __restrict__ slot,
                            const float* __restrict__ g,
                            const float* __restrict__ h,
                            const float* __restrict__ w,
                            int64_t n, int F, int B, int64_t KFB,
                            float* __restrict__ out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t s = slot[i];
    float gi = g[i], hi = h[i], wi = w[i];
    const uint8_t* row = binned + i * (int64_t)F;
    float* o0 = out;
    float* o1 = out + KFB;
    float* o2 = out + 2 * KFB;
    int64_t base = s * (int64_t)F * B;
    for (int f = 0; f < F; f++) {
        int64_t idx = base + (int64_t)f * B + row[f];
        atomicAdd(&o0[idx], gi);
        atomicAdd(&o1[idx], hi);
        atomicAdd(&o2[idx], wi);
    }
}

// Root-only privatized histogram (K == 1). Each block owns a contiguous tile of
// features [f0, f0+FT) and accumulates a (3, FT, B) histogram for those features
// across a strided slice of rows in shared memory, then adds it once to global
// memory. This turns per-row global atomics on hot bins into shared-memory
// atomics (far cheaper) plus one coalesced flush.
extern __shared__ float smem[];
__global__ void hist_priv_kernel(const uint8_t* __restrict__ binned,
                                 const float* __restrict__ g,
                                 const float* __restrict__ h,
                                 const float* __restrict__ w,
                                 int64_t n, int F, int B, int FT,
                                 float* __restrict__ out) {
    int f0 = blockIdx.y * FT;
    int ft = min(FT, F - f0);
    int tile = ft * B;              // bins in this block's feature tile
    float* sg = smem;               // (ft, B)
    float* sh = smem + tile;
    float* sc = smem + 2 * tile;
    for (int t = threadIdx.x; t < tile; t += blockDim.x) {
        sg[t] = 0.f; sh[t] = 0.f; sc[t] = 0.f;
    }
    __syncthreads();

    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float gi = g[i], hi = h[i], wi = w[i];
        const uint8_t* row = binned + i * (int64_t)F + f0;
        for (int f = 0; f < ft; f++) {
            int idx = f * B + row[f];
            atomicAdd(&sg[idx], gi);
            atomicAdd(&sh[idx], hi);
            atomicAdd(&sc[idx], wi);
        }
    }
    __syncthreads();

    // Flush the tile to global memory (slot 0). KFB == F*B since K==1.
    int64_t KFB = (int64_t)F * B;
    float* o0 = out;
    float* o1 = out + KFB;
    float* o2 = out + 2 * KFB;
    for (int t = threadIdx.x; t < tile; t += blockDim.x) {
        int f = t / B, b = t % B;
        int64_t idx = (int64_t)(f0 + f) * B + b;
        atomicAdd(&o0[idx], sg[t]);
        atomicAdd(&o1[idx], sh[t]);
        atomicAdd(&o2[idx], sc[t]);
    }
}

torch::Tensor hist_build(torch::Tensor binned, torch::Tensor slot,
                         torch::Tensor g, torch::Tensor h, torch::Tensor w,
                         int64_t K, int64_t F, int64_t B) {
    binned = binned.contiguous();
    auto out = torch::zeros({3, K, F, B}, g.options());
    int64_t n = slot.size(0);
    int64_t KFB = K * F * B;
    const uint8_t* bp = binned.data_ptr<uint8_t>();
    const float* gp = g.data_ptr<float>();
    const float* hp = h.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    float* op = out.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (K == 1) {
        // Privatized path for the root: tile features so the shared histogram
        // fits, spread rows over enough blocks to keep the GPU busy.
        int FT = 8;                                 // features per block tile
        size_t shmem = (size_t)3 * FT * B * sizeof(float);
        int threads = 256;
        int row_blocks = (int)min((int64_t)512, (n + threads - 1) / threads);
        if (row_blocks < 1) row_blocks = 1;
        int feat_blocks = (int)((F + FT - 1) / FT);
        dim3 grid(row_blocks, feat_blocks);
        hist_priv_kernel<<<grid, threads, shmem, stream>>>(bp, gp, hp, wp, n, F, B, FT, op);
    } else {
        int threads = 256;
        int64_t blocks = (n + threads - 1) / threads;
        hist_kernel<<<blocks, threads, 0, stream>>>(bp, slot.data_ptr<int64_t>(),
                                                    gp, hp, wp, n, F, B, KFB, op);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hist_build", &hist_build, "Fused gradient/hessian/count histogram (CUDA)");
}
