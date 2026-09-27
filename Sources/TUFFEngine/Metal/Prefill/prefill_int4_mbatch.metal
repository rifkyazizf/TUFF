#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_int4_mbatch — batched small-M INT4 affine projection.
//
//   Y[M, N] = X[M, K] · Wᵀ,  W affine-quantised (group 64), M ≤ 64.
//
// Small-M prefill projections are weight-bandwidth bound. The decode GEMV
// re-reads every weight once per token row, and the fixed 64-row-tile kernels
// pay a whole tile for a handful of rows. Here one threadgroup owns a tile of
// `NT` output columns and *all* M token rows, so a weight element is read from
// device memory once per tile, dequantised once into threadgroup memory, and
// multiplied against every row by the SIMD-group matrix units.
//
// Layout (identical to dequant_int4.metal / prefill_dequant_int4_qmm_f16_block):
//   W      uint8  [N, K/2]   low nibble = even k, high nibble = odd k
//   scales bfloat [N, K/64]
//   biases bfloat [N, K/64]
//   w = q·scale + bias
// ============================================================================

// One affine group per K step.
constant constexpr uint kMBKGroup = 64u;

// RB is the row-block count: ceil(M/8) when the kernel is instantiated, so the
// accumulator count is a compile-time constant and the MMA loop unrolls into
// independent chains instead of a predicated runtime loop.
template <uint NT, uint SG, uint RB>
static inline void prefill_int4_mbatch_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    X,
    device half*          Y,
    uint                  M,
    uint                  N,
    uint                  K,
    uint                  xRowStride,
    uint                  yRowStride,
    uint3                 tgid,
    uint                  sgid,
    uint                  lane,
    threadgroup half*     xTile,
    threadgroup half*     wTile,
    threadgroup float*    outTile)
{
    static_assert(NT == SG * 8u, "one SIMD group per 8-column block");
    constexpr uint threads   = SG * 32u;
    // Rows of a threadgroup tile are padded by 8 halves: that keeps them
    // 16-byte aligned (so simdgroup_load can use vector loads rather than a
    // scalar fallback) and walks distinct banks. x and w live on different
    // axes — x as [row][64], w as [n][64] — so they need their own strides.
    constexpr uint xStride   = 64u + 8u;
    constexpr uint wStride   = 64u + 8u;
    constexpr uint colsPerSG = NT / SG;

    const uint mPad     = (M + 7u) / 8u * 8u;
    const uint groups   = K / kMBKGroup;
    const uint rowBytes = K / 2u;
    const uint tileN    = tgid.x * NT;

    // wTile is stored as [n][64]: the dequant writes eight consecutive halves
    // per (row, 8-nibble octet) chunk, which is conflict-free, and the MMA can
    // then load the transposed weight fragment straight out of it.
    simdgroup_float8x8 acc[RB];
    #pragma unroll
    for (uint rb = 0u; rb < RB; ++rb) {
        acc[rb] = simdgroup_float8x8(0.0f);
    }

    const uint column0 = sgid * colsPerSG;

    for (uint g = 0u; g < groups; ++g) {
        const uint kBase     = g * kMBKGroup;
        const uint kBaseByte = kBase >> 1;

        // Staging x as halves keeps the SIMD-group loads in threadgroup memory
        // instead of re-reading device memory once per output column block.
        // Every device load this thread owns is issued before any threadgroup
        // store: one dependent load/store pair per iteration leaves the memory
        // pipeline almost idle, and these few loads are the whole latency bill.
        constexpr uint xElemsPerThread =
            (8u * RB * kMBKGroup + threads - 1u) / threads;
        half xValues[xElemsPerThread];
        #pragma unroll
        for (uint i = 0u; i < xElemsPerThread; ++i) {
            const uint e   = sgid * 32u + lane + i * threads;
            const uint row = e >> 6;
            const uint k   = e & 63u;
            half value = 0.0h;
            if (row < M) {
                value = X[row * xRowStride + kBase + k];
            }
            xValues[i] = value;
        }
        #pragma unroll
        for (uint i = 0u; i < xElemsPerThread; ++i) {
            const uint e = sgid * 32u + lane + i * threads;
            xTile[(e >> 6) * xStride + (e & 63u)] = xValues[i];
        }

        // Dequantise one 8-nibble chunk per iteration: four bytes of packed
        // weights become eight halves, so the whole tile costs NT·K/2 bytes.
        // Loads for both chunks this thread owns go out together, same reason.
        constexpr uint chunksPerThread = NT * 8u / threads;
        uint packedValues[chunksPerThread];
        float scalesValues[chunksPerThread];
        float biasesValues[chunksPerThread];
        #pragma unroll
        for (uint i = 0u; i < chunksPerThread; ++i) {
            const uint chunk = sgid * 32u + lane + i * threads;
            const uint row   = chunk >> 3;
            const uint octet = chunk & 7u;
            const uint n     = tileN + row;
            packedValues[i] = 0u;
            scalesValues[i] = 0.0f;
            biasesValues[i] = 0.0f;
            if (n < N) {
                const uint groupIndex = n * groups + g;
                scalesValues[i] = float(scales[groupIndex]);
                biasesValues[i] = float(biases[groupIndex]);
                const device uint* word = reinterpret_cast<const device uint*>(
                    W + n * rowBytes + kBaseByte + octet * 4u);
                packedValues[i] = word[0];
            }
        }
        #pragma unroll
        for (uint i = 0u; i < chunksPerThread; ++i) {
            const uint chunk = sgid * 32u + lane + i * threads;
            const uint row   = chunk >> 3;
            const uint k0    = (chunk & 7u) * 8u;
            #pragma unroll
            for (uint j = 0u; j < 8u; ++j) {
                wTile[row * wStride + k0 + j] =
                    half(fma(float((packedValues[i] >> (j * 4u)) & 0xFu),
                             scalesValues[i], biasesValues[i]));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load every x fragment this SIMD group needs for the k step up front
        // so the loads overlap instead of serialising behind each MMA.
        simdgroup_half8x8 xFragment[RB];
        #pragma unroll
        for (uint ks = 0u; ks < kMBKGroup / 8u; ++ks) {
            simdgroup_half8x8 weightFragment;
            simdgroup_load(weightFragment, wTile, wStride,
                           ulong2(ulong(ks * 8u), ulong(column0)), true);
            #pragma unroll
            for (uint rb = 0u; rb < RB; ++rb) {
                simdgroup_load(xFragment[rb], xTile, xStride,
                               ulong2(ulong(ks * 8u), ulong(rb * 8u)), false);
            }
            #pragma unroll
            for (uint rb = 0u; rb < RB; ++rb) {
                simdgroup_multiply_accumulate(acc[rb], xFragment[rb],
                                              weightFragment, acc[rb]);
            }
        }

        // Keep another SIMD group from overwriting the tile while this one is
        // still loading fragments from it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // fp32 accumulators land in a per-SIMD-group scratch, then go to Y as half
    // with the padded rows and columns masked off.
    threadgroup float* scratch = &outTile[sgid * 64u];
    #pragma unroll
    for (uint rb = 0u; rb < RB; ++rb) {
        simdgroup_store(acc[rb], scratch, 8u, ulong2(0ul, 0ul), false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint row0 = rb * 8u;
        #pragma unroll
        for (uint j = 0u; j < 2u; ++j) {
            const uint index  = lane + j * 32u;
            const uint row    = index >> 3;
            const uint column = index & 7u;
            const uint m = row0 + row;
            const uint n = tileN + column0 + column;
            if (m < M && n < N) {
                Y[m * yRowStride + n] = half(scratch[index]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Entry points are (tileN, simdgroups, row blocks); tileN must be 8 × simdgroups
// and row blocks is ceil(M/8), so the wrapper picks the one matching its M.
#define PREFILL_INT4_MBATCH_KERNEL(NAME, NT, SG, RB)                         \
kernel void NAME(                                                            \
    device const uint8_t* W      [[buffer(0)]],                              \
    device const bfloat*  scales [[buffer(1)]],                              \
    device const bfloat*  biases [[buffer(2)]],                              \
    device const half*    X      [[buffer(3)]],                              \
    device half*          Y      [[buffer(4)]],                              \
    constant uint&        M      [[buffer(5)]],                              \
    constant uint&        N      [[buffer(6)]],                              \
    constant uint&        K      [[buffer(7)]],                              \
    constant uint&        xRowStride [[buffer(8)]],                          \
    constant uint&        yRowStride [[buffer(9)]],                          \
    uint3                 tgid   [[threadgroup_position_in_grid]],           \
    uint                  sgid   [[simdgroup_index_in_threadgroup]],         \
    uint                  lane   [[thread_index_in_simdgroup]])              \
{                                                                            \
    threadgroup half  xTile[8u * RB * 72u];                                  \
    threadgroup half  wTile[64u * 72u];                                      \
    threadgroup float outTile[SG * 64u];                                     \
    prefill_int4_mbatch_body<NT, SG, RB>(W, scales, biases, X, Y, M, N, K,   \
                                         xRowStride, yRowStride,             \
                                         tgid, sgid, lane,                   \
                                         xTile, wTile, outTile);             \
}

PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r1, 32u, 4u, 1u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r2, 32u, 4u, 2u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r3, 32u, 4u, 3u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r4, 32u, 4u, 4u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r5, 32u, 4u, 5u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r6, 32u, 4u, 6u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r7, 32u, 4u, 7u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t32_s4_r8, 32u, 4u, 8u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r1, 64u, 8u, 1u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r2, 64u, 8u, 2u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r3, 64u, 8u, 3u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r4, 64u, 8u, 4u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r5, 64u, 8u, 5u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r6, 64u, 8u, 6u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r7, 64u, 8u, 7u)
PREFILL_INT4_MBATCH_KERNEL(prefill_int4_mbatch_t64_s8_r8, 64u, 8u, 8u)
