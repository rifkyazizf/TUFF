import Foundation
import Metal

/// Batched small-M INT4 affine projection: `Y[M, N] = X[M, K] · Wᵀ` for
/// M ≤ 64. One threadgroup covers `tileN` output columns and every token row,
/// so each weight is read from device memory once per tile and reused for all
/// rows — unlike `DequantInt4GEMV` (one weight pass per row) and
/// `MPPPrefillInt4QMM` (a fixed 64-row tile).
///
/// The kernel is specialised for affine group 64. One pipeline per
/// (column tile, row-block count) pair: the row-block count `ceil(M/8)` is a
/// compile-time constant so the accumulator chains unroll into registers,
/// which is what the M=1..64 range needs.
final class PrefillInt4MBatchQMM {
    /// Weight column tile per threadgroup for `N >= 64`.
    static let tileN64 = 64
    /// Narrower tile, for outputs (`N`) shorter than the wide tile.
    static let tileN32 = 32
    /// Largest token count a single threadgroup covers.
    static let maxM = 64
    /// Row blocks the kernel is instantiated for.
    static let rowBlocks = 8

    /// `pipelines[tile][rb]` — tile 0 is the 32-column tile, tile 1 the
    /// 64-column one; `rb` is the row-block count minus one.
    private let pipelines: [[MTLComputePipelineState]]

    init(context: MetalContext, groupSize: Int = Quantization.groupSize) throws {
        precondition(groupSize == 64,
                     "PrefillInt4MBatchQMM is specialised for affine group 64, got \(groupSize)")
        var built: [[MTLComputePipelineState]] = []
        for tile in [Self.tileN32, Self.tileN64] {
            let simdGroups = tile / 8
            var forTile: [MTLComputePipelineState] = []
            for rb in 1...Self.rowBlocks {
                forTile.append(try context.pipeline(
                    "prefill_int4_mbatch_t\(tile)_s\(simdGroups)_r\(rb)"))
            }
            built.append(forTile)
        }
        self.pipelines = built
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: Int,
                n: Int,
                k: Int,
                xRowStride: Int? = nil,
                yRowStride: Int? = nil) {
        precondition(k > 0 && k % 64 == 0,
                     "K must be a positive multiple of the affine group 64, got \(k)")
        precondition(m >= 1 && m <= Self.maxM,
                     "M must be in 1...\(Self.maxM), got \(m)")
        precondition(n >= 1, "N must be positive, got \(n)")
        // The kernel reads packed weights four bytes (eight nibbles) per load.
        precondition(weightsOffset % 4 == 0,
                     "PrefillInt4MBatchQMM needs a 4-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        let tileN = n >= Self.tileN64 ? Self.tileN64 : Self.tileN32
        let simdGroups = tileN / 8
        let pipeline = pipelines[tileN == Self.tileN64 ? 1 : 0][(m + 7) / 8 - 1]

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        var xStrideValue = UInt32(xRowStride ?? k)
        var yStrideValue = UInt32(yRowStride ?? n)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&xStrideValue, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&yStrideValue, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + tileN - 1) / tileN, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: simdGroups * 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
