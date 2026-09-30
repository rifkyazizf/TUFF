import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

/// Microbenchmark for causal full-attention prefill at the qwen36 shape
/// (256-wide heads, 16 query heads, 2 KV heads): the tiled kernel against the
/// opt-in simdgroup kernel, per chunk position a System One prompt produces.
///
/// Opt-in via `TUFF_BENCH_ATTENTION=1` so it never runs in a normal test pass.
/// Timing is GPU time of one dispatch per command buffer, reported, never
/// asserted.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["TUFF_BENCH_ATTENTION"] == "1",
                "Set TUFF_BENCH_ATTENTION=1 to run the prefill attention microbenchmark"))
struct PrefillAttentionBenchmarkTests {
    /// (keys already in the cache, queries in this chunk): short single
    /// prefills, a 256-token chunk of a 466-token prompt, and the last chunk
    /// of a 2,499-token prefix.
    private static let cases: [(start: Int, chunk: Int)] = [
        (0, 74), (0, 192), (210, 256), (0, 256), (1_024, 128), (2_371, 128),
    ]
    private static let headDim = 256
    private static let qHeads = 16
    private static let kvHeads = 2
    private static let warmupRuns = 3
    private static let timedRuns = 20

    @Test func tiledVersusSimdgroup() throws {
        let context = try MetalContext()
        let tiled = try PrefillAttention(context: context, simdgroupCausal: false)
        let simdgroup = try PrefillAttention(context: context, simdgroupCausal: true)
        print("prefill attention microbenchmark, device: \(context.device.name)")
        print("  start chunk   tiled ms  simdgroup ms  speedup")
        for c in Self.cases {
            let keys = c.start + c.chunk
            let qStride = Self.qHeads * Self.headDim
            let kvStride = Self.kvHeads * Self.headDim
            guard let q = Fp16Buffer.make(context.device, values: Self.values(c.chunk * qStride, seed: 1)),
                  let k = Fp16Buffer.make(context.device, values: Self.values(keys * kvStride, seed: 2)),
                  let v = Fp16Buffer.make(context.device, values: Self.values(keys * kvStride, seed: 3)),
                  let out = Fp16Buffer.make(context.device, count: c.chunk * qStride) else {
                Issue.record("alloc failed")
                return
            }
            // The runner's encoding of a full layer: a window spanning every key.
            let params = PrefillAttentionParams(
                startPosition: UInt32(c.start), queryCount: UInt32(c.chunk),
                headDim: UInt32(Self.headDim), numQHeads: UInt32(Self.qHeads),
                numKVHeads: UInt32(Self.kvHeads), kvValidCount: UInt32(keys),
                slidingWindow: UInt32(keys), kvTokenStrideElements: UInt32(kvStride),
                qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(qStride),
                scale: 1.0 / Float(Self.headDim).squareRoot())
            func time(_ kernel: PrefillAttention) -> Double {
                var samples: [Double] = []
                for run in 0..<(Self.warmupRuns + Self.timedRuns) {
                    let cb = context.queue.makeCommandBuffer()!
                    kernel.encodeCausal(commandBuffer: cb, q: q, k: k, v: v, out: out,
                                        params: params, layerKind: .full)
                    cb.commit()
                    cb.waitUntilCompleted()
                    if run >= Self.warmupRuns {
                        samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
                    }
                }
                return samples.sorted()[samples.count / 2]
            }
            let tiledMs = time(tiled)
            let simdMs = time(simdgroup)
            print(String(format: "  %5d %5d   %8.3f   %11.3f   %6.2fx",
                         c.start, c.chunk, tiledMs, simdMs, tiledMs / simdMs))
        }
    }

    private static func values(_ count: Int, seed: UInt64) -> [Float] {
        var rng = SeedTree(seed).key("prefill-attention-bench")
        return (0..<count).map { _ in rng.uniform(-0.35, 0.35) }
    }
}
