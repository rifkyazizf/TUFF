import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Correctness for `PrefillInt4MBatchQMM`, the small-M batched INT4 projection.
///
/// Always-on: every shape/M combination compares the new kernel against the
/// repeated-`DequantInt4GEMV` reference (the decode path inlined per token row)
/// and requires the normalised error `max|a-b| / max|ref|` to stay within the
/// fp16 noise floor. Weight packing, activation distribution and the affine
/// magnitudes match the microbenchmark so the two agree on what "correct"
/// means.
@Suite struct Int4MBatchProjectionTests {
    private static let shapes: [(n: Int, k: Int)] = [
        (8192, 2048),
        (2048, 4096),
        (32, 2048),
        (100, 128),
    ]
    private static let tokenCounts = [1, 7, 8, 21, 33, 64]
    private static let errorLimit: Float = 1e-2

    @Test func matchesRepeatedGEMV() throws {
        let context = try MetalContext()
        let gemv = try DequantInt4GEMV(context: context)
        let mbatch = try PrefillInt4MBatchQMM(context: context)

        for (shapeIndex, shape) in Self.shapes.enumerated() {
            let (n, k) = shape
            let weights = Self.makeWeights(n: n, k: k, seed: 0x9B00 + UInt64(shapeIndex))
            guard let wBuf = Self.buffer(context.device, weights.packed),
                  let sBuf = Self.buffer(context.device, weights.scales),
                  let bBuf = Self.buffer(context.device, weights.biases) else {
                Issue.record("weight buffer allocation failed for N=\(n) K=\(k)")
                continue
            }

            for m in Self.tokenCounts {
                // Exercise a non-contiguous row layout once per shape: the
                // wrapper must honour both strides rather than assume K and N.
                // The x padding is a multiple of 8 elements because the
                // repeated-GEMV reference reads each activation row as half4
                // vectors and needs the row start 16-byte aligned.
                let strided = m == Self.tokenCounts.last
                let xRowStride = strided ? k + 8 : k
                let yRowStride = strided ? n + 3 : n

                var rng = SeedTree(0x9C00 + UInt64(shapeIndex))
                    .key("int4-mbatch-x-n\(n)-k\(k)-m\(m)")
                var x = [Float16](repeating: 0,
                                  count: (m - 1) * xRowStride + k)
                for row in 0..<m {
                    for column in 0..<k {
                        x[row * xRowStride + column] = Float16(rng.uniform(-1, 1))
                    }
                }

                guard let xBuf = Fp16Buffer.make(context.device, halves: x),
                      let refBuf = Fp16Buffer.make(context.device,
                                                   count: (m - 1) * yRowStride + n),
                      let outBuf = Fp16Buffer.make(context.device,
                                                   count: (m - 1) * yRowStride + n),
                      let commandBuffer = context.queue.makeCommandBuffer() else {
                    Issue.record("buffer allocation failed for N=\(n) K=\(k) M=\(m)")
                    continue
                }

                for row in 0..<m {
                    gemv.encode(commandBuffer: commandBuffer,
                                weights: wBuf,
                                scales: sBuf,
                                biases: bBuf,
                                x: xBuf,
                                xOffset: row * xRowStride * MemoryLayout<Float16>.stride,
                                y: refBuf,
                                yOffset: row * yRowStride * MemoryLayout<Float16>.stride,
                                m: UInt32(n),
                                n: UInt32(k))
                }
                mbatch.encode(commandBuffer: commandBuffer,
                              weights: wBuf,
                              scales: sBuf,
                              biases: bBuf,
                              x: xBuf,
                              y: outBuf,
                              m: m,
                              n: n,
                              k: k,
                              xRowStride: xRowStride,
                              yRowStride: yRowStride)
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error {
                    Issue.record("command buffer error at N=\(n) K=\(k) M=\(m): \(error)")
                    continue
                }

                let reference = Fp16Buffer.read(refBuf, count: (m - 1) * yRowStride + n)
                let actual = Fp16Buffer.read(outBuf, count: (m - 1) * yRowStride + n)
                var maxDiff: Float = 0
                var refNorm: Float = 0
                for row in 0..<m {
                    for column in 0..<n {
                        let index = row * yRowStride + column
                        maxDiff = max(maxDiff, abs(actual[index] - reference[index]))
                        refNorm = max(refNorm, abs(reference[index]))
                    }
                }
                let error = maxDiff / max(refNorm, 1e-6)
                let message = Comment(rawValue:
                    "N=\(n) K=\(k) M=\(m) strided=\(strided) "
                    + "normalised=\(error) maxDiff=\(maxDiff) refNorm=\(refNorm)")
                #expect(error <= Self.errorLimit, message)
            }
        }
    }

    // MARK: - Fixtures

    private struct Weights {
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
    }

    /// Random affine INT4 weights in the resident layout, with the same nibble
    /// and bf16 scale/bias magnitudes the microbenchmark and the existing
    /// projection tests use.
    private static func makeWeights(n: Int, k: Int, seed: UInt64) -> Weights {
        let groups = k / Quantization.groupSize
        var rng = SeedTree(seed).key("int4-mbatch-weights-n\(n)-k\(k)")
        var packed = [UInt8](repeating: 0, count: n * k / 2)
        for index in packed.indices {
            packed[index] = UInt8(truncatingIfNeeded: rng.next())
        }
        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        for row in 0..<n {
            for group in 0..<groups {
                scales[row * groups + group] = Quantization.bf16Bits(rng.uniform(0.0008, 0.0025))
                biases[row * groups + group] = Quantization.bf16Bits(rng.uniform(-0.01, 0.01))
            }
        }
        return Weights(packed: packed, scales: scales, biases: biases)
    }

    private static func buffer(_ device: MTLDevice, _ values: [UInt8]) -> MTLBuffer? {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!, length: values.count,
                              options: .storageModeShared)
        }
    }

    private static func buffer(_ device: MTLDevice, _ values: [UInt16]) -> MTLBuffer? {
        values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                              options: .storageModeShared)
        }
    }
}
