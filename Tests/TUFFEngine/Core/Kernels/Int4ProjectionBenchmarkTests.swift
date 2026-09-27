import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

/// Microbenchmark for the INT4 projection kernels that chunked prefill can
/// dispatch: repeated `DequantInt4GEMV` (one dispatch per token row, the decode
/// path), the `MPPPrefillInt4QMM` control variant, `PrefillInt4QMM`, and the
/// small-M batched `PrefillInt4MBatchQMM`.
///
/// Opt-in via `TUFF_BENCH_INT4=1` so it never runs in a normal test pass.
/// Timing is reported, never asserted; the test only fails when a kernel
/// errors or a non-reference kernel drifts past `errorLimit` (normalised
/// `max|a-b| / max|ref|`) against the repeated-GEMV reference.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["TUFF_BENCH_INT4"] == "1",
                "Set TUFF_BENCH_INT4=1 to run the INT4 projection microbenchmark"))
struct Int4ProjectionBenchmarkTests {
    private enum Kernel: CaseIterable {
        case repeatedGEMV
        case mppControl
        case prefillQMM
        case mbatchQMM

        var label: String {
            switch self {
            case .repeatedGEMV: return "gemv"
            case .mppControl: return "mpp"
            case .prefillQMM: return "qmm"
            case .mbatchQMM: return "mbat"
            }
        }
    }

    /// One packed-weight buffer laid out the way the resident loader does it:
    /// `uint8 [N, K/2]` weights, then bf16 `[N, K/64]` scales, then bf16
    /// `[N, K/64]` biases, handed to the kernels as offsets into one buffer
    /// (the `TensorView` shape).
    private struct Fixture {
        let buffer: MTLBuffer
        let weightBytes: Int
        let scaleBytes: Int
        let biasBytes: Int
        let scaleOffset: Int
        let biasOffset: Int
        var totalBytes: Int { weightBytes + scaleBytes + biasBytes }
    }

    private struct Result {
        var ms = Double.nan
        var gbPerSecond = Double.nan
        var tflops = Double.nan
        var note = ""
        var mppPath: MPPPrefillInt4QMM.Path?
    }

    private static let shapes: [(n: Int, k: Int)] = [
        (8192, 2048), (4096, 2048), (2048, 4096), (32, 2048), (512, 2048), (2048, 2048),
    ]
    private static let tokenCounts = [1, 8, 16, 21, 32, 48, 64, 128]
    private static let warmupRuns = 3
    private static let timedRuns = 20
    /// `MPPPrefillInt4QMM` reads a whole 64-row A tile regardless of M, so x is
    /// padded to a whole tile to keep those reads inside the allocation. The
    /// kernel bound-checks its stores, so the padding never reaches the output.
    private static let mppTileM = 64
    /// Non-reference kernels must stay within this normalised
    /// `max|a-b| / max|ref|` error of the repeated-GEMV reference.
    private static let errorLimit: Float = 1e-2
    private static let cellHeader = "       ms     GB/s   TFLOPS   nerr"

    @Test func projectionKernels() throws {
        let context = try MetalContext()
        let gemv = try DequantInt4GEMV(context: context)
        let mpp = MPPPrefillInt4QMM(context: context, variant: .control)
        let qmm = try PrefillInt4QMM(context: context)
        let mbatch = try PrefillInt4MBatchQMM(context: context)

        print("INT4 projection microbenchmark")
        print("device: \(context.device.name)")
        print("supportsFamily(.apple10): \(context.device.supportsFamily(.apple10))")
        print("mpp control available: \(mpp.isAvailable)"
            + (mpp.unavailableReason.map { " (\($0))" } ?? ""))
        print("runs: \(Self.warmupRuns) warm-up + \(Self.timedRuns) timed per "
            + "(kernel, shape, M), each in its own command buffer; "
            + "ms = median(gpuEndTime - gpuStartTime)")
        print("GB/s counts packed weights + scales + biases once; "
            + "TFLOPS = 2*M*N*K/time; nerr = max|a-b|/max|ref| vs repeated gemv")

        for (n, k) in Self.shapes {
            guard let fixture = Self.makeFixture(device: context.device, n: n, k: k) else {
                Issue.record("weight buffer allocation failed for N=\(n) K=\(k)")
                continue
            }
            print("")
            print("Shape N=\(n) K=\(k) (N output rows x K input columns, "
                + "group \(Quantization.groupSize))")
            print("   M |" + Kernel.allCases.map { _ in " \(Self.cellHeader) |" }.joined())

            var mppPath: MPPPrefillInt4QMM.Path?
            var mppPathM = 0
            for m in Self.tokenCounts {
                let xRows = max(m, Self.mppTileM)
                var rng = SeedTree(0x3144).key("int4-bench-x-n\(n)-k\(k)-m\(m)")
                var xValues = [Float16](repeating: 0, count: xRows * k)
                let activationCount = m * k
                for index in 0..<activationCount {
                    xValues[index] = Float16(rng.uniform(-1, 1))
                }
                guard let x = Fp16Buffer.make(context.device, halves: xValues),
                      let y = Fp16Buffer.make(context.device, count: m * n) else {
                    Issue.record("activation buffer allocation failed for M=\(m) N=\(n) K=\(k)")
                    continue
                }

                var reference: [Float] = []
                var row = String(format: "%4d |", m)
                for kernel in Kernel.allCases {
                    if kernel == .mbatchQMM && m > PrefillInt4MBatchQMM.maxM {
                        row += Self.unsupportedCell
                        continue
                    }
                    let result = Self.measure(kernel: kernel,
                                              context: context,
                                              fixture: fixture,
                                              n: n, k: k, m: m,
                                              x: x, y: y,
                                              gemv: gemv, mpp: mpp,
                                              qmm: qmm, mbatch: mbatch)
                    if kernel == .repeatedGEMV {
                        reference = Fp16Buffer.read(y, count: m * n)
                        row += Self.cell(ms: result.ms, gbPerSecond: result.gbPerSecond,
                                         tflops: result.tflops, error: 0)
                    } else {
                        let error = Self.normalizedError(
                            Fp16Buffer.read(y, count: m * n), reference)
                        row += Self.cell(ms: result.ms, gbPerSecond: result.gbPerSecond,
                                         tflops: result.tflops, error: error)
                        #expect(result.note.isEmpty,
                                "\(kernel.label) failed at N=\(n) K=\(k) M=\(m): \(result.note)")
                        #expect(error <= Self.errorLimit,
                                "\(kernel.label) normalised error \(error) > \(Self.errorLimit) at N=\(n) K=\(k) M=\(m)")
                    }
                    if kernel == .mppControl {
                        mppPath = result.mppPath
                        mppPathM = m
                    }
                }
                print(row)
            }

            if let mppPath {
                let suffix = mppPathM == Self.tokenCounts[0] ? "" : " (reported at M=\(mppPathM))"
                print("mpp control path: \(mppPath.rawValue)\(suffix)")
            } else {
                print("mpp control path: not measured")
            }
        }
    }

    // MARK: - Measurement

    private static func measure(kernel: Kernel,
                                context: MetalContext,
                                fixture: Fixture,
                                n: Int, k: Int, m: Int,
                                x: MTLBuffer, y: MTLBuffer,
                                gemv: DequantInt4GEMV,
                                mpp: MPPPrefillInt4QMM,
                                qmm: PrefillInt4QMM,
                                mbatch: PrefillInt4MBatchQMM) -> Result {
        var result = Result()
        var times: [Double] = []
        times.reserveCapacity(timedRuns)
        for run in 0..<(warmupRuns + timedRuns) {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                result.note = "makeCommandBuffer returned nil"
                return result
            }
            let path = encode(kernel: kernel,
                              commandBuffer: commandBuffer,
                              fixture: fixture,
                              n: n, k: k, m: m,
                              x: x, y: y,
                              gemv: gemv, mpp: mpp, qmm: qmm, mbatch: mbatch)
            result.mppPath = path
            if path == .fallback {
                result.note = "MPP reported the fallback path (no work encoded)"
                return result
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                result.note = "command buffer error: \(error)"
                return result
            }
            if run >= warmupRuns {
                times.append(commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
            }
        }

        let seconds = median(times)
        result.ms = seconds * 1000
        result.gbPerSecond = Double(fixture.totalBytes) / seconds / 1e9
        result.tflops = 2 * Double(m) * Double(n) * Double(k) / seconds / 1e12
        return result
    }

    /// Encode one whole projection onto a fresh command buffer, mirroring
    /// `RealForwardRunner.encodeInt4Projection`'s three dispatches. Returns the
    /// MPP metadata path when the MPP kernel was the one dispatched.
    private static func encode(kernel: Kernel,
                               commandBuffer: MTLCommandBuffer,
                               fixture: Fixture,
                               n: Int, k: Int, m: Int,
                               x: MTLBuffer, y: MTLBuffer,
                               gemv: DequantInt4GEMV,
                               mpp: MPPPrefillInt4QMM,
                               qmm: PrefillInt4QMM,
                               mbatch: PrefillInt4MBatchQMM) -> MPPPrefillInt4QMM.Path? {
        switch kernel {
        case .repeatedGEMV:
            for row in 0..<m {
                gemv.encode(commandBuffer: commandBuffer,
                            weights: fixture.buffer, weightsOffset: 0,
                            scales: fixture.buffer, scalesOffset: fixture.scaleOffset,
                            biases: fixture.buffer, biasesOffset: fixture.biasOffset,
                            x: x, xOffset: row * k * MemoryLayout<Float16>.stride,
                            y: y, yOffset: row * n * MemoryLayout<Float16>.stride,
                            m: UInt32(n), n: UInt32(k))
            }
            return nil
        case .mppControl:
            let metadata = mpp.encode(commandBuffer: commandBuffer,
                                      weights: fixture.buffer, weightsOffset: 0,
                                      scales: fixture.buffer, scalesOffset: fixture.scaleOffset,
                                      biases: fixture.buffer, biasesOffset: fixture.biasOffset,
                                      x: x, y: y, m: m, n: n, k: k)
            return metadata.path
        case .prefillQMM:
            qmm.encode(commandBuffer: commandBuffer,
                       weights: fixture.buffer, weightsOffset: 0,
                       scales: fixture.buffer, scalesOffset: fixture.scaleOffset,
                       biases: fixture.buffer, biasesOffset: fixture.biasOffset,
                       x: x, y: y, t: m, n: n, k: k)
            return nil
        case .mbatchQMM:
            mbatch.encode(commandBuffer: commandBuffer,
                          weights: fixture.buffer, weightsOffset: 0,
                          scales: fixture.buffer, scalesOffset: fixture.scaleOffset,
                          biases: fixture.buffer, biasesOffset: fixture.biasOffset,
                          x: x, y: y, m: m, n: n, k: k)
            return nil
        }
    }

    // MARK: - Fixtures

    /// Random affine INT4 weights in the resident layout. Nibbles and the
    /// per-group bf16 scale/bias are drawn from the same magnitudes the kernel
    /// correctness tests use, so the outputs land in a realistic range and the
    /// repeated-GEMV reference is comparable.
    private static func makeFixture(device: MTLDevice, n: Int, k: Int) -> Fixture? {
        precondition(k % Quantization.groupSize == 0)
        let groups = k / Quantization.groupSize
        let weightBytes = n * k / 2
        let tableBytes = n * groups * MemoryLayout<UInt16>.stride
        let scaleOffset = align(weightBytes, to: 8)
        let biasOffset = align(scaleOffset + tableBytes, to: 8)
        guard let buffer = device.makeBuffer(length: biasOffset + tableBytes,
                                             options: .storageModeShared) else {
            return nil
        }
        let base = buffer.contents()
        var rng = SeedTree(0x3144).key("int4-bench-weights-n\(n)-k\(k)")

        var packed = [UInt8](repeating: 0, count: weightBytes)
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
        packed.withUnsafeBytes {
            base.copyMemory(from: $0.baseAddress!, byteCount: weightBytes)
        }
        scales.withUnsafeBytes {
            base.advanced(by: scaleOffset).copyMemory(from: $0.baseAddress!, byteCount: tableBytes)
        }
        biases.withUnsafeBytes {
            base.advanced(by: biasOffset).copyMemory(from: $0.baseAddress!, byteCount: tableBytes)
        }
        return Fixture(buffer: buffer,
                       weightBytes: weightBytes,
                       scaleBytes: tableBytes,
                       biasBytes: tableBytes,
                       scaleOffset: scaleOffset,
                       biasOffset: biasOffset)
    }

    // MARK: - Formatting

    private static func cell(ms: Double,
                             gbPerSecond: Double,
                             tflops: Double,
                             error: Float) -> String {
        String(format: " %9.4f %8.1f %8.3f %9.2e |", ms, gbPerSecond, tflops, Double(error))
    }

    /// Placeholder for a kernel that cannot run at this M (the batched kernel
    /// caps at 64 token rows); same column width as a measured cell, NaN-filled.
    private static let unsupportedCell = String(
        format: " %9.4f %8.1f %8.3f %9.2e |",
        Double.nan, Double.nan, Double.nan, Double.nan)

    // MARK: - Math

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    /// `max|a-b| / max|ref|`: a global normalisation, unlike a per-element
    /// relative error that explodes wherever a reference value crosses zero.
    private static func normalizedError(_ actual: [Float], _ reference: [Float]) -> Float {
        precondition(actual.count == reference.count, "length mismatch")
        var maxDiff: Float = 0
        var refNorm: Float = 0
        for index in 0..<actual.count {
            maxDiff = max(maxDiff, abs(actual[index] - reference[index]))
            refNorm = max(refNorm, abs(reference[index]))
        }
        return maxDiff / max(refNorm, 1e-6)
    }

    private static func align(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }
}
