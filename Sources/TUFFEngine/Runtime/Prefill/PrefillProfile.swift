import Foundation
import Metal

/// Wall-clock and GPU nanoseconds for one `prefillChunked` call, split at the
/// chunked prefill's CPU/GPU sync points (see `TUFF_PREFILL_PROFILE`).
struct PrefillProfile {
    var linearEncode: UInt64 = 0
    var linearAttnWait: UInt64 = 0
    var linearAttnGPU: UInt64 = 0
    var linearLayerTotal: UInt64 = 0
    var fullEncode: UInt64 = 0
    var fullAttnWait: UInt64 = 0
    var fullAttnGPU: UInt64 = 0
    var fullLayerTotal: UInt64 = 0
    var routeCPUAndSharedEncode: UInt64 = 0
    var sharedWait: UInt64 = 0
    var sharedGPU: UInt64 = 0
    var expertFetch: UInt64 = 0
    var tileWait: UInt64 = 0
    var tileGPU: UInt64 = 0
    /// CPU time blocked on GPU work (shared expert + tile drains) while fetch
    /// overlap is on. Small next to `expertFetch` shows the two overlapped.
    var overlapWait: UInt64 = 0
    var tailWait: UInt64 = 0
    var headWait: UInt64 = 0
    var chunkTotal: UInt64 = 0
    var chunks = 0
    var tokens = 0
    var tiles = 0
    var expertsUsed = 0
    var expertsMissed = 0
    /// GPU nanoseconds per Gated-DeltaNet stage, in first-seen order.
    var stages: [(name: String, nanos: UInt64)] = []

    mutating func addStage(_ name: String, _ nanos: UInt64) {
        if let index = stages.firstIndex(where: { $0.name == name }) {
            stages[index].nanos &+= nanos
        } else {
            stages.append((name, nanos))
        }
    }

    mutating func add(_ field: WritableKeyPath<PrefillProfile, UInt64>, _ nanos: UInt64) {
        self[keyPath: field] &+= nanos
    }

    func report() -> String {
        func ms(_ n: UInt64) -> String { String(format: "%8.1f ms", Double(n) / 1e6) }
        func pct(_ n: UInt64) -> String {
            chunkTotal == 0 ? "" : String(format: " %5.1f%%", 100 * Double(n) / Double(chunkTotal))
        }
        let routedWall = linearLayerTotal + fullLayerTotal
            &- (linearEncode + linearAttnWait + fullEncode + fullAttnWait
                + routeCPUAndSharedEncode + sharedWait + tailWait)
        var s = "[prefill profile] \(tokens) tok in \(chunks) chunk(s): \(ms(chunkTotal))\n"
        s += "  GDN layers  encode        \(ms(linearEncode))\(pct(linearEncode))\n"
        s += "  GDN layers  attn+router   \(ms(linearAttnWait))\(pct(linearAttnWait))  gpu \(ms(linearAttnGPU))\n"
        s += "  full layers encode        \(ms(fullEncode))\(pct(fullEncode))\n"
        s += "  full layers attn+router   \(ms(fullAttnWait))\(pct(fullAttnWait))  gpu \(ms(fullAttnGPU))\n"
        s += "  route CPU + shared encode \(ms(routeCPUAndSharedEncode))\(pct(routeCPUAndSharedEncode))\n"
        s += "  shared expert             \(ms(sharedWait))\(pct(sharedWait))  gpu \(ms(sharedGPU))\n"
        s += "  routed experts (wall)     \(ms(routedWall))\(pct(routedWall))\n"
        s += "    expert fetch (SSD/cache)\(ms(expertFetch))\(pct(expertFetch))  "
        s += "\(expertsMissed) missed / \(expertsUsed) used in \(tiles) tiles\n"
        s += "    tile GPU wait           \(ms(tileWait))\(pct(tileWait))  gpu \(ms(tileGPU))\n"
        s += "    overlap wait (shared+tiles)\(ms(overlapWait))\(pct(overlapWait))  "
        s += "cpu blocked on GPU while fetch runs\n"
        s += "  layer tail                \(ms(tailWait))\(pct(tailWait))\n"
        s += "  LM head                   \(ms(headWait))\(pct(headWait))\n"
        for stage in stages {
            s += "  GDN stage \(stage.name.padding(toLength: 24, withPad: " ", startingAt: 0))gpu \(ms(stage.nanos))\n"
        }
        return s
    }
}

/// Every prefill command buffer's GPU interval, gathered from completion
/// handlers under `TUFF_PREFILL_PROFILE`. The per-stage fields above time only
/// the buffers at sync points; this counts all of them, so busy time against
/// the first-start-to-last-end span shows how long the GPU sat idle between
/// buffers (encoding, commit, and CPU-side waits).
final class PrefillGPUClock: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var busy = 0.0
    private var first = Double.infinity
    private var last = 0.0

    func track(_ commandBuffer: MTLCommandBuffer) {
        commandBuffer.addCompletedHandler { [self] cb in
            lock.lock()
            count += 1
            busy += max(0, cb.gpuEndTime - cb.gpuStartTime)
            first = min(first, cb.gpuStartTime)
            last = max(last, cb.gpuEndTime)
            lock.unlock()
        }
    }

    /// One report line, then the clock starts over.
    func drainReport() -> String {
        lock.lock()
        defer {
            count = 0; busy = 0; first = .infinity; last = 0
            lock.unlock()
        }
        guard count > 0 else { return "" }
        let span = last - first
        return String(format: "  command buffers           %d, GPU busy %.1f ms of %.1f ms span "
                      + "(idle between buffers %.1f ms)\n",
                      count, busy * 1e3, span * 1e3, max(0, span - busy) * 1e3)
    }
}

@inline(__always) func now() -> UInt64 {
    clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
}

@inline(__always) func gpuNanos(_ cb: MTLCommandBuffer) -> UInt64 {
    UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
}
