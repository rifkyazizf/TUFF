import Foundation
import Metal
import TUFFFormat

/// Preallocated per-runner staging arena for chunked MoE prefill.
///
/// The decode expert cache holds 8...128 slots, sized for top-k decode. A
/// prefill chunk routes far more: a 60-token qwen36 prompt touches ~117
/// distinct experts in a single layer, so tile-by-tile streaming through that
/// cache never hits and every tile blocks on its own eight-expert `pread`
/// burst. The arena keeps one slot per expert id, so a whole layer's routed
/// union fits without remapping, and is reused for every layer because layers
/// run strictly in order and a layer's tiles are drained before the next
/// layer's burst.
final class PrefillExpertStagingArena: @unchecked Sendable {
    /// Slots per layer, one per expert id.
    let capacity: Int
    /// Page-rounded expert stride; slots are laid out on this boundary.
    let slotSize: Int
    private let buffer: MTLBuffer

    init(device: MTLDevice, capacity: Int, slotSize: Int) throws {
        precondition(capacity > 0, "staging arena needs at least one slot")
        precondition(slotSize > 0, "staging arena slot size must be positive")
        guard let buffer = device.makeBuffer(length: capacity * slotSize,
                                             options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = "prefill.expertStagingArena"
        self.capacity = capacity
        self.slotSize = slotSize
        self.buffer = buffer
    }

    func destinationPointer(forExpert expert: Int) -> UnsafeMutableRawPointer {
        buffer.contents().advanced(by: expert * slotSize)
    }

    /// A view of the staged expert's whole region, shaped exactly like the
    /// decode streamer's expert views so the MoE kernels cannot tell which
    /// path produced them.
    func view(layer: Int, expert: Int, length: Int) -> TensorView {
        TensorView(buffer: buffer,
                   offset: UInt64(expert * slotSize),
                   length: UInt64(length),
                   scaleOffset: 0, scaleLength: 0,
                   biasOffset: 0, biasLength: 0,
                   shape: (UInt32(layer), UInt32(expert), 0, 0),
                   dtype: GTurboFormatV1.DType.u32.rawValue)
    }
}
