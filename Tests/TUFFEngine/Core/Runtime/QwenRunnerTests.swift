import Testing
import Foundation
import Metal
@testable import TUFFEngine

/// Qwen 3.6 runtime integration: runner construction against the qwen toy
/// fixture (no Gemma sandwich tensors present — init must not touch them),
/// decode and chunked-prefill smoke over the hybrid linear/full layer graph,
/// and the KV-cache + GDN-state reset interplay.
@Suite struct QwenRunnerTests {

    private func makeRunner() throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// (a) Runner init with the qwen arch: GDN kernels, state manager, and
    /// ones router-scale buffers must construct without touching the Gemma
    /// sandwich tensors (the fixture does not contain them — any access
    /// would throw `tensorNotFound`).
    @Test func runnerInit_qwenToy_doesNotTouchSandwichTensors() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 64)
        #expect(runner.usesFusedGreedyHead)
        #expect(!runner.supportsSpeculativeVerification)
    }

    @Test func runnerRejectsCacheSmallerThanRoutedTopK() throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir,
            device: ctx.device,
            expecting: .qwen36Toy(),
            streamingMode: .pread(slotCount: 4))

        #expect(throws: RuntimeConfigurationError.expertCacheTooSmall(
            configured: 4, required: 8)) {
            _ = try RealForwardRunner(
                model: model,
                context: ctx,
                maxContext: 64,
                runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 4))
        }
    }

    /// Decode smoke over the hybrid layer graph: two linear (GDN) layers and
    /// two gated full-attention layers, 8-of-8 routing, gated shared expert,
    /// untied greedy head. Verifies tokens are produced, the cursor advances,
    /// and reset() rewinds both the KV cache and the GDN state.
    @Test func decodeSmoke_hybridLayerGraph() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        #expect(first < 1024)
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        // Same input from the empty state must reproduce the same argmax —
        // this fails if reset() leaves stale GDN state or conv tail behind.
        #expect(runner.lastGreedyToken == first)
    }

    /// Chunked prefill smoke: one chunk through the qwen prefill path
    /// (batched GDN projections + conv tail carry, packed q_proj split,
    /// sub-dim RoPE, no V norm, ones router scales, residual-add tail),
    /// then a decode continuation on top of the prefilled state.
    @Test func prefillChunkedSmoke_thenDecodeContinuation() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        let tokens: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let result = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == 8)
        if case .greedyToken(let token) = result.seed {
            #expect(token < 1024)
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }

        try await runner.produce(token: 9, position: 8, into: logits)
        #expect(runner.lastGreedyToken < 1024)
        #expect(runner.continuationPosition == 9)
    }

    /// Prefill/decode consistency: prefilling [t0] then decoding t1 must give
    /// the same argmax as decoding t0, t1 step by step from a fresh state —
    /// the two paths share state (KV + GDN recurrent state + conv tail).
    @Test func prefillThenDecode_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        // Pure decode reference.
        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        // Prefill the first token, then decode the second.
        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    /// Staging only moves where the routed weights are read from, so a chunked
    /// prefill must produce byte-identical logits with staging on and off —
    /// including when the prompt spans more than one chunk and the off path's
    /// decode cache is warm from the previous chunk.
    @Test func prefillExpertStaging_matchesStreamedTileLogits() async throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()

        func prefillBytes(staging: Bool,
                          tokens: [Int32],
                          chunkTokens: Int) async throws -> [UInt8] {
            let model = try Model.load(directoryURL: dir,
                                       device: ctx.device,
                                       expecting: .qwen36Toy())
            let runner = try RealForwardRunner(
                model: model,
                context: ctx,
                maxContext: 64,
                runtimeConfiguration: RuntimeConfiguration(
                    prefillExpertStaging: staging))
            let logits = try makeLogits(ctx, vocab: 1024)
            let result = try await runner.prefillChunked(
                tokens: tokens[...],
                startPosition: 0,
                outputMode: .logits,
                config: .production(chunkTokens: chunkTokens),
                into: logits,
                onProgress: { _ in })
            #expect(result.newPosition == tokens.count)
            return Array(UnsafeRawBufferPointer(start: logits.contents(),
                                                count: logits.length))
        }

        let singleChunk: [Int32] = Array(1...16)
        let stagedSingle = try await prefillBytes(staging: true, tokens: singleChunk,
                                                  chunkTokens: 32)
        let streamedSingle = try await prefillBytes(staging: false, tokens: singleChunk,
                                                    chunkTokens: 32)
        #expect(stagedSingle == streamedSingle)

        let multiChunk: [Int32] = Array(1...40)
        let stagedMulti = try await prefillBytes(staging: true, tokens: multiChunk,
                                                 chunkTokens: 32)
        let streamedMulti = try await prefillBytes(staging: false, tokens: multiChunk,
                                                   chunkTokens: 32)
        #expect(stagedMulti == streamedMulti)
    }

    /// Fetch overlap only changes when the staged reads run relative to GPU
    /// work, so an overlapped chunked prefill must produce byte-identical
    /// logits to the pre-overlap staging path — single chunk and across a chunk
    /// boundary. Exercises the deferred shared-expert wait and the burst driver.
    @Test func prefillFetchOverlap_matchesSerialStagingLogits() async throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()

        func prefillBytes(overlap: Bool,
                          tokens: [Int32],
                          chunkTokens: Int) async throws -> [UInt8] {
            let model = try Model.load(directoryURL: dir,
                                       device: ctx.device,
                                       expecting: .qwen36Toy())
            let runner = try RealForwardRunner(
                model: model,
                context: ctx,
                maxContext: 64,
                runtimeConfiguration: RuntimeConfiguration(
                    prefillExpertStaging: true,
                    prefillFetchOverlap: overlap))
            let logits = try makeLogits(ctx, vocab: 1024)
            let result = try await runner.prefillChunked(
                tokens: tokens[...],
                startPosition: 0,
                outputMode: .logits,
                config: .production(chunkTokens: chunkTokens),
                into: logits,
                onProgress: { _ in })
            #expect(result.newPosition == tokens.count)
            return Array(UnsafeRawBufferPointer(start: logits.contents(),
                                                count: logits.length))
        }

        let singleChunk: [Int32] = Array(1...16)
        let overlappedSingle = try await prefillBytes(overlap: true, tokens: singleChunk,
                                                      chunkTokens: 32)
        let serialSingle = try await prefillBytes(overlap: false, tokens: singleChunk,
                                                  chunkTokens: 32)
        #expect(overlappedSingle == serialSingle)

        let multiChunk: [Int32] = Array(1...40)
        let overlappedMulti = try await prefillBytes(overlap: true, tokens: multiChunk,
                                                     chunkTokens: 32)
        let serialMulti = try await prefillBytes(overlap: false, tokens: multiChunk,
                                                 chunkTokens: 32)
        #expect(overlappedMulti == serialMulti)
    }

    /// (b) KV manager + GDN state manager interplay under the qwen mask:
    /// linear layers carry no per-token KV storage, full layers are linear
    /// append-only, and reset() returns both to the empty-context state.
    @Test func kvManagerAndGdnState_resetInterplay() throws {
        let cfg = ArchConfig.qwen36Toy()
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: cfg,
                                    maxContext: 32,
                                    fp16RingEnabled: true,
                                    slidingWindow: cfg.slidingWindow,
                                    maxPrefillChunkTokens: 32)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.layerKind(1) == .full)
        #expect(kv.layerKind(2) == .linear)
        #expect(kv.layerKind(3) == .full)
        // Linear layers: no KV rows; full layers: 2 heads * 32 dim * 2 B.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 1) == 32)
        #expect(kv.stride(layer: 1) == 2 * 32 * 2)
        // No SWA layers => the FP16 ring never engages anywhere.
        for L in 0..<cfg.numLayers {
            #expect(kv.ringCapacity(layer: L) == 0)
        }

        let gdnState = try GDNStateManager(device: ctx.device, config: cfg)
        #expect(gdnState.isLinear(layer: 0))
        #expect(!gdnState.isLinear(layer: 1))
        let la = cfg.linearAttention
        #expect(gdnState.stateBytesPerLayer
                == la.numVHeads * la.valueHeadDim * la.keyHeadDim * 4)
        #expect(gdnState.convTailBytesPerLayer
                == (la.convKernelSize - 1) * la.qkvDim * 2)

        // Dirty the recurrent state and the KV cursor, then reset both.
        let state = gdnState.stateBuffer(layer: 0)
        state.contents().assumingMemoryBound(to: Float.self)[0] = 42
        let tail = gdnState.convTailBuffer(layer: 2)
        tail.contents().assumingMemoryBound(to: UInt16.self)[0] = 0x3C00
        kv.advance(by: 4)
        #expect(kv.position == 4)

        kv.reset()
        gdnState.reset()
        #expect(kv.position == 0)
        #expect(state.contents().assumingMemoryBound(to: Float.self)[0] == 0)
        #expect(tail.contents().assumingMemoryBound(to: UInt16.self)[0] == 0)
    }

    /// Prefill scratch layout: the qwen shape must size the packed q_proj /
    /// split-gate / GDN buffers; the Gemma production shape must be
    /// unchanged (all qwen extensions zero-sized).
    @Test func prefillScratchLayout_qwenAndGemmaSizes() {
        let qwen = PrefillChunkScratchLayout(config: .qwen36Toy(), chunkTokens: 32)
        // max(packed q_proj 2*4*32, gdn qkvDim 256) = 256.
        #expect(qwen.qProjElementsPerToken == 256)
        #expect(qwen.attnGateElementsPerToken == 4 * 32)
        #expect(qwen.gdnQKVDim == 256)
        #expect(qwen.gdnValueDim == 128)
        #expect(qwen.gdnVHeads == 4)
        #expect(qwen.sharedScalarGateElements == 1)
        #expect(qwen.qElements == 32 * 256)
        #expect(qwen.attentionOutputElements == 32 * 4 * 32)

        let gemma = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 128)
        #expect(gemma.qProjElementsPerToken == gemma.maxQElementsPerToken)
        #expect(gemma.attnGateElementsPerToken == 0)
        #expect(gemma.gdnQKVDim == 0)
        #expect(gemma.gdnValueDim == 0)
        #expect(gemma.sharedScalarGateElements == 0)
        #expect(gemma.qElements == 128 * gemma.maxQElementsPerToken)
    }
}
