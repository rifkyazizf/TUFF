import Foundation

/// Prefills `promptIds` from a reset runner and returns the next-token
/// log-probability of each entry in `labelIDs`, normalised over the full
/// vocabulary. Nothing is generated, so the model cannot emit anything but a
/// distribution — this is the readout behind typed decision endpoints.
///
/// The logits must come from the real head: a fused greedy head only leaves
/// an argmax behind, so prefill is always asked for `.logits`.
public func prefillLabelLogProbs(producer: any LogitProducer,
                                 promptIds: [Int32],
                                 labelIDs: [Int32],
                                 scratch: RawCompletionScratch,
                                 prefillConfig: PrefillRuntimeConfig = .defaultChunked)
    async throws -> [Double] {
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let vocab = scratch.logits.length / MemoryLayout<Float16>.size
    for id in labelIDs where id < 0 || Int(id) >= vocab {
        throw GeneratorError.invalidGenerationConfig(
            "label token id \(id) is outside the vocabulary of \(vocab)")
    }
    if let context = producer as? any ContextWindowReporting,
       promptIds.count > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count, maxNew: 0,
                                             maxContext: context.maxContext)
    }

    producer.reset()
    if prefillConfig.mode == .chunked,
       let chunked = producer as? any ChunkedPrefillRunner,
       chunked.supportsChunkedPrefill {
        let result = try await chunked.prefillChunked(tokens: promptIds[...],
                                                      startPosition: 0,
                                                      outputMode: .logits,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { _ in }
        guard result.seed == .logitsWritten else {
            throw PrefillError.unsupportedPrefillSeed(
                "label readout requested logits but producer returned \(result.seed)")
        }
    } else {
        // Token-at-a-time fallback; the last `produce` leaves the final row's
        // logits in the buffer, which is exactly the readout position.
        for (position, token) in promptIds.enumerated() {
            try Task.checkCancellation()
            try await producer.produce(token: token, position: position,
                                       into: scratch.logits)
        }
    }

    return try readLabelLogProbs(scratch: scratch, labelIDs: labelIDs)
}

/// One question's tokens after the shared prefix, and the labels to read.
public struct LabelReadout: Sendable {
    public let suffixIds: [Int32]
    public let labelIDs: [Int32]

    public init(suffixIds: [Int32], labelIDs: [Int32]) {
        self.suffixIds = suffixIds
        self.labelIDs = labelIDs
    }
}

/// Prefills `prefixIds` once, then answers each readout by prefilling only its
/// suffix from a checkpoint of the prefix. Results match
/// `prefillLabelLogProbs(prefix + suffix)` because every suffix starts from
/// the same restored KV cursor and recurrent state.
public func prefillSharedPrefixLabelLogProbs(runner: ModelForwardRunner,
                                             prefixIds: [Int32],
                                             readouts: [LabelReadout],
                                             scratch: RawCompletionScratch,
                                             prefillConfig: PrefillRuntimeConfig = .defaultChunked)
    async throws -> [[Double]] {
    guard !prefixIds.isEmpty, readouts.allSatisfy({ !$0.suffixIds.isEmpty }) else {
        throw GeneratorError.emptyPrompt
    }
    guard prefillConfig.mode == .chunked, runner.supportsChunkedPrefill else {
        throw PrefillError.chunkedUnsupported(PrefillError.chunkedRequiresChunkedRunnerReason)
    }
    let vocab = scratch.logits.length / MemoryLayout<Float16>.size
    for readout in readouts {
        for id in readout.labelIDs where id < 0 || Int(id) >= vocab {
            throw GeneratorError.invalidGenerationConfig(
                "label token id \(id) is outside the vocabulary of \(vocab)")
        }
        let total = prefixIds.count + readout.suffixIds.count
        if total > runner.maxContext {
            throw GeneratorError.contextOverflow(prompt: total, maxNew: 0,
                                                 maxContext: runner.maxContext)
        }
    }

    runner.reset()
    let prefix = try await runner.prefillChunked(tokens: prefixIds[...],
                                                 startPosition: 0,
                                                 outputMode: .logits,
                                                 config: prefillConfig,
                                                 into: scratch.logits) { _ in }
    guard prefix.newPosition == prefixIds.count else {
        throw PrefillError.prefillCursorMismatch(
            "prefix prefill ended at \(prefix.newPosition), expected \(prefixIds.count)")
    }
    let checkpoint = try runner.checkpoint()

    var results: [[Double]] = []
    results.reserveCapacity(readouts.count)
    for readout in readouts {
        try Task.checkCancellation()
        try runner.restore(checkpoint)
        let result = try await runner.prefillChunked(tokens: readout.suffixIds[...],
                                                     startPosition: prefixIds.count,
                                                     outputMode: .logits,
                                                     config: prefillConfig,
                                                     into: scratch.logits) { _ in }
        guard result.seed == .logitsWritten else {
            throw PrefillError.unsupportedPrefillSeed(
                "label readout requested logits but producer returned \(result.seed)")
        }
        results.append(try readLabelLogProbs(scratch: scratch, labelIDs: readout.labelIDs))
    }
    return results
}

private func readLabelLogProbs(scratch: RawCompletionScratch,
                               labelIDs: [Int32]) throws -> [Double] {
    let vocab = scratch.logits.length / MemoryLayout<Float16>.size
    let logits = scratch.logits.contents().bindMemory(to: Float16.self, capacity: vocab)
    return try labelLogProbs(logits: UnsafeBufferPointer(start: logits, count: vocab),
                             labelIDs: labelIDs)
}

/// Full-vocabulary log-softmax gathered at `labelIDs`. Accumulates in Double:
/// FP16 probabilities would flush small label masses to zero.
func labelLogProbs(logits: UnsafeBufferPointer<Float16>,
                   labelIDs: [Int32]) throws -> [Double] {
    var maxLogit = -Double.infinity
    for value in logits {
        let v = Double(value)
        guard !v.isNaN else {
            throw PrefillError.unsupportedPrefillSeed("logits contain NaN")
        }
        if v > maxLogit { maxLogit = v }
    }
    guard maxLogit.isFinite else {
        throw PrefillError.unsupportedPrefillSeed("logits have no finite maximum")
    }
    var sum = 0.0
    for value in logits {
        sum += exp(Double(value) - maxLogit)
    }
    let logNormaliser = maxLogit + log(sum)
    return labelIDs.map { Double(logits[Int($0)]) - logNormaliser }
}
