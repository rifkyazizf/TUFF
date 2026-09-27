public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public enum RuntimeConfigurationError: Error, CustomStringConvertible, Equatable {
    case expertCacheTooSmall(configured: Int, required: Int)

    public var description: String {
        switch self {
        case .expertCacheTooSmall(let configured, let required):
            return "expert cache has \(configured) slots; this model routes "
                + "\(required) experts per token"
        }
    }
}

public struct RuntimeConfiguration: Sendable, Equatable {
    /// Larger caches let high-memory Macs retain more routed experts instead
    /// of rereading them from SSD. The runtime clamps the requested count to
    /// the model's actual experts-per-layer, so a 32-expert model never pays
    /// for (for example) a 128-slot cache.
    public static let allowedExpertCacheSlots = [4, 8, 16, 24, 32, 48, 64, 96, 128]
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedChunkTokens
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    /// Stage a chunked prefill chunk's whole per-layer routed-expert union into
    /// a dedicated arena in one burst instead of streaming it tile by tile
    /// through the decode expert cache.
    public let prefillExpertStaging: Bool
    /// With `prefillExpertStaging`, overlap the staged read with the GPU: the
    /// shared expert runs while the read starts, and each sub-burst's tiles
    /// commit as soon as their experts land while the next sub-burst is read.
    /// Off reproduces the pre-overlap staging path exactly.
    public let prefillFetchOverlap: Bool
    /// Dispatch eligible chunked-prefill INT4 projections to
    /// `PrefillInt4MBatchQMM`, which reads every weight tile once and reuses it
    /// for all token rows. Off keeps the per-family MPP/qmm/GEMV choice.
    public let prefillMBatchInt4: Bool
    /// Output width (`rows`) below which the batched kernel's single
    /// threadgroup cannot amortize its weight read, so the wide-N kernels stay
    /// in charge. `nil` uses `PrefillProjectionDispatchPolicy.mbatchMinimumRows`;
    /// tests lower it to reach toy projections.
    public let prefillMBatchMinimumRows: Int?
    public let headPath: RuntimeHeadPath

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                prefillExpertStaging: Bool = true,
                prefillFetchOverlap: Bool = true,
                prefillMBatchInt4: Bool = false,
                prefillMBatchMinimumRows: Int? = nil,
                forceLogitsHead: Bool = false) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.prefillExpertStaging = prefillExpertStaging
        self.prefillFetchOverlap = prefillFetchOverlap
        self.prefillMBatchInt4 = prefillMBatchInt4
        self.prefillMBatchMinimumRows = prefillMBatchMinimumRows
        self.headPath = forceLogitsHead ? .logits : .fusedRows
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
