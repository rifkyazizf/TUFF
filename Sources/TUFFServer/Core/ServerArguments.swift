import Foundation
import TUFFEngine

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// registry identifier.
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillExpertStaging: Bool
    public let prefillFetchOverlap: Bool
    public let prefillMBatchInt4: Bool
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let visionPack: String?
    public let visionResidency: VisionResidencyPolicy
    /// Extra system text for every /v1/systemone prompt that carries none.
    public let systemOneSystemPrompt: String?
    /// Whether /v1/systemone prefills the prefix shared by all of a request's
    /// questions once, or re-prefills each question's whole prompt.
    public let systemOnePrefixReuse: Bool
    /// `--systemone-prefix-reuse auto`: reuse only for requests with more than
    /// one question. A single question gains nothing from the checkpoint and
    /// pays for a second prefill call, which reads the routed experts again.
    public let systemOnePrefixReuseMultiOnly: Bool
    /// Temperature applied to /v1/systemone label log-probabilities before
    /// they are renormalised; 1 leaves them as the model produced them.
    public let systemOneTemperature: Double

    public static let usage = """
    usage: TUFFServer --model <completed .gturbo directory> [options]

      --model <dir>              Required model directory.
      --vision-pack <dir>        Vision companion pack (default beside text model).
      --vision-residency <on-demand|keep-ready>
                                 Routed-expert residency during vision (default on-demand).
      --port <1...65535>         Loopback port (default 8080).
      --model-id <id>            API model identifier (default derived from the
                                 installed model: gemma-4-e4b-it,
                                 gemma-4-26b-a4b-it, qwen3.6-35b-a3b,
                                 gpt-oss-20b, gpt-oss-120b, or minimax-m2.7).
      --max-context <tokens>     4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>      Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                                 Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>   Expert-cache slots: 8, 16, 24, or 32 (default 16).
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n> Prefill chunk size: 32, 64, 128, or 256
                                 (default 128). Each chunk re-reads the routed
                                 expert pool, so larger chunks read less.
      --prefill-expert-staging <on|off>
                                 Read each prefill layer's whole routed expert
                                 union in one staged burst (default on). Off
                                 streams it tile by tile through the decode
                                 expert cache.
      --prefill-fetch-overlap <on|off>
                                 With expert staging, overlap the staged reads
                                 with GPU work so tiles run as their experts
                                 land (default on). Off reads the whole union
                                 before running any tile.
      --prefill-mbatch-int4 <on|off>
                                 Use the batched small-M INT4 kernel for wide
                                 chunked-prefill projections (default off: ~8%
                                 faster, but probabilities shift up to ~0.06).
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive
                                 (default off).
      --systemone-system-prompt <text>
                                 System text appended to the /v1/systemone
                                 framing for requests that send none.
      --systemone-prefix-reuse <on|off|auto>
                                 Prefill the prefix every /v1/systemone question
                                 shares once (default on). Off re-prefills each
                                 question's whole prompt; auto reuses only when
                                 a request has more than one question.
      --systemone-temperature <T>
                                 Divide /v1/systemone label log-probabilities by
                                 T before renormalising (default 1). T > 1
                                 softens overconfident answers; it never
                                 changes which label wins.
      --help                     Show this help.
    """

    // Mirrors the CLI's runtime flags so both binaries accept the same options
    // with the same validation, instead of the server pinning production
    // defaults. RuntimeConfiguration traps on unsupported values, so every
    // bound is checked here before the initializer runs.
    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool = true
    ) throws -> RuntimeConfiguration {
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ServerArgumentError.invalid("--expert-cache-slots must be 8, 16, 24, or 32")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ServerArgumentError.invalid(
                "--prefill-chunk-tokens must be one of "
                    + RuntimeConfiguration.allowedPrefillChunkTokens
                        .map(String.init).joined(separator: ", "))
        }
        guard prefillPolicy == .off
                || expertCacheSlots >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        else {
            throw ServerArgumentError.invalid(
                "--expert-cache-slots \(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            prefillExpertStaging: prefillExpertStaging,
            prefillFetchOverlap: prefillFetchOverlap,
            prefillMBatchInt4: prefillMBatchInt4,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var visionPack: String?
        var visionResidency: VisionResidencyPolicy = .onDemand
        var expertCacheSlots = 16
        var expertCachePolicy = RuntimeExpertCachePolicy.lfu
        var prefillPolicy = RuntimePrefillPolicy.chunked
        var prefillChunkTokens = 128
        var prefillExpertStaging = true
        var prefillFetchOverlap = true
        var prefillMBatchInt4 = false
        var rdadvisePolicy = RDAdvicePolicyMode.off
        var systemOneSystemPrompt: String?
        var systemOnePrefixReuse = true
        var systemOnePrefixReuseMultiOnly = false
        var systemOneTemperature = 1.0
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--vision-pack":
                visionPack = value
            case "--vision-residency":
                guard let parsed = VisionResidencyPolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--vision-residency must be on-demand or keep-ready")
                }
                visionResidency = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.invalid("--expert-cache-slots must be 8, 16, 24, or 32")
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid("--expert-cache-policy must be lfu or lru")
                }
                expertCachePolicy = parsed
            case "--prefill":
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ServerArgumentError.invalid("--prefill must be on or off")
                }
            case "--prefill-chunk-tokens":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid("--prefill-chunk-tokens must be 32, 64, or 128")
                }
                prefillChunkTokens = parsed
            case "--prefill-expert-staging":
                switch value {
                case "on": prefillExpertStaging = true
                case "off": prefillExpertStaging = false
                default:
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-staging must be on or off")
                }
            case "--prefill-fetch-overlap":
                switch value {
                case "on": prefillFetchOverlap = true
                case "off": prefillFetchOverlap = false
                default:
                    throw ServerArgumentError.invalid(
                        "--prefill-fetch-overlap must be on or off")
                }
            case "--prefill-mbatch-int4":
                switch value {
                case "on": prefillMBatchInt4 = true
                case "off": prefillMBatchInt4 = false
                default:
                    throw ServerArgumentError.invalid(
                        "--prefill-mbatch-int4 must be on or off")
                }
            case "--systemone-system-prompt":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid(
                        "--systemone-system-prompt must not be empty")
                }
                systemOneSystemPrompt = value
            case "--systemone-prefix-reuse":
                switch value {
                case "on": (systemOnePrefixReuse, systemOnePrefixReuseMultiOnly) = (true, false)
                case "off": (systemOnePrefixReuse, systemOnePrefixReuseMultiOnly) = (false, false)
                case "auto": (systemOnePrefixReuse, systemOnePrefixReuseMultiOnly) = (true, true)
                default:
                    throw ServerArgumentError.invalid(
                        "--systemone-prefix-reuse must be on, off, or auto")
                }
            case "--systemone-temperature":
                guard let parsed = Double(value), parsed.isFinite, parsed > 0, parsed <= 100 else {
                    throw ServerArgumentError.invalid(
                        "--systemone-temperature must be a number in (0, 100]")
                }
                systemOneTemperature = parsed
            case "--rdadvise":
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--rdadvise must be off, default, bounded, or adaptive")
                }
                rdadvisePolicy = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCachePolicy: expertCachePolicy,
                               prefillPolicy: prefillPolicy,
                               prefillChunkTokens: prefillChunkTokens,
                               prefillExpertStaging: prefillExpertStaging,
                               prefillFetchOverlap: prefillFetchOverlap,
                               prefillMBatchInt4: prefillMBatchInt4,
                               rdadvisePolicy: rdadvisePolicy,
                               visionPack: visionPack,
                               visionResidency: visionResidency,
                               systemOneSystemPrompt: systemOneSystemPrompt,
                               systemOnePrefixReuse: systemOnePrefixReuse,
                               systemOnePrefixReuseMultiOnly: systemOnePrefixReuseMultiOnly,
                               systemOneTemperature: systemOneTemperature)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
