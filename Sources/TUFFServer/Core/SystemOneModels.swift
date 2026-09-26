import Foundation
import TUFFEngine

/// Typed-decision endpoint types: request parsing, validation, prompt
/// rendering, and the label-probability arithmetic. Everything here is pure so
/// the wire contract and the prompt text can be tested without a model.

// MARK: - Errors

public enum SystemOneError: Error, Equatable, CustomStringConvertible, Sendable {
    case labelIsNotSingleToken(label: String, count: Int)
    case nonFiniteLabelLogProbabilities
    case noLabelLogProbabilities
    case answerLabelMismatch
    case unencodableValue

    public var description: String {
        switch self {
        case .labelIsNotSingleToken(let label, let count):
            "label \(String(reflecting: label)) encodes to \(count) tokens; "
                + "systemone requires exactly one"
        case .nonFiniteLabelLogProbabilities:
            "label log-probabilities are not finite"
        case .noLabelLogProbabilities:
            "no label log-probabilities were read"
        case .answerLabelMismatch:
            "label probabilities do not match the question's labels"
        case .unencodableValue:
            "state or criteria could not be rendered as JSON"
        }
    }
}

// MARK: - Ordered JSON

struct OrderedJSONMember: Equatable, Sendable {
    let key: String
    let value: OrderedJSON
}

/// A JSON value that remembers the order its object keys were written in.
///
/// This request's meaning depends on that order: a `choice` question's letters
/// follow the order the client wrote its options, and the answer comes back as
/// the option's key. Neither route through Foundation preserves it —
/// `[String: T]` loses the order outright and `JSONDecoder.allKeys` reports hash
/// order on this toolchain — so the body is parsed here.
indirect enum OrderedJSON: Equatable, Sendable {
    case object([OrderedJSONMember])
    case array([OrderedJSON])
    case string(String)
    /// Parsed from the raw token, so an integer stays an integer and a decimal
    /// keeps its digits instead of widening to a Double.
    case number(JSONValue)
    case bool(Bool)
    case null

    var isContainer: Bool {
        switch self {
        case .object, .array: true
        default: false
        }
    }

    func member(_ name: String) -> OrderedJSON? {
        guard case .object(let members) = self else { return nil }
        return members.last { $0.key == name }?.value
    }

    var members: [OrderedJSONMember]? {
        guard case .object(let members) = self else { return nil }
        return members
    }

    var elements: [OrderedJSON]? {
        guard case .array(let elements) = self else { return nil }
        return elements
    }

    var stringValue: String? {
        guard case .string(let text) = self else { return nil }
        return text
    }

    var jsonValue: JSONValue {
        switch self {
        case .object(let members):
            return .object(Dictionary(members.map { ($0.key, $0.value.jsonValue) },
                                      uniquingKeysWith: { first, _ in first }))
        case .array(let elements):
            return .array(elements.map(\.jsonValue))
        case .string(let text):
            return .string(text)
        case .number(let value):
            return value
        case .bool(let value):
            return .bool(value)
        case .null:
            return .null
        }
    }
}

/// Recursive-descent JSON reader over the request bytes.
///
/// Nesting is bounded because the decode runs on the event-loop thread, whose
/// stack is the 512 KiB pthread default; the chat body parser caps depth for the
/// same reason. Only long enough to read one request: there is no streaming
/// interface, because the endpoint answers one body per request.
struct OrderedJSONParser {
    static let maximumDepth = 64

    private let bytes: [UInt8]
    private var index = 0

    private init(_ data: Data) {
        bytes = Array(data)
    }

    static func parse(_ data: Data) throws -> OrderedJSON {
        var parser = OrderedJSONParser(data)
        return try parser.document()
    }

    private static var malformed: ServerRequestError {
        .invalid(message: "malformed JSON request", param: nil, code: "invalid_json")
    }

    private mutating func document() throws -> OrderedJSON {
        let value = try parseValue(depth: 0)
        skipWhitespace()
        guard index == bytes.count else { throw Self.malformed }
        return value
    }

    private mutating func parseValue(depth: Int) throws -> OrderedJSON {
        guard depth <= Self.maximumDepth else { throw Self.malformed }
        skipWhitespace()
        guard let byte = peek else { throw Self.malformed }
        switch byte {
        case UInt8(ascii: "{"):
            return try parseObject(depth: depth)
        case UInt8(ascii: "["):
            return try parseArray(depth: depth)
        case UInt8(ascii: "\""):
            return .string(try parseString())
        case UInt8(ascii: "t"):
            return try parseLiteral("true", .bool(true))
        case UInt8(ascii: "f"):
            return try parseLiteral("false", .bool(false))
        case UInt8(ascii: "n"):
            return try parseLiteral("null", .null)
        default:
            return .number(try parseNumber())
        }
    }

    private mutating func parseObject(depth: Int) throws -> OrderedJSON {
        index += 1
        var members: [OrderedJSONMember] = []
        skipWhitespace()
        if peek == UInt8(ascii: "}") {
            index += 1
            return .object(members)
        }
        while true {
            skipWhitespace()
            guard peek == UInt8(ascii: "\"") else { throw Self.malformed }
            let key = try parseString()
            skipWhitespace()
            guard peek == UInt8(ascii: ":") else { throw Self.malformed }
            index += 1
            let value = try parseValue(depth: depth + 1)
            members.append(OrderedJSONMember(key: key, value: value))
            skipWhitespace()
            switch peek {
            case UInt8(ascii: ","):
                index += 1
            case UInt8(ascii: "}"):
                index += 1
                return .object(members)
            default:
                throw Self.malformed
            }
        }
    }

    private mutating func parseArray(depth: Int) throws -> OrderedJSON {
        index += 1
        var elements: [OrderedJSON] = []
        skipWhitespace()
        if peek == UInt8(ascii: "]") {
            index += 1
            return .array(elements)
        }
        while true {
            elements.append(try parseValue(depth: depth + 1))
            skipWhitespace()
            switch peek {
            case UInt8(ascii: ","):
                index += 1
            case UInt8(ascii: "]"):
                index += 1
                return .array(elements)
            default:
                throw Self.malformed
            }
        }
    }

    private mutating func parseString() throws -> String {
        index += 1
        var output: [UInt8] = []
        while index < bytes.count {
            let byte = bytes[index]
            if byte == UInt8(ascii: "\"") {
                index += 1
                return String(decoding: output, as: UTF8.self)
            }
            if byte != UInt8(ascii: "\\") {
                guard byte >= 0x20 else { throw Self.malformed }
                output.append(byte)
                index += 1
                continue
            }
            index += 1
            guard index < bytes.count else { throw Self.malformed }
            let escape = bytes[index]
            index += 1
            switch escape {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"):
                output.append(escape)
            case UInt8(ascii: "b"): output.append(8)
            case UInt8(ascii: "f"): output.append(12)
            case UInt8(ascii: "n"): output.append(10)
            case UInt8(ascii: "r"): output.append(13)
            case UInt8(ascii: "t"): output.append(9)
            case UInt8(ascii: "u"): try appendUnicodeEscape(to: &output)
            default: throw Self.malformed
            }
        }
        throw Self.malformed
    }

    private mutating func appendUnicodeEscape(to output: inout [UInt8]) throws {
        let high = try parseHexQuad()
        let scalar: Unicode.Scalar
        if (0xD800...0xDBFF).contains(high) {
            // A high surrogate must be followed by its pair; a lone one is not
            // a character, and the state it would render is not the state the
            // client sent.
            guard index + 1 < bytes.count,
                  bytes[index] == UInt8(ascii: "\\"),
                  bytes[index + 1] == UInt8(ascii: "u") else { throw Self.malformed }
            index += 2
            let low = try parseHexQuad()
            guard (0xDC00...0xDFFF).contains(low),
                  let paired = Unicode.Scalar(
                    0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)) else {
                throw Self.malformed
            }
            scalar = paired
        } else {
            guard let single = Unicode.Scalar(high) else { throw Self.malformed }
            scalar = single
        }
        output.append(contentsOf: Array(String(scalar).utf8))
    }

    private mutating func parseHexQuad() throws -> UInt32 {
        guard index + 4 <= bytes.count else { throw Self.malformed }
        var value: UInt32 = 0
        for _ in 0..<4 {
            guard let digit = Self.hexValue(bytes[index]) else { throw Self.malformed }
            value = value << 4 | UInt32(digit)
            index += 1
        }
        return value
    }

    private mutating func parseLiteral(_ text: String, _ value: OrderedJSON) throws -> OrderedJSON {
        let literal = Array(text.utf8)
        guard index + literal.count <= bytes.count,
              Array(bytes[index..<(index + literal.count)]) == literal else {
            throw Self.malformed
        }
        index += literal.count
        return value
    }

    private mutating func parseNumber() throws -> JSONValue {
        let start = index
        while index < bytes.count, Self.isNumberByte(bytes[index]) {
            index += 1
        }
        guard index > start else { throw Self.malformed }
        let token = Data(bytes[start..<index])
        guard let value = try? JSONDecoder().decode(JSONValue.self, from: token),
              !Self.isNonFinite(value) else {
            throw Self.malformed
        }
        return value
    }

    private static func isNonFinite(_ value: JSONValue) -> Bool {
        switch value {
        case .number(let double): !double.isFinite
        case .decimal(let decimal): !NSDecimalNumber(decimal: decimal).doubleValue.isFinite
        default: false
        }
    }

    private static func isNumberByte(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"), UInt8(ascii: "-"), UInt8(ascii: "+"),
             UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E"):
            true
        default:
            false
        }
    }

    private static func hexValue(_ byte: UInt8) -> UInt8? {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): byte - UInt8(ascii: "0")
        case UInt8(ascii: "a")...UInt8(ascii: "f"): byte - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A")...UInt8(ascii: "F"): byte - UInt8(ascii: "A") + 10
        default: nil
        }
    }

    private var peek: UInt8? {
        index < bytes.count ? bytes[index] : nil
    }

    private mutating func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0A, 0x0D:
                index += 1
            default:
                return
            }
        }
    }
}

// MARK: - Values

/// A field that accepts either a string or arbitrary JSON. The structured case
/// keeps the parsed value so the prompt gets re-rendered JSON rather than a
/// Swift description of a dictionary.
public enum SystemOneValue: Equatable, Sendable {
    case text(String)
    case structure(JSONValue)

    /// Objects and arrays render pretty-printed with sorted keys, so the same
    /// logical state always produces the same prompt — and so a cached KV
    /// prefix is not invalidated by dictionary ordering.
    public func renderedText() throws -> String {
        switch self {
        case .text(let text):
            return text
        case .structure(let value):
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(value)
            guard let text = String(data: data, encoding: .utf8) else {
                throw SystemOneError.unencodableValue
            }
            return text
        }
    }

    init(_ json: OrderedJSON) throws {
        switch json {
        case .string(let text):
            self = .text(text)
        case .object, .array:
            self = .structure(json.jsonValue)
        default:
            throw ServerRequestError.invalid(
                message: "value must be a string, object, or array",
                param: nil, code: "invalid_value")
        }
    }
}

/// One `choice` option, or a `noul` clarification entry, in the order the
/// client wrote it.
public struct SystemOneOption: Equatable, Sendable {
    public let key: String
    public let description: String

    public init(key: String, description: String) {
        self.key = key
        self.description = description
    }
}

public struct SystemOneQuestion: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case noul(clarification: SystemOneValue?)
        case choice(options: [SystemOneOption])
        case score(levels: [String])
    }

    public let key: String
    public let instructions: SystemOneValue
    public let kind: Kind

    public init(key: String, instructions: SystemOneValue, kind: Kind) {
        self.key = key
        self.instructions = instructions
        self.kind = kind
    }
}

/// A request that passed validation: every field the backend needs to render,
/// tokenize, and answer, and nothing it would have to re-check.
public struct ValidatedSystemOneRequest: Equatable, Sendable {
    public let model: String
    public let state: SystemOneValue
    public let questions: [SystemOneQuestion]
    /// The request's own system text, empty when it sent none. The effective
    /// text is the request's when non-empty, else the server flag's.
    public let system: String?

    public init(model: String,
                state: SystemOneValue,
                questions: [SystemOneQuestion],
                system: String?) {
        self.model = model
        self.state = state
        self.questions = questions
        self.system = system
    }
}

// MARK: - Validation

public enum SystemOneRequestValidator {
    public static let maximumQuestions = 32
    public static let choiceOptionRange = 2...26
    public static let scoreLevelRange = 2...10

    public static func validate(_ json: Data,
                                modelID: String,
                                dialect: ChatDialect) throws -> ValidatedSystemOneRequest {
        try validate(OrderedJSONParser.parse(json), modelID: modelID, dialect: dialect)
    }

    static func validate(_ root: OrderedJSON,
                         modelID: String,
                         dialect: ChatDialect) throws -> ValidatedSystemOneRequest {
        // Absent and wrong both mean "not this server's model"; the endpoint
        // answers one way for both rather than leaking which ids exist.
        let model = root.member("model")
        guard model != nil else { throw ServerRequestError.unknownModel }
        guard let requestedModel = model?.stringValue else {
            throw invalid("model must be a string", "model", "invalid_value")
        }
        guard requestedModel == modelID else { throw ServerRequestError.unknownModel }
        guard dialect == .chatml else {
            throw invalid("systemone requires a ChatML model", nil, "unsupported_value")
        }
        guard let stateJSON = root.member("state"), stateJSON.isContainer
                || stateJSON.stringValue != nil else {
            throw invalid("state must be a string, object, or array",
                          "state", "invalid_value")
        }
        let state = try validatedValue(stateJSON, param: "state")
        let system: String?
        switch root.member("system") {
        case nil, .some(.null):
            system = nil
        case .some(let value):
            guard let text = value.stringValue else {
                throw invalid("system must be a string", "system", "invalid_value")
            }
            system = text
        }
        guard let questions = root.member("questions")?.members else {
            throw invalid("questions must be an object keyed by question name",
                          "questions", "invalid_value")
        }
        guard !questions.isEmpty, questions.count <= maximumQuestions else {
            throw invalid("questions must contain 1 to \(maximumQuestions) entries",
                          "questions", "invalid_value")
        }
        guard Set(questions.map(\.key)).count == questions.count else {
            throw invalid("questions must not repeat a question name",
                          "questions", "invalid_value")
        }
        var validated: [SystemOneQuestion] = []
        validated.reserveCapacity(questions.count)
        for question in questions {
            validated.append(try validate(question))
        }
        return ValidatedSystemOneRequest(model: requestedModel, state: state,
                                         questions: validated, system: system)
    }

    private static func validate(_ entry: OrderedJSONMember) throws -> SystemOneQuestion {
        let param = "questions.\(entry.key)"
        guard entry.value.members != nil else {
            throw invalid("question must be an object", param, "invalid_value")
        }
        guard let type = entry.value.member("type")?.stringValue else {
            throw invalid("question type is required", "\(param).type", "invalid_value")
        }
        guard let instructionsJSON = entry.value.member("instructions") else {
            throw invalid("instructions are required", "\(param).instructions", "invalid_value")
        }
        let instructions = try validatedValue(instructionsJSON, param: "\(param).instructions")
        let criteria = entry.value.member("criteria")
        let kind: SystemOneQuestion.Kind
        switch type {
        case "noul":
            kind = .noul(clarification: try clarification(criteria, param: param))
        case "choice":
            guard let options = criteria?.members else {
                throw invalid("choice criteria must be an object of option descriptions",
                              "\(param).criteria", "invalid_value")
            }
            guard choiceOptionRange.contains(options.count) else {
                throw invalid(
                    "choice criteria must contain "
                        + "\(choiceOptionRange.lowerBound) to \(choiceOptionRange.upperBound) options",
                    "\(param).criteria", "invalid_value")
            }
            kind = .choice(options: try options.map { option in
                guard let description = option.value.stringValue else {
                    throw invalid("option \(String(reflecting: option.key)) needs a description",
                                  "\(param).criteria", "invalid_value")
                }
                return SystemOneOption(key: option.key, description: description)
            })
        case "score":
            guard let levels = criteria?.elements else {
                throw invalid("score criteria must be an array of level descriptions",
                              "\(param).criteria", "invalid_value")
            }
            guard scoreLevelRange.contains(levels.count) else {
                throw invalid(
                    "score criteria must contain "
                        + "\(scoreLevelRange.lowerBound) to \(scoreLevelRange.upperBound) levels",
                    "\(param).criteria", "invalid_value")
            }
            kind = .score(levels: try levels.map { level in
                guard let description = level.stringValue else {
                    throw invalid("every level needs a description",
                                  "\(param).criteria", "invalid_value")
                }
                return description
            })
        default:
            throw invalid("question type must be noul, choice, or score",
                          "\(param).type", "unsupported_value")
        }
        return SystemOneQuestion(key: entry.key, instructions: instructions, kind: kind)
    }

    private static func clarification(_ criteria: OrderedJSON?,
                                      param: String) throws -> SystemOneValue? {
        guard let criteria, criteria != .null else { return nil }
        guard criteria.stringValue != nil || criteria.members != nil else {
            throw invalid("noul criteria must be a string or an object",
                          "\(param).criteria", "invalid_value")
        }
        return try validatedValue(criteria, param: "\(param).criteria")
    }

    private static func validatedValue(_ json: OrderedJSON,
                                       param: String) throws -> SystemOneValue {
        do {
            return try SystemOneValue(json)
        } catch let error as ServerRequestError {
            guard case .invalid(let message, _, let code) = error else { throw error }
            throw invalid(message, param, code)
        }
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}

// MARK: - Prompt

public enum SystemOnePrompt {
    public static let framing = "You are a precise classifier. Read the state, "
        + "then answer the question with exactly one of the given labels."

    /// Labels read at the assistant's first position, in the order the answer
    /// arithmetic indexes them.
    public static func labels(for question: SystemOneQuestion) -> [String] {
        switch question.kind {
        case .noul:
            ["Yes", "No"]
        case .choice(let options):
            options.indices.map(letter)
        case .score(let levels):
            levels.indices.map(letter)
        }
    }

    /// A → Z; every option and level count is bounded by validation.
    public static func letter(_ index: Int) -> String {
        String(UnicodeScalar(UInt8(65 + index)))
    }

    /// A request's own system text wins; the server's `--systemone-system-prompt`
    /// applies only when the request sent none.
    public static func effectiveSystem(request: String?, server: String?) -> String? {
        if let request, !request.isEmpty { return request }
        return server
    }

    /// The part of the prompt every question of a request shares: the system
    /// turn and the state block. Split out so a request can prefill it once and
    /// only append each question's suffix after it.
    public static func renderPrefix(state: SystemOneValue, system: String?) throws -> String {
        let suffix = system.map { "\n\n\($0)" } ?? ""
        return "<|im_start|>system\n\(framing)\(suffix)<|im_end|>\n"
            + "<|im_start|>user\n<state>\n\(try state.renderedText())\n</state>\n\n"
    }

    /// One question's block followed by the assistant turn that reads its
    /// label. Appended to `renderPrefix` it is the whole prompt.
    public static func renderSuffix(question: SystemOneQuestion) throws -> String {
        try block(for: question)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// The whole ChatML turn for one question. Nothing here enables thinking:
    /// the assistant turn opens and closes an empty think block, so the prompt
    /// is identical whether or not the model can reason.
    ///
    /// Defined as prefix plus suffix so the split rendering cannot drift from
    /// the single-piece prompt the endpoint has always sent.
    public static func render(question: SystemOneQuestion,
                              state: SystemOneValue,
                              system: String?) throws -> String {
        try renderPrefix(state: state, system: system)
            + (try renderSuffix(question: question))
    }

    public static func block(for question: SystemOneQuestion) throws -> String {
        let instructions = try question.instructions.renderedText()
        switch question.kind {
        case .noul(let clarification):
            var text = "Question: \(instructions)\n"
            if let clarification {
                text += "Clarification: \(try clarification.renderedText())\n"
            }
            return text + "Answer Yes or No."
        case .choice(let options):
            var text = "Question: \(instructions)\nOptions:\n"
            for (index, option) in options.enumerated() {
                text += "\(letter(index)). \(option.key): \(option.description)\n"
            }
            return text + "Answer with the option letter only."
        case .score(let levels):
            var text = "Question: \(instructions)\nLevels (lowest to highest):\n"
            for (index, level) in levels.enumerated() {
                text += "\(letter(index)). \(level)\n"
            }
            return text + "Answer with the level letter only."
        }
    }
}

// MARK: - Answer arithmetic

public struct SystemOneProbability: Equatable, Sendable {
    public let key: String
    public let probability: Double

    public init(key: String, probability: Double) {
        self.key = key
        self.probability = probability
    }
}

public enum SystemOneAnswer: Equatable, Sendable {
    case noul(Double)
    case choice(value: String, probabilities: [SystemOneProbability], confidence: Double)
    case score(value: Double, legend: [String], probabilities: [SystemOneProbability], confidence: Double)
}

extension SystemOneAnswer: Encodable {
    private enum Field: String, CodingKey {
        case noul, choice, score, legend, probabilities, confidence
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Field.self)
        switch self {
        case .noul(let probability):
            try container.encode(probability, forKey: .noul)
        case .choice(let value, let probabilities, let confidence):
            try container.encode(value, forKey: .choice)
            try container.encode(Self.map(probabilities), forKey: .probabilities)
            try container.encode(confidence, forKey: .confidence)
        case .score(let value, let legend, let probabilities, let confidence):
            try container.encode(value, forKey: .score)
            try container.encode(legend, forKey: .legend)
            try container.encode(Self.map(probabilities), forKey: .probabilities)
            try container.encode(confidence, forKey: .confidence)
        }
    }

    static func map(_ probabilities: [SystemOneProbability]) -> [String: Double] {
        Dictionary(probabilities.map { ($0.key, $0.probability) },
                   uniquingKeysWith: { first, _ in first })
    }
}

public enum SystemOneMath {
    /// Softmax over the label log-probabilities, in Double. The readout is
    /// already a full-vocabulary log-softmax; renormalising over the labels
    /// alone is what turns it into a distribution over this question's labels.
    public static func probabilities(fromLogProbs logProbs: [Double]) throws -> [Double] {
        guard !logProbs.isEmpty else { throw SystemOneError.noLabelLogProbabilities }
        var maximum = -Double.infinity
        for value in logProbs where value > maximum {
            maximum = value
        }
        guard maximum.isFinite else { throw SystemOneError.nonFiniteLabelLogProbabilities }
        let weights = logProbs.map { exp($0 - maximum) }
        let total = weights.reduce(0, +)
        guard total > 0, total.isFinite else {
            throw SystemOneError.nonFiniteLabelLogProbabilities
        }
        return weights.map { $0 / total }
    }

    public static func answer(for question: SystemOneQuestion,
                              probabilities: [Double]) throws -> SystemOneAnswer {
        let labels = SystemOnePrompt.labels(for: question)
        guard probabilities.count == labels.count else {
            throw SystemOneError.answerLabelMismatch
        }
        switch question.kind {
        case .noul:
            return .noul(probabilities[0])
        case .choice(let options):
            let index = winningIndex(probabilities)
            return .choice(
                value: options[index].key,
                probabilities: zip(options, probabilities).map {
                    SystemOneProbability(key: $0.key, probability: $1)
                },
                confidence: probabilities[index])
        case .score(let levels):
            let index = winningIndex(probabilities)
            let expected = probabilities.enumerated()
                .reduce(0.0) { $0 + Double($1.offset) * $1.element }
            return .score(
                value: expected,
                legend: levels,
                probabilities: probabilities.enumerated().map {
                    SystemOneProbability(key: String($0.offset), probability: $0.element)
                },
                confidence: probabilities[index])
        }
    }

    /// Ties resolve to the first label in order.
    static func winningIndex(_ probabilities: [Double]) -> Int {
        var best = 0
        for offset in probabilities.indices where probabilities[offset] > probabilities[best] {
            best = offset
        }
        return best
    }
}

// MARK: - Response

public struct SystemOneAnswerEntry: Equatable, Sendable {
    public let key: String
    public let answer: SystemOneAnswer

    public init(key: String, answer: SystemOneAnswer) {
        self.key = key
        self.answer = answer
    }
}

/// Answers keyed as the request keyed them.
public struct SystemOneAnswers: Equatable, Sendable {
    public let entries: [SystemOneAnswerEntry]

    public init(entries: [SystemOneAnswerEntry]) {
        self.entries = entries
    }
}

extension SystemOneAnswers: Encodable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(
            Dictionary(entries.map { ($0.key, $0.answer) },
                       uniquingKeysWith: { first, _ in first }))
    }
}

public struct SystemOneUsage: Equatable, Sendable {
    /// Prompt tokens summed over every question: each question is its own
    /// prefill, and none of them is a generation.
    public let inputTokens: Int
    public let outputTokens: Int

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

extension SystemOneUsage: Encodable {
    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

public struct SystemOneResponse: Equatable, Sendable {
    public let model: String
    public let answers: SystemOneAnswers
    public let usage: SystemOneUsage

    public init(model: String, answers: SystemOneAnswers, usage: SystemOneUsage) {
        self.model = model
        self.answers = answers
        self.usage = usage
    }
}

extension SystemOneResponse: Encodable {}
