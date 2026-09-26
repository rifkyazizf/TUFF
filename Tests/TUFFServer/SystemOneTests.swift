import Foundation
import NIOCore
import Testing
@testable import TUFFEngine
@testable import TUFFServerCore

private let framingLiteral = "You are a precise classifier. Read the state, "
    + "then answer the question with exactly one of the given labels."

/// Backend that answers every question with a fixed, checkable distribution so
/// the HTTP tests exercise decoding and the response shape rather than math.
private actor ScriptedSystemOneBackend: ServerInferenceBackend {
    private(set) var received: ValidatedSystemOneRequest?
    private(set) var callCount = 0

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw TestSystemOneError.unexpectedCompletion
    }

    func scoreSystemOne(
        _ request: ValidatedSystemOneRequest
    ) async throws -> SystemOneResponse {
        callCount += 1
        received = request
        var entries: [SystemOneAnswerEntry] = []
        var tokens = 0
        for (index, question) in request.questions.enumerated() {
            switch question.kind {
            case .noul:
                entries.append(SystemOneAnswerEntry(key: question.key,
                                                    answer: .noul(0.8)))
            case .choice(let options):
                let share = 0.2 / Double(max(options.count - 1, 1))
                entries.append(SystemOneAnswerEntry(
                    key: question.key,
                    answer: .choice(
                        value: options[0].key,
                        probabilities: options.enumerated().map {
                            SystemOneProbability(key: $0.element.key,
                                                 probability: $0.offset == 0 ? 0.8 : share)
                        },
                        confidence: 0.8)))
            case .score(let levels):
                entries.append(SystemOneAnswerEntry(
                    key: question.key,
                    answer: .score(
                        value: 1.0,
                        legend: levels,
                        probabilities: levels.enumerated().map {
                            SystemOneProbability(key: String($0.offset),
                                                 probability: $0.offset == 1 ? 1.0 : 0.0)
                        },
                        confidence: 1.0)))
            }
            tokens += 10 + index
        }
        return SystemOneResponse(
            model: request.model,
            answers: SystemOneAnswers(entries: entries),
            usage: SystemOneUsage(inputTokens: tokens, outputTokens: 0))
    }
}

private enum TestSystemOneError: Error {
    case unexpectedCompletion
}

/// A backend that only knows how to complete: it inherits the protocol's
/// "systemone is not supported" default.
private actor CompletionOnlyBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(content: "hello",
                         toolCalls: [],
                         finishReason: "stop",
                         usage: OpenAIUsage(promptTokens: 1, completionTokens: 1,
                                            totalTokens: 2))
    }
}

/// Stands in for a tokenizer whose label is not one token.
private actor UnscorableLabelBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw TestSystemOneError.unexpectedCompletion
    }

    func scoreSystemOne(
        _ request: ValidatedSystemOneRequest
    ) async throws -> SystemOneResponse {
        throw SystemOneError.labelIsNotSingleToken(label: "Yes", count: 3)
    }
}

private func expectRejection(_ json: String,
                             param: String?,
                             code: String,
                             modelID: String = "test-model",
                             dialect: ChatDialect = .chatml,
                             sourceLocation: SourceLocation = #_sourceLocation) {
    do {
        _ = try SystemOneRequestValidator.validate(
            Data(json.utf8), modelID: modelID, dialect: dialect)
        Issue.record("expected a rejection with code \(code)",
                     sourceLocation: sourceLocation)
    } catch let error as ServerRequestError {
        guard case .invalid(let message, let actualParam, let actualCode) = error else {
            Issue.record("expected an invalid rejection, got \(error)",
                         sourceLocation: sourceLocation)
            return
        }
        #expect(actualParam == param, sourceLocation: sourceLocation)
        #expect(actualCode == code, sourceLocation: sourceLocation)
        #expect(!message.isEmpty, sourceLocation: sourceLocation)
    } catch {
        Issue.record("expected ServerRequestError, got \(error)",
                     sourceLocation: sourceLocation)
    }
}

private func validate(_ json: String,
                      modelID: String = "test-model",
                      dialect: ChatDialect = .chatml) throws -> ValidatedSystemOneRequest {
    try SystemOneRequestValidator.validate(
        Data(json.utf8), modelID: modelID, dialect: dialect)
}

private func question(_ json: String) throws -> SystemOneQuestion {
    let request = try validate(json)
    return try #require(request.questions.first)
}

private func expectedPrompt(state: String, block: String, system: String? = nil) -> String {
    let suffix = system.map { "\n\n\($0)" } ?? ""
    return "<|im_start|>system\n\(framingLiteral)\(suffix)<|im_end|>\n"
        + "<|im_start|>user\n<state>\n\(state)\n</state>\n\n\(block)"
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

/// The endpoint's wire response, decoded for order-insensitive checks: JSON
/// object key order is the encoder's business, so nothing asserts on it.
private struct DecodedSystemOneAnswer: Decodable {
    let noul: Double?
    let choice: String?
    let score: Double?
    let legend: [String]?
    let probabilities: [String: Double]?
    let confidence: Double?
}

private struct DecodedSystemOneResponse: Decodable {
    struct Usage: Decodable {
        let inputTokens: Int
        let outputTokens: Int
    }

    let model: String
    let answers: [String: DecodedSystemOneAnswer]
    let usage: Usage
}

private func decodeResponse(_ data: Data) throws -> DecodedSystemOneResponse {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(DecodedSystemOneResponse.self, from: data)
}

@Suite("SystemOne validation")
struct SystemOneValidationTests {
    @Test func missingStateAndQuestionsAreRejected() {
        expectRejection(#"{"model":"test-model","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
                        param: "state", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s"}"#,
                        param: "questions", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{}}"#,
                        param: "questions", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":[]}"#,
                        param: "questions", code: "invalid_value")
    }

    @Test func stateMustBeAStringObjectOrArray() {
        expectRejection(#"{"model":"test-model","state":42,"questions":{"q":{"type":"noul","instructions":"x"}}}"#,
                        param: "state", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":null,"questions":{"q":{"type":"noul","instructions":"x"}}}"#,
                        param: "state", code: "invalid_value")
        // The three accepted shapes.
        #expect(throws: Never.self) {
            _ = try validate(#"{"model":"test-model","state":"text","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
            _ = try validate(#"{"model":"test-model","state":{"a":1},"questions":{"q":{"type":"noul","instructions":"x"}}}"#)
            _ = try validate(#"{"model":"test-model","state":[1,2],"questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        }
    }

    @Test func questionTypeAndInstructionsAreRequired() {
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"instructions":"x"}}}"#,
                        param: "questions.q.type", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul"}}}"#,
                        param: "questions.q.instructions", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"lottery","instructions":"x"}}}"#,
                        param: "questions.q.type", code: "unsupported_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":"noul"}}"#,
                        param: "questions.q", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":7}}}"#,
                        param: "questions.q.instructions", code: "invalid_value")
    }

    @Test func choiceOptionCountIsBounded() {
        expectRejection(choiceJSON(optionCount: 1),
                        param: "questions.q.criteria", code: "invalid_value")
        expectRejection(choiceJSON(optionCount: 27),
                        param: "questions.q.criteria", code: "invalid_value")
        #expect(throws: Never.self) { _ = try validate(choiceJSON(optionCount: 2)) }
        #expect(throws: Never.self) { _ = try validate(choiceJSON(optionCount: 26)) }
    }

    @Test func scoreLevelCountIsBounded() {
        expectRejection(scoreJSON(levelCount: 1),
                        param: "questions.q.criteria", code: "invalid_value")
        expectRejection(scoreJSON(levelCount: 11),
                        param: "questions.q.criteria", code: "invalid_value")
        #expect(throws: Never.self) { _ = try validate(scoreJSON(levelCount: 2)) }
        #expect(throws: Never.self) { _ = try validate(scoreJSON(levelCount: 10)) }
    }

    @Test func criteriaShapeMustMatchTheQuestionType() {
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":["a","b"]}}}"#,
                        param: "questions.q.criteria", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"score","instructions":"x","criteria":{"a":"b"}}}}"#,
                        param: "questions.q.criteria", code: "invalid_value")
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":["a"]}}}"#,
                        param: "questions.q.criteria", code: "invalid_value")
        // noul accepts a bare string or an object clarification and no criteria at all.
        #expect(throws: Never.self) {
            _ = try validate(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":"yes means confirmed"}}}"#)
            _ = try validate(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":{"yes":"confirmed"}}}}"#)
        }
    }

    @Test func questionCountIsBounded() {
        expectRejection(questionsJSON(count: 33), param: "questions", code: "invalid_value")
        #expect(throws: Never.self) { _ = try validate(questionsJSON(count: 32)) }
    }

    @Test func repeatedQuestionNamesAreRejected() {
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"},"q":{"type":"noul","instructions":"x"}}}"#,
                        param: "questions", code: "invalid_value")
    }

    @Test func malformedJSONIsRefused() {
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":"#,
                        param: nil, code: "invalid_json")
        expectRejection(#"{"model":"test-model","state":"s","questions":{}"#,
                        param: nil, code: "invalid_json")
        expectRejection(#"{"model":"test-model","state":01,"questions":{}}"#,
                        param: nil, code: "invalid_json")
        expectRejection("", param: nil, code: "invalid_json")
    }

    @Test func nonStringModelIsRejected() {
        expectRejection(#"{"model":7,"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
                        param: "model", code: "invalid_value")
    }

    @Test func escapedStringsAreDecoded() throws {
        let request = try validate(
            #"{"model":"test-model","state":"line\n\t\"quoted\" é😀 tail","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        #expect(request.state == .text("line\n\t\"quoted\" é😀 tail"))
    }

    @Test func wrongModelIsNotFound() throws {
        let error = try #require(throws: ServerRequestError.self) {
            try validate(#"{"model":"other-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        }
        #expect(error == .unknownModel)
        #expect(error.envelope.error.code == "model_not_found")
    }

    @Test func nonChatMLDialectsAreUnsupported() {
        expectRejection(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
                        param: nil, code: "unsupported_value", dialect: .gemma)
    }

    @Test func unknownTopLevelFieldsAreIgnored() throws {
        let request = try validate(
            #"{"model":"test-model","state":"s","temperature":0.7,"stream":true,"nested":{"a":1},"questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        #expect(request.questions.count == 1)
    }

    @Test func choiceCriteriaKeyOrderSurvivesDecoding() throws {
        let request = try validate(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"zeta":"last","alpha":"first","mu":"middle","beta":"second"}}}}"#)
        let question = try #require(request.questions.first)
        guard case .choice(let options) = question.kind else {
            Issue.record("expected a choice question")
            return
        }
        #expect(options.map(\.key) == ["zeta", "alpha", "mu", "beta"])
        #expect(options.map(\.description) == ["last", "first", "middle", "second"])
    }

    @Test func questionsAndAnswersKeepDocumentOrder() throws {
        let request = try validate(
            #"{"model":"test-model","state":"s","questions":{"third":{"type":"noul","instructions":"x"},"first":{"type":"noul","instructions":"x"},"second":{"type":"score","instructions":"x","criteria":["a","b"]}}}"#)
        #expect(request.questions.map(\.key) == ["third", "first", "second"])

        let answers = try SystemOneMath.answer(
            for: try #require(request.questions.last),
            probabilities: [0.25, 0.75])
        #expect(answers == .score(value: 0.75,
                                  legend: ["a", "b"],
                                  probabilities: [
                                    SystemOneProbability(key: "0", probability: 0.25),
                                    SystemOneProbability(key: "1", probability: 0.75),
                                  ],
                                  confidence: 0.75))
    }

    @Test func systemFieldIsOptionalAndMustBeAString() throws {
        let withoutSystem = try validate(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        #expect(withoutSystem.system == nil)
        let withSystem = try validate(
            #"{"model":"test-model","state":"s","system":"be terse","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        #expect(withSystem.system == "be terse")
    }
}

@Suite("SystemOne prompt")
struct SystemOnePromptTests {
    @Test func noulWithoutClarification() throws {
        let question = try question(
            #"{"model":"test-model","state":"The sky is blue.","questions":{"q":{"type":"noul","instructions":"Is the sky blue?"}}}"#)
        let prompt = try SystemOnePrompt.render(
            question: question, state: .text("The sky is blue."), system: nil)
        #expect(prompt == expectedPrompt(
            state: "The sky is blue.",
            block: "Question: Is the sky blue?\nAnswer Yes or No."))
        #expect(prompt.hasSuffix("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    @Test func noulWithClarification() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"Is the user satisfied?","criteria":"Yes means they said so."}}}"#)
        let prompt = try SystemOnePrompt.render(
            question: question, state: .text("s"), system: nil)
        #expect(prompt == expectedPrompt(
            state: "s",
            block: "Question: Is the user satisfied?\n"
                + "Clarification: Yes means they said so.\n"
                + "Answer Yes or No."))
    }

    @Test func noulClarificationObjectIsRenderedAsSortedJSON() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"q","criteria":{"b":"second","a":"first"}}}}"#)
        let prompt = try SystemOnePrompt.render(
            question: question, state: .text("s"), system: nil)
        let text = try #require(question.kind.noulClarification).renderedText()
        #expect(prompt.contains("Clarification: \(text)\n"))
        #expect(text.contains("\n"))
        #expect(text.range(of: "\"a\"")!.lowerBound < text.range(of: "\"b\"")!.lowerBound)
    }

    @Test func choiceOptionsGetLettersInOrder() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"Pick a language.","criteria":{"fr":"French","en":"English","de":"German"}}}}"#)
        let prompt = try SystemOnePrompt.render(
            question: question, state: .text("s"), system: nil)
        #expect(prompt == expectedPrompt(
            state: "s",
            block: "Question: Pick a language.\n"
                + "Options:\n"
                + "A. fr: French\n"
                + "B. en: English\n"
                + "C. de: German\n"
                + "Answer with the option letter only."))
        #expect(SystemOnePrompt.labels(for: question) == ["A", "B", "C"])
    }

    @Test func scoreLevelsGetLettersInOrder() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"score","instructions":"How frustrated?","criteria":["calm","annoyed","angry"]}}}"#)
        let prompt = try SystemOnePrompt.render(
            question: question, state: .text("s"), system: nil)
        #expect(prompt == expectedPrompt(
            state: "s",
            block: "Question: How frustrated?\n"
                + "Levels (lowest to highest):\n"
                + "A. calm\n"
                + "B. annoyed\n"
                + "C. angry\n"
                + "Answer with the level letter only."))
        #expect(SystemOnePrompt.labels(for: question) == ["A", "B", "C"])
    }

    @Test func noulLabelsAreYesAndNo() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        #expect(SystemOnePrompt.labels(for: question) == ["Yes", "No"])
    }

    @Test func lettersRunToZ() {
        #expect(SystemOnePrompt.letter(0) == "A")
        #expect(SystemOnePrompt.letter(9) == "J")
        #expect(SystemOnePrompt.letter(25) == "Z")
    }

    @Test func systemTextComesFromRequestThenFlag() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        let block = "Question: x\nAnswer Yes or No."

        let flagged = try SystemOnePrompt.render(
            question: question, state: .text("s"),
            system: SystemOnePrompt.effectiveSystem(request: nil, server: "Flag text."))
        #expect(flagged == expectedPrompt(state: "s", block: block, system: "Flag text."))

        let fromRequest = try SystemOnePrompt.render(
            question: question, state: .text("s"),
            system: SystemOnePrompt.effectiveSystem(request: "Request text.",
                                                    server: "Flag text."))
        #expect(fromRequest == expectedPrompt(state: "s", block: block, system: "Request text."))
        #expect(!fromRequest.contains("Flag text."))

        let emptyRequest = SystemOnePrompt.effectiveSystem(request: "", server: "Flag text.")
        #expect(emptyRequest == "Flag text.")
        #expect(SystemOnePrompt.effectiveSystem(request: "", server: nil) == nil)
    }

    @Test func objectStateIsRenderedAsPrettySortedJSON() throws {
        let state = try SystemOneRequestValidator.validate(
            Data(#"{"model":"test-model","state":{"b":[1,2],"a":{"z":true}},"questions":{"q":{"type":"noul","instructions":"x"}}}"#.utf8),
            modelID: "test-model",
            dialect: .chatml).state
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        let prompt = try SystemOnePrompt.render(question: question, state: state, system: nil)
        let rendered = try state.renderedText()
        #expect(prompt.contains("<state>\n\(rendered)\n</state>"))
        #expect(rendered.contains("\n"))
        #expect(rendered.range(of: "\"a\"")!.lowerBound < rendered.range(of: "\"b\"")!.lowerBound)
        let roundTripped = try JSONDecoder().decode(JSONValue.self, from: Data(rendered.utf8))
        #expect(roundTripped == (try #require(state.structuredValue)))
    }

    @Test func arrayStateIsRenderedAsPrettyJSON() throws {
        let request = try validate(
            #"{"model":"test-model","state":["first","second"],"questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        let rendered = try request.state.renderedText()
        #expect(rendered.hasPrefix("["))
        #expect(rendered.contains("\n"))
        #expect(rendered.contains("first"))
    }

    /// The shared-prefix path prefills `renderPrefix` and then each question's
    /// `renderSuffix`; together they must be exactly the prompt the endpoint
    /// has always sent, for every question kind and with or without a system
    /// message.
    @Test func prefixPlusSuffixIsTheWholePrompt() throws {
        let questions = [
            try question(#"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"Is the sky blue?","criteria":"Means the sky is clear."}}}"#),
            try question(#"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"Pick a language.","criteria":{"fr":"French","en":"English","de":"German"}}}}"#),
            try question(#"{"model":"test-model","state":"s","questions":{"q":{"type":"score","instructions":"How frustrated?","criteria":["calm","annoyed","angry"]}}}"#),
        ]
        let state = SystemOneValue.text("The sky is blue.")
        for system: String? in [nil, "be terse"] {
            let prefix = try SystemOnePrompt.renderPrefix(state: state, system: system)
            #expect(prefix.hasPrefix("<|im_start|>system\n\(framingLiteral)"))
            #expect(prefix.hasSuffix("<state>\nThe sky is blue.\n</state>\n\n"))
            for question in questions {
                let suffix = try SystemOnePrompt.renderSuffix(question: question)
                let whole = try SystemOnePrompt.render(
                    question: question, state: state, system: system)
                #expect(suffix.hasSuffix(
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                #expect(prefix + suffix == whole)
            }
        }
    }

    /// The prefix is shared: it must not depend on which question follows it.
    @Test func prefixIsIndependentOfTheQuestion() throws {
        let request = try validate(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        let state = request.state
        let first = try SystemOnePrompt.renderPrefix(state: state, system: nil)
        let second = try SystemOnePrompt.renderPrefix(state: state, system: "be terse")
        #expect(first == "<|im_start|>system\n\(framingLiteral)<|im_end|>\n"
            + "<|im_start|>user\n<state>\ns\n</state>\n\n")
        #expect(second == "<|im_start|>system\n\(framingLiteral)\n\nbe terse<|im_end|>\n"
            + "<|im_start|>user\n<state>\ns\n</state>\n\n")
    }
}

@Suite("SystemOne math")
struct SystemOneMathTests {
    @Test func probabilitiesNormalise() throws {
        let probabilities = try SystemOneMath.probabilities(fromLogProbs: [-1.0, -2.0, -0.5])
        #expect(abs(probabilities.reduce(0, +) - 1) < 1e-12)
        #expect(probabilities[2] > probabilities[0])
        #expect(probabilities[0] > probabilities[1])
        #expect(probabilities.allSatisfy { (0...1).contains($0) })
    }

    @Test func equalLogProbsSplitEvenly() throws {
        let probabilities = try SystemOneMath.probabilities(fromLogProbs: [-3.0, -3.0])
        #expect(probabilities == [0.5, 0.5])
    }

    @Test func nonFiniteLogProbsAreRefused() {
        #expect(throws: SystemOneError.nonFiniteLabelLogProbabilities) {
            try SystemOneMath.probabilities(fromLogProbs: [.nan, -1])
        }
        #expect(throws: SystemOneError.nonFiniteLabelLogProbabilities) {
            try SystemOneMath.probabilities(fromLogProbs: [-.infinity, -.infinity])
        }
        #expect(throws: SystemOneError.noLabelLogProbabilities) {
            try SystemOneMath.probabilities(fromLogProbs: [])
        }
    }

    @Test func noulAnswerIsTheYesProbability() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#)
        let answer = try SystemOneMath.answer(for: question, probabilities: [0.9, 0.1])
        #expect(answer == .noul(0.9))
    }

    @Test func choiceAnswerIsTheWinningKey() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"hot":"h","cold":"c"}}}}"#)
        let answer = try SystemOneMath.answer(for: question, probabilities: [0.2, 0.8])
        #expect(answer == .choice(
            value: "cold",
            probabilities: [
                SystemOneProbability(key: "hot", probability: 0.2),
                SystemOneProbability(key: "cold", probability: 0.8),
            ],
            confidence: 0.8))
    }

    @Test func scoreAnswerIsTheExpectation() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"score","instructions":"x","criteria":["a","b","c","d"]}}}"#)
        let answer = try SystemOneMath.answer(for: question, probabilities: [0.1, 0.2, 0.3, 0.4])
        #expect(answer == .score(
            value: 2.0,
            legend: ["a", "b", "c", "d"],
            probabilities: [
                SystemOneProbability(key: "0", probability: 0.1),
                SystemOneProbability(key: "1", probability: 0.2),
                SystemOneProbability(key: "2", probability: 0.3),
                SystemOneProbability(key: "3", probability: 0.4),
            ],
            confidence: 0.4))
    }

    @Test func tiesGoToTheFirstLabel() throws {
        #expect(SystemOneMath.winningIndex([0.5, 0.5, 0.0]) == 0)
        #expect(SystemOneMath.winningIndex([0.2, 0.4, 0.4]) == 1)
        #expect(SystemOneMath.winningIndex([0.0, 0.0]) == 0)

        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":"a","b":"b","c":"c"}}}}"#)
        let answer = try SystemOneMath.answer(for: question, probabilities: [0.4, 0.4, 0.2])
        #expect(answer == .choice(
            value: "a",
            probabilities: [
                SystemOneProbability(key: "a", probability: 0.4),
                SystemOneProbability(key: "b", probability: 0.4),
                SystemOneProbability(key: "c", probability: 0.2),
            ],
            confidence: 0.4))
    }

    @Test func probabilityCountMustMatchTheLabels() throws {
        let question = try question(
            #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":"a","b":"b"}}}}"#)
        #expect(throws: SystemOneError.answerLabelMismatch) {
            try SystemOneMath.answer(for: question, probabilities: [1.0])
        }
    }

    /// JSON object key order is the encoder's business, so the response is
    /// checked by what it decodes to; what the endpoint owes a client is that
    /// each answer sits under the key it asked with, and that the same answers
    /// always serialize the same way.
    @Test func responseEncodesEveryAnswerAndIsStable() throws {
        let response = SystemOneResponse(
            model: "test-model",
            answers: SystemOneAnswers(entries: [
                SystemOneAnswerEntry(key: "second", answer: .noul(0.8)),
                SystemOneAnswerEntry(key: "first", answer: .score(
                    value: 0.5,
                    legend: ["a", "b"],
                    probabilities: [
                        SystemOneProbability(key: "0", probability: 0.5),
                        SystemOneProbability(key: "1", probability: 0.5),
                    ],
                    confidence: 0.5)),
            ]),
            usage: SystemOneUsage(inputTokens: 12, outputTokens: 0))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let encoded = try encoder.encode(response)
        let decoded = try decodeResponse(encoded)
        #expect(decoded.model == "test-model")
        #expect(decoded.usage.inputTokens == 12)
        #expect(decoded.usage.outputTokens == 0)
        #expect(decoded.answers.count == 2)
        #expect(decoded.answers["second"]?.noul == 0.8)
        #expect(decoded.answers["first"]?.score == 0.5)
        #expect(decoded.answers["first"]?.legend == ["a", "b"])
        #expect(decoded.answers["first"]?.probabilities == ["0": 0.5, "1": 0.5])
        #expect(try encoder.encode(response) == encoded)
    }
}

@Suite("SystemOne request body")
struct SystemOneRequestBodyTests {
    /// The chat body parser is shared with systemone, so a body that carries no
    /// `messages` must survive it byte for byte.
    @Test func bodyWithoutMessagesPassesThroughUnchanged() throws {
        let body = Data(#"{"model":"test-model","state":"line\nbreak","system":"be terse","unknown":{"image_url":{"url":"data:image/png;base64,AAAA"}},"questions":{"q1":{"type":"noul","instructions":"is it?","criteria":{"a":"b"}}}}"#.utf8)
        let parser = StreamingChatRequestBody()
        var offset = 0
        for size in [1, 7, 33, 128] {
            let end = min(offset + size, body.count)
            var buffer = ByteBufferAllocator().buffer(capacity: end - offset)
            buffer.writeBytes(body[offset..<end])
            try parser.feed(&buffer)
            offset = end
        }
        var remainder = ByteBufferAllocator().buffer(capacity: body.count - offset)
        remainder.writeBytes(body[offset...])
        try parser.feed(&remainder)

        let parsed = try parser.finish()
        #expect(parsed.json == body)
        #expect(parsed.stagedImages.isEmpty)
        #expect(parsed.lease == nil)
    }

    /// An image_url-shaped object outside `messages` is data, not an
    /// attachment: it must not be staged, and the bytes must not change.
    @Test func nestedImageURLIsNotStaged() throws {
        let body = Data(#"{"state":{"messages":[{"content":[{"image_url":{"url":"data:image/png;base64,\(String(repeating: "A", count: 4_000))"}}]}]}}"#.utf8)
        let parser = StreamingChatRequestBody(visionCapability: "ready")
        var buffer = ByteBufferAllocator().buffer(capacity: body.count)
        buffer.writeBytes(body)
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        #expect(parsed.json == body)
        #expect(parsed.stagedImages.isEmpty)
    }
}

@Suite("SystemOne HTTP", .serialized)
struct SystemOneHTTPTests {
    @Test func validRequestAnswersInRequestOrder() async throws {
        let backend = ScriptedSystemOneBackend()
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: backend,
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let body = #"""
        {"model":"test-model","state":"the user is angry","system":"be terse","questions":{
          "second":{"type":"choice","instructions":"sentiment?","criteria":{"negative":"bad","positive":"good"}},
          "first":{"type":"noul","instructions":"is it about billing?"}}}
        """#
        let (data, response) = try await post(port: port, body: body)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)

        let decoded = try decodeResponse(data)
        #expect(decoded.model == "test-model")
        #expect(decoded.usage.inputTokens == 21)
        #expect(decoded.usage.outputTokens == 0)
        #expect(Set(decoded.answers.keys) == ["second", "first"])
        #expect(decoded.answers["second"]?.choice == "negative")
        #expect(decoded.answers["second"]?.probabilities == ["negative": 0.8, "positive": 0.2])
        #expect(decoded.answers["second"]?.confidence == 0.8)
        #expect(decoded.answers["first"]?.noul == 0.8)

        let received = try #require(await backend.received)
        #expect(received.model == "test-model")
        #expect(received.system == "be terse")
        // The order the client wrote its questions in survives the body parse.
        #expect(received.questions.map(\.key) == ["second", "first"])

        try await server.shutdown()
    }

    @Test func invalidRequestAnswers400WithEnvelope() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: ScriptedSystemOneBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"only":"one"}}}}
        """#)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let envelope = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
        #expect(envelope.error.code == "invalid_value")
        #expect(envelope.error.param == "questions.q.criteria")

        try await server.shutdown()
    }

    @Test func unknownModelAnswers404() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: ScriptedSystemOneBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"nope","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}
        """#)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
            .error.code == "model_not_found")

        try await server.shutdown()
    }

    @Test func nonChatMLServerAnswers400() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: ScriptedSystemOneBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}
        """#)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let envelope = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
        #expect(envelope.error.code == "unsupported_value")
        #expect(envelope.error.message == "systemone requires a ChatML model")

        try await server.shutdown()
    }

    @Test func backendWithoutSystemOneSupportAnswers400() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: CompletionOnlyBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}
        """#)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let envelope = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
        #expect(envelope.error.code == "unsupported_value")
        #expect(envelope.error.message == "systemone is not supported by this backend")

        try await server.shutdown()
    }

    /// A label the tokenizer cannot read at one position is the server's fault,
    /// not the client's: the answer comes back as a 500 rather than a rejection.
    @Test func unscorableLabelAnswers500() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: UnscorableLabelBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"test-model","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}
        """#)
        #expect((response as? HTTPURLResponse)?.statusCode == 500)
        #expect(try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
            .error.code == "internal_error")

        try await server.shutdown()
    }

    @Test func getIsMethodNotAllowed() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: ScriptedSystemOneBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/systemone")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 405)
        #expect(try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
            .error.code == "method_not_allowed")

        try await server.shutdown()
    }

    @Test func wrongContentTypeIsUnsupportedMediaType() async throws {
        let server = TUFFHTTPServer(modelID: "test-model",
                                    queueLimit: 1,
                                    backend: ScriptedSystemOneBackend(),
                                    chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/systemone")!)
        request.httpMethod = "POST"
        request.setValue("text/plain", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"{"model":"test-model"}"#.utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 415)

        try await server.shutdown()
    }

    private func post(port: Int, body: String) async throws -> (Data, URLResponse) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/systemone")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        return try await URLSession.shared.data(for: request)
    }
}

@Suite("SystemOne arguments")
struct SystemOneArgumentTests {
    @Test func systemPromptFlagParses() throws {
        let flagged = try ServerArguments.parse(
            ["--model", "/models/m.gturbo", "--systemone-system-prompt", "be terse"])
        #expect(flagged.systemOneSystemPrompt == "be terse")
        let unflagged = try ServerArguments.parse(["--model", "/models/m.gturbo"])
        #expect(unflagged.systemOneSystemPrompt == nil)
    }

    @Test func emptySystemPromptFlagIsRejected() {
        #expect(throws: ServerArgumentError.invalid(
            "--systemone-system-prompt must not be empty")) {
            _ = try ServerArguments.parse(
                ["--model", "/models/m.gturbo", "--systemone-system-prompt", ""])
        }
    }

    @Test func prefixReuseFlagDefaultsOnAndParses() throws {
        let unflagged = try ServerArguments.parse(["--model", "/models/m.gturbo"])
        #expect(unflagged.systemOnePrefixReuse)
        let on = try ServerArguments.parse(
            ["--model", "/models/m.gturbo", "--systemone-prefix-reuse", "on"])
        #expect(on.systemOnePrefixReuse)
        let off = try ServerArguments.parse(
            ["--model", "/models/m.gturbo", "--systemone-prefix-reuse", "off"])
        #expect(!off.systemOnePrefixReuse)
    }

    @Test func invalidPrefixReuseFlagIsRejected() {
        #expect(throws: ServerArgumentError.invalid(
            "--systemone-prefix-reuse must be on or off")) {
            _ = try ServerArguments.parse(
                ["--model", "/models/m.gturbo", "--systemone-prefix-reuse", "maybe"])
        }
    }

    @Test func labelErrorNamesTheLabel() {
        let error = SystemOneError.labelIsNotSingleToken(label: "Yes", count: 3)
        #expect(error.description.contains("Yes"))
        #expect(error.description.contains("3 tokens"))
    }
}

// MARK: - Builders

private func choiceJSON(optionCount: Int) -> String {
    let criteria = (0..<optionCount)
        .map { #""k\#($0)":"option \#($0)""# }
        .joined(separator: ",")
    return #"{"model":"test-model","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"#
        + criteria + "}}}}"
}

private func scoreJSON(levelCount: Int) -> String {
    let criteria = (0..<levelCount)
        .map { #""level \#($0)""# }
        .joined(separator: ",")
    return #"{"model":"test-model","state":"s","questions":{"q":{"type":"score","instructions":"x","criteria":["#
        + criteria + "]}}}"
}

private func questionsJSON(count: Int) -> String {
    let questions = (0..<count)
        .map { #""q\#($0)":{"type":"noul","instructions":"x"}"# }
        .joined(separator: ",")
    return #"{"model":"test-model","state":"s","questions":{"# + questions + "}}"
}

private extension SystemOneQuestion.Kind {
    var noulClarification: SystemOneValue? {
        if case .noul(let clarification) = self { return clarification }
        return nil
    }
}

private extension SystemOneValue {
    var structuredValue: JSONValue? {
        if case .structure(let value) = self { return value }
        return nil
    }
}
