#!/usr/bin/env python3
"""Smoke test for POST /v1/systemone against a running TUFFServer.

Standard library only. Every case is a real request, and the run exits non-zero
if any case fails, so it can gate a release the same way the Swift tests do.

    python3 Scripts/systemone_smoke.py --port 8089
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

POSITIVE, NEGATIVE, NEUTRAL = "positive", "negative", "neutral"

TICKETS = {
    "billing": "Payments, charges, invoices, refunds, and subscription prices.",
    "technical": "Crashes, errors, sync failures, and anything that will not open.",
    "account": "Sign-in, passwords, profile details, and notification settings.",
    "other": "Anything that is not about payments, software, or the account.",
}

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "id": "Indonesian",
}

ANIMALS = {
    "squirrel": "A small grey mammal with a very bushy tail that climbs trees and buries acorns.",
    "rabbit": "A small mammal with long ears that lives in a burrow and eats grass.",
    "cat": "A whiskered pet that purrs and hunts mice.",
    "horse": "A large hoofed animal that is ridden and eats hay.",
    "whale": "A huge marine mammal that surfaces to breathe through a blowhole.",
    "parrot": "A brightly coloured bird that mimics speech.",
    "snake": "A long legless reptile that slithers and hisses.",
    "cow": "A large farm animal that is milked and chews cud.",
    "fox": "A reddish wild canid with a pointed snout and a bushy tail.",
    "elephant": "A huge grey animal with a trunk and tusks.",
}

NOUL_CASES = [
    ("noul_true_fact",
     "The Pacific Ocean is larger than any other ocean on Earth.",
     "Is this statement true?", True),
    ("noul_false_fact",
     "The Earth is flat and the Moon orbits it once a day.",
     "Is this statement true?", False),
    ("noul_refund_yes",
     "Hello, I was charged twice for the same subscription this month and I would "
     "like my money back for the duplicate charge.",
     "Does the customer want a refund?", True),
    ("noul_refund_no",
     "Hi, I moved house last week and need the delivery address on my account "
     "changed before the next shipment goes out.",
     "Does the customer want a refund?", False),
    ("noul_negation_yes",
     "I am not satisfied at all with this product; it stopped working after a day.",
     "Does the text say the user is NOT satisfied?", True),
    ("noul_negation_no",
     "I am very satisfied with this product; it works exactly as described.",
     "Does the text say the user is NOT satisfied?", False),
    ("noul_weather_delay",
     "Our flight was cancelled because of the storm, can we move to tomorrow?",
     "Was the delay caused by weather?", True),
    ("noul_password_help",
     "I forgot my password and the reset link in the email has expired.",
     "Does the user need help with their password?", True),
    ("noul_shipping_question",
     "Where is my order? It was supposed to arrive on Tuesday.",
     "Is the user asking where their order is?", True),
]

CHOICE_CASES = [
    ("choice_sentiment_positive",
     "The app is fast and the new editor is a joy to use.",
     "What is the sentiment of the text?",
     POSITIVE, {POSITIVE: "Praise or approval.", NEGATIVE: "Complaint or disapproval.",
                NEUTRAL: "A statement of fact with no evaluation."}),
    ("choice_sentiment_negative",
     "The update broke sync and I lost an hour of work.",
     "What is the sentiment of the text?",
     NEGATIVE, {POSITIVE: "Praise or approval.", NEGATIVE: "Complaint or disapproval.",
                NEUTRAL: "A statement of fact with no evaluation."}),
    ("choice_sentiment_neutral",
     "The package includes a charging cable and a quick start guide.",
     "What is the sentiment of the text?",
     NEUTRAL, {POSITIVE: "Praise or approval.", NEGATIVE: "Complaint or disapproval.",
                NEUTRAL: "A statement of fact with no evaluation."}),
    ("choice_route_billing",
     "I was charged twice for the same subscription.",
     "Which team should handle this message?", "billing", TICKETS),
    ("choice_route_technical",
     "The app crashes every time I open the camera.",
     "Which team should handle this message?", "technical", TICKETS),
    ("choice_route_account",
     "I need to change the email address on my profile.",
     "Which team should handle this message?", "account", TICKETS),
    ("choice_route_other",
     "Do you have a store in Berlin?",
     "Which team should handle this message?", "other", TICKETS),
    ("choice_language_en",
     "Could you send the invoice to my new address?",
     "Which language is this message written in?", "en", LANGUAGES),
    ("choice_language_fr",
     "Merci de me confirmer la date de livraison.",
     "Which language is this message written in?", "fr", LANGUAGES),
    ("choice_language_de",
     "Bitte senden Sie mir die Rechnung erneut.",
     "Which language is this message written in?", "de", LANGUAGES),
    ("choice_language_es",
     "Necesito cambiar la direccion de entrega del pedido.",
     "Which language is this message written in?", "es", LANGUAGES),
    ("choice_language_id",
     "Saya ingin mengubah alamat pengiriman pesanan saya.",
     "Which language is this message written in?", "id", LANGUAGES),
    # Ten options: the widest choice the endpoint accepts, and the deepest one
    # the option-order check shuffles.
    ("choice_animal_ten",
     "A small grey mammal with a very bushy tail that climbs trees and buries acorns.",
     "Which animal does the text describe?", "squirrel", ANIMALS),
]

SCORE_CASES = [
    ("score_frustration_angry",
     "This is the third time I have had to explain the same problem and nobody "
     "has fixed it yet. I am losing patience.",
     "How frustrated is the writer?", ["calm", "annoyed", "angry"], "angry"),
    ("score_frustration_calm",
     "I just wanted to check the status of my order, no rush at all.",
     "How frustrated is the writer?", ["calm", "annoyed", "angry"], "calm"),
    ("score_review_excellent",
     "Delivered a day early and the quality is far better than what I paid for.",
     "How positive is this review?",
     ["terrible", "poor", "average", "good", "excellent"], "excellent"),
    ("score_review_terrible",
     "The item arrived scratched and the box was crushed.",
     "How positive is this review?",
     ["terrible", "poor", "average", "good", "excellent"], "terrible"),
    ("score_review_average",
     "It works as described. Nothing special either way.",
     "How positive is this review?",
     ["terrible", "poor", "average", "good", "excellent"], "average"),
]

MULTI_STATE = (
    "I was charged twice for the subscription and now the app will not let me log "
    "in. This is the second time I have contacted support and I am losing patience."
)

MULTI_QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": TICKETS},
    "frustrated": {"type": "noul",
                   "instructions": "Is the writer frustrated?", "criteria": None},
    "sentiment": {"type": "choice", "instructions": "What is the sentiment of the text?",
                  "criteria": {POSITIVE: "Praise or approval.",
                               NEGATIVE: "Complaint or disapproval.",
                               NEUTRAL: "A statement of fact with no evaluation."}},
    "language": {"type": "choice", "instructions": "Which language is this message written in?",
                 "criteria": LANGUAGES},
    "positivity": {"type": "score", "instructions": "How positive is this message?",
                   "criteria": ["terrible", "poor", "average", "good", "excellent"]},
}

# PIZZA_CASE is the system-field probe: the ticket says nothing about pizza, so
# only an effective system prompt can move the answer toward billing.
PIZZA_STATE = "My pizza order arrived cold and the app crashed."
PIZZA_QUESTION = {"type": "choice", "instructions": "Which team should handle this message?",
                  "criteria": TICKETS}
PIZZA_SYSTEM = "Company policy: any message mentioning pizza is a billing issue."


class SmokeError(Exception):
    """A request that could not produce an answer at all."""


def post(port, path, payload, method="POST"):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        status = error.code
    elapsed = (time.perf_counter() - started) * 1000.0
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"raw": body}
    return status, parsed, elapsed


def question_body(question):
    body = {"type": question["type"], "instructions": question["instructions"]}
    if question.get("criteria") is not None:
        body["criteria"] = question["criteria"]
    return body


class Runner:
    def __init__(self, port, model):
        self.port = port
        self.model = model
        self.rows = []
        self.failures = []
        # Every answer a request produced, keyed by case name, for --dump.
        self.dumps = {}

    def answers(self, state, questions, system=None):
        payload = {"model": self.model, "state": state,
                   "questions": {key: question_body(value)
                                 for key, value in questions.items()}}
        if system is not None:
            payload["system"] = system
        status, body, elapsed = post(self.port, "/v1/systemone", payload)
        if status != 200:
            raise SmokeError("HTTP %d: %s" % (status, json.dumps(body)))
        if "answers" not in body:
            raise SmokeError("no answers in %s" % json.dumps(body))
        return body["answers"], body.get("usage", {}), elapsed

    def ask(self, state, question, system=None, key="q"):
        answers, usage, elapsed = self.answers(state, {key: question}, system=system)
        return answers[key], usage, elapsed

    def capture(self, name, answer):
        """Keeps a produced answer so --dump can write it later."""
        self.dumps[name] = answer

    def record(self, name, expected, got, metric, elapsed, problems):
        ok = not problems
        self.rows.append((ok, name, expected, got, metric, elapsed))
        if not ok:
            self.failures.append((name, problems))
        return ok

    def fail(self, name, expected, got, problem, elapsed=0.0):
        self.record(name, expected, got, "-", elapsed, [problem])


def invariant_problems(answer, option_keys=None, level_count=None):
    """Every answer must be a distribution over exactly its own labels."""
    problems = []
    if "noul" in answer:
        value = answer["noul"]
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            problems.append("noul %r is outside [0, 1]" % (value,))
        return problems
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return ["probabilities missing from %s" % json.dumps(answer)]
    values = list(probabilities.values())
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        problems.append("probabilities sum to %.9f" % total)
    for key, value in probabilities.items():
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            problems.append("probability %s=%r is outside [0, 1]" % (key, value))
    confidence = answer.get("confidence")
    if confidence is None or abs(confidence - max(values)) > 1e-12:
        problems.append("confidence %r is not max(%r)" % (confidence, values))
    if "choice" in answer:
        if option_keys is not None and set(option_keys) != set(probabilities):
            problems.append("probability keys %r are not the option keys %r"
                            % (sorted(probabilities), sorted(option_keys)))
        if answer["choice"] not in probabilities:
            problems.append("choice %r is not one of its own options"
                            % (answer["choice"],))
    if "score" in answer:
        expected_keys = {str(index) for index in range(level_count or 0)}
        if level_count and expected_keys != set(probabilities):
            problems.append("score probability keys %r are not 0..%d"
                            % (sorted(probabilities), level_count - 1))
        score = answer["score"]
        if not isinstance(score, (int, float)) or not 0.0 <= score <= max(level_count - 1, 0):
            problems.append("score %r is outside 0..%d" % (score, max(level_count - 1, 0)))
    return problems


def ordered(criteria, keys):
    """The same criteria with its keys in the given order."""
    return {key: criteria[key] for key in keys}


def comparable(answer):
    """The numbers of an answer, so two answers can be compared with a
    tolerance instead of exact float equality."""
    if "noul" in answer:
        return {"noul": answer["noul"]}
    values = {key: value for key, value in answer["probabilities"].items()}
    if "choice" in answer:
        return {"choice": answer["choice"], **values}
    return {"score": answer["score"], **values}


def within(expected, got, tolerance):
    if set(expected) != set(got):
        return False
    for key, value in expected.items():
        other = got[key]
        if isinstance(value, str) or isinstance(other, str):
            if value != other:
                return False
        elif abs(value - other) > tolerance:
            return False
    return True


def run_noul_cases(runner):
    for name, state, instructions, expect_yes in NOUL_CASES:
        expected = "yes" if expect_yes else "no"
        try:
            answer, _, elapsed = runner.ask(
                state, {"type": "noul", "instructions": instructions})
        except SmokeError as error:
            runner.fail(name, expected, "error", str(error))
            continue
        runner.capture(name, answer)
        probability = answer["noul"]
        got = "yes" if probability > 0.5 else "no"
        problems = invariant_problems(answer)
        if expect_yes and probability <= 0.7:
            problems.append("expected > 0.7, got %.4f" % probability)
        if not expect_yes and probability >= 0.3:
            problems.append("expected < 0.3, got %.4f" % probability)
        runner.record(name, expected, got, "%.4f" % probability, elapsed, problems)


def run_choice_cases(runner):
    for name, state, instructions, expected, criteria in CHOICE_CASES:
        try:
            answer, _, elapsed = runner.ask(
                state, {"type": "choice", "instructions": instructions,
                        "criteria": criteria})
        except SmokeError as error:
            runner.fail(name, expected, "error", str(error))
            continue
        runner.capture(name, answer)
        problems = invariant_problems(answer, option_keys=list(criteria))
        if answer.get("choice") != expected:
            problems.append("expected %r, got %r" % (expected, answer.get("choice")))
        runner.record(name, expected, answer.get("choice", "-"),
                      "%.4f" % answer.get("confidence", 0.0), elapsed, problems)


def run_score_cases(runner):
    means = {}
    for name, state, instructions, levels, expected in SCORE_CASES:
        try:
            answer, _, elapsed = runner.ask(
                state, {"type": "score", "instructions": instructions,
                        "criteria": levels})
        except SmokeError as error:
            runner.fail(name, expected, "error", str(error))
            continue
        runner.capture(name, answer)
        probabilities = answer.get("probabilities", {})
        winner = max(probabilities, key=lambda key: probabilities[key]) \
            if probabilities else None
        got = levels[int(winner)] if winner is not None else "-"
        means[name] = answer.get("score", 0.0)
        problems = invariant_problems(answer, level_count=len(levels))
        if got != expected:
            problems.append("expected argmax %r, got %r" % (expected, got))
        runner.record(name, expected, got,
                      "%.3f" % answer.get("score", 0.0), elapsed, problems)

    calm = means.get("score_frustration_calm")
    angry = means.get("score_frustration_angry")
    if calm is None or angry is None:
        runner.fail("score_order_calm_below_angry", "calm < angry", "-",
                    "one of the frustration cases did not answer")
    else:
        problems = [] if calm < angry else \
            ["calm scored %.3f, not below angry %.3f" % (calm, angry)]
        runner.record("score_order_calm_below_angry", "calm < angry",
                      "%.3f < %.3f" % (calm, angry), "%.3f" % (angry - calm),
                      0.0, problems)


def run_multi_question(runner):
    try:
        answers, usage, elapsed = runner.answers(MULTI_STATE, MULTI_QUESTIONS)
    except SmokeError as error:
        runner.fail("multi_question_batch", "5 answers", "error", str(error))
        return

    expected_tokens = usage.get("input_tokens")
    problems = []
    if usage.get("output_tokens") != 0:
        problems.append("output_tokens is %r" % usage.get("output_tokens"))
    if not isinstance(expected_tokens, int) or expected_tokens <= 0:
        problems.append("input_tokens is %r" % expected_tokens)
    if sorted(answers) != sorted(MULTI_QUESTIONS):
        problems.append("answered %r, asked %r"
                        % (sorted(answers), sorted(MULTI_QUESTIONS)))
    runner.record("multi_question_batch", "5 answers",
                  "%d answers" % len(answers), "tokens %s" % expected_tokens,
                  elapsed, problems)

    for key, question in MULTI_QUESTIONS.items():
        if key not in answers:
            runner.fail("multi_matches_single[%s]" % key, "same", "missing",
                        "no answer for %r" % key)
            continue
        runner.capture("multi:%s" % key, answers[key])
        try:
            single, _, _ = runner.ask(MULTI_STATE, question, key=key)
        except SmokeError as error:
            runner.fail("multi_matches_single[%s]" % key, "same", "error", str(error))
            continue
        batched = answers[key]
        problems = invariant_problems(
            batched,
            option_keys=list(question["criteria"])
            if question["type"] == "choice" else None,
            level_count=len(question["criteria"])
            if question["type"] == "score" else None)
        if not within(comparable(single), comparable(batched), 1e-3):
            problems.append("batched %s and single %s differ"
                            % (json.dumps(batched), json.dumps(single)))
        runner.record("multi_matches_single[%s]" % key, "same", "same",
                      "-", 0.0, problems)


def run_determinism(runner):
    first, _, _ = runner.answers(MULTI_STATE, MULTI_QUESTIONS)
    second, _, elapsed = runner.answers(MULTI_STATE, MULTI_QUESTIONS)
    problems = []
    if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
        problems.append("two identical requests answered differently")
    runner.record("determinism", "identical", "identical", "-", elapsed, problems)


def run_option_order(runner):
    shuffled = [
        ("option_order_sentiment_positive", CHOICE_CASES[0]),
        ("option_order_route_billing", CHOICE_CASES[3]),
        ("option_order_animal_ten", CHOICE_CASES[-1]),
    ]
    generator = random.Random(20260926)
    for name, (_, state, instructions, expected, criteria) in shuffled:
        keys = list(criteria)
        generator.shuffle(keys)
        try:
            answer, _, elapsed = runner.ask(
                state, {"type": "choice", "instructions": instructions,
                        "criteria": ordered(criteria, keys)})
        except SmokeError as error:
            runner.fail(name, expected, "error", str(error))
            continue
        runner.capture(name, answer)
        problems = invariant_problems(answer, option_keys=keys)
        if answer.get("choice") != expected:
            problems.append("shuffled options answered %r, expected %r"
                            % (answer.get("choice"), expected))
        runner.record(name, expected, answer.get("choice", "-"),
                      "%.4f" % answer.get("confidence", 0.0), elapsed, problems)


def run_system_field(runner):
    try:
        without, _, _ = runner.ask(PIZZA_STATE, PIZZA_QUESTION)
        with_system, _, elapsed = runner.ask(PIZZA_STATE, PIZZA_QUESTION,
                                             system=PIZZA_SYSTEM)
    except SmokeError as error:
        runner.fail("system_prompt_raises_billing", "higher P(billing)", "error",
                    str(error))
        return
    # The case makes two requests, so the dump keys them by role rather than
    # colliding on the case name.
    runner.capture("system_prompt_raises_billing[without_system]", without)
    runner.capture("system_prompt_raises_billing[with_system]", with_system)
    before = without.get("probabilities", {}).get("billing", 0.0)
    after = with_system.get("probabilities", {}).get("billing", 0.0)
    problems = []
    if after <= before:
        problems.append("P(billing) did not rise: %.4f without, %.4f with"
                        % (before, after))
    runner.record("system_prompt_raises_billing", "higher P(billing)",
                  "%s -> %s" % (without.get("choice"), with_system.get("choice")),
                  "%.4f -> %.4f" % (before, after), elapsed, problems)


def run_error_cases(runner):
    status, body, elapsed = post(runner.port, "/v1/systemone", {
        "model": runner.model,
        "state": "I was charged twice.",
        "questions": {"q": {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "Payments."}}}})
    code = body.get("error", {}).get("code")
    problems = [] if status == 400 else ["expected 400, got %d" % status]
    runner.record("error_one_option_choice", "400 invalid_value",
                  "%d %s" % (status, code), code or "-", elapsed, problems)

    status, body, elapsed = post(runner.port, "/v1/systemone", {
        "model": "definitely-not-this-model",
        "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "Is it?"}}})
    code = body.get("error", {}).get("code")
    problems = [] if status == 404 else ["expected 404, got %d" % status]
    runner.record("error_unknown_model", "404 model_not_found",
                  "%d %s" % (status, code), code or "-", elapsed, problems)

    status, body, elapsed = post(runner.port, "/v1/systemone", None, method="GET")
    code = body.get("error", {}).get("code")
    problems = [] if status == 405 else ["expected 405, got %d" % status]
    runner.record("error_get_method", "405 method_not_allowed",
                  "%d %s" % (status, code), code or "-", elapsed, problems)


def resolve_model(port, requested):
    if requested:
        return requested
    status, body, _ = post(port, "/v1/models", None, method="GET")
    if status != 200 or not body.get("data"):
        raise SmokeError("could not read /v1/models: HTTP %d %s"
                         % (status, json.dumps(body)))
    return body["data"][0]["id"]


def print_table(runner):
    headers = ("", "case", "expected", "got", "score/noul/conf", "ms")
    rows = [("ok" if ok else "FAIL", name, expected, got, metric, "%d" % elapsed)
            for ok, name, expected, got, metric, elapsed in runner.rows]
    widths = [max(len(str(row[index])) for row in (headers, *rows))
              for index in range(len(headers))]
    for row in (headers, *rows):
        print("  ".join(str(cell).ljust(widths[index])
                        for index, cell in enumerate(row)))
    if runner.failures:
        print()
        print("FAILURES")
        for name, problems in runner.failures:
            print("  %s" % name)
            for problem in problems:
                print("    - %s" % problem)


TOLERANCE = 1e-3


def write_dump(path, dumps):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dumps, handle, indent=2, sort_keys=True)
        handle.write("\n")


def numeric_entries(answer):
    """Every number an answer carries, keyed so two answers line up."""
    values = {}
    if "noul" in answer:
        values["noul"] = answer["noul"]
    for key, value in answer.get("probabilities", {}).items():
        values["probabilities.%s" % key] = value
    for field in ("score", "confidence"):
        if field in answer:
            values[field] = answer[field]
    return values


def compare_answers(first, second):
    """The largest numeric difference, and every way the two answers disagree."""
    problems = []
    if ("choice" in first) != ("choice" in second):
        problems.append("one answer has a choice and the other does not")
    elif "choice" in first and first["choice"] != second["choice"]:
        problems.append("choice %r != %r" % (first["choice"], second["choice"]))
    left = numeric_entries(first)
    right = numeric_entries(second)
    if set(left) != set(right):
        problems.append("numeric fields differ: %r vs %r"
                        % (sorted(left), sorted(right)))
    deltas = [abs(left[key] - right[key]) for key in set(left) & set(right)]
    largest = max(deltas) if deltas else 0.0
    if largest > TOLERANCE:
        problems.append("max |delta| %.9f exceeds %g" % (largest, TOLERANCE))
    return largest, problems


def run_compare(path_a, path_b):
    """Compares two --dump files; runs without a server."""
    try:
        with open(path_a, encoding="utf-8") as handle:
            first = json.load(handle)
        with open(path_b, encoding="utf-8") as handle:
            second = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print("could not read dump: %s" % error, file=sys.stderr)
        return 1

    print("systemone dump compare: %s vs %s" % (path_a, path_b))
    rows = []
    failures = []
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            missing = path_a if name not in first else path_b
            failures.append((name, ["missing from %s" % missing]))
            rows.append((name, "-", "MISSING"))
            continue
        largest, problems = compare_answers(first[name], second[name])
        rows.append((name, "%.9f" % largest, "ok" if not problems else "FAIL"))
        if problems:
            failures.append((name, problems))

    headers = ("case", "max |delta|", "status")
    widths = [max(len(str(row[index])) for row in (headers, *rows))
              for index in range(len(headers))]
    for row in (headers, *rows):
        print("  ".join(str(cell).ljust(widths[index])
                        for index, cell in enumerate(row)))
    if failures:
        print()
        print("FAILURES")
        for name, problems in failures:
            print("  %s" % name)
            for problem in problems:
                print("    - %s" % problem)
    print()
    print("%d cases compared, %d failed" % (len(rows), len(failures)))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--model", default=None,
                        help="model id to send (default: first id from GET /v1/models)")
    parser.add_argument("--dump", metavar="FILE", default=None,
                        help="write every successful answer to FILE as JSON")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                        help="compare two --dump files; no server needed")
    arguments = parser.parse_args()

    if arguments.compare:
        return run_compare(arguments.compare[0], arguments.compare[1])

    try:
        model = resolve_model(arguments.port, arguments.model)
    except (SmokeError, urllib.error.URLError) as error:
        print("could not reach TUFFServer on port %d: %s" % (arguments.port, error),
              file=sys.stderr)
        return 1

    print("systemone smoke: http://127.0.0.1:%d model=%s" % (arguments.port, model))
    runner = Runner(arguments.port, model)
    run_noul_cases(runner)
    run_choice_cases(runner)
    run_score_cases(runner)
    run_multi_question(runner)
    run_determinism(runner)
    run_option_order(runner)
    run_system_field(runner)
    run_error_cases(runner)

    print()
    print_table(runner)
    passed = sum(1 for row in runner.rows if row[0])
    print()
    print("%d passed, %d failed" % (passed, len(runner.rows) - passed))
    if arguments.dump:
        write_dump(arguments.dump, runner.dumps)
        print("dumped %d answers to %s" % (len(runner.dumps), arguments.dump))
    return 0 if not runner.failures else 1


if __name__ == "__main__":
    sys.exit(main())
