#!/usr/bin/env python3
"""Accuracy and calibration eval for POST /v1/systemone.

Standard library only (runs on the Mac mini's Python 3.9). Three steps:

    # 1. Download and freeze the item sets (cached under scratch/systemone_eval).
    python3 Scripts/systemone_eval.py fetch

    # 2. Run a request variant against a running TUFFServer.
    python3 Scripts/systemone_eval.py run --port 8089 --tag apodex --variant base

    # 3. Score the stored runs on the TEST split, with eval-side transforms.
    python3 Scripts/systemone_eval.py report --tags apodex kat ornith

Every set has 300 items drawn with a fixed seed: 100 calibration items (used
only to fit transforms such as temperature or batch calibration, never to
score) and 200 test items (the only ones reported).

Request variants (`run --variant`) change what is sent to the server:
    base          the endpoint's own prompt
    cf            content-free probes (Zhao et al. 2021): "N/A", "[MASK]", ""
    perm          every choice question also asked under cyclic option shifts
                  in the same request (shared prefix), for permutation
                  averaging and PriDe
    fewshot<k>    k labelled examples from the set's train split, sent in the
                  request's `system` text so they sit in the shared prefix
    sys-<name>    one of FRAMINGS below as the request's `system` text

`run --save-as <name>` stores a `base` request run under another name, for
server-side options such as `--systemone-label-variants on`.

Eval-side transforms (`report`) need no extra requests except where noted:
    raw, temp (per-set T), gtemp (one T per model), cc (needs cf),
    bc, bc+temp, permavg (needs perm), pride (needs perm on cal only).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SEED = 20260927
CAL_COUNT, TEST_COUNT = 100, 200
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch", "systemone_eval")
DATA_DIR = os.path.join(ROOT, "data")
ITEMS_DIR = os.path.join(ROOT, "items")
RUNS_DIR = os.path.join(ROOT, "runs")

ROWS_API = "https://datasets-server.huggingface.co/rows"
KEV_COMMIT = "5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109"
KEV_FILES = {
    "test": ("evals/hard-v1/test.jsonl",
             "246ee92234d33f8e3f10ac6e1542651531b128fd7cc66f3a9964589b77dfa22b"),
    "development": ("evals/hard-v1/development.jsonl",
                    "3cdbf12a6b7c3b70c73e8c39f2677127ea61ef32f0ea75b55a43556d884b2d98"),
}

YES_NO = ["Yes", "No"]

AG_NEWS = [
    ("world", "World news: international affairs, politics, conflicts, and governments."),
    ("sports", "Sports: games, athletes, teams, and competitions."),
    ("business", "Business: companies, markets, the economy, and finance."),
    ("sci_tech", "Science and technology: computing, the internet, research, and space."),
]

# TREC coarse classes in the dataset's label_coarse order.
TREC = {
    "ABBR": ("abbreviation", "Asks for an abbreviation or what an abbreviation stands for."),
    "ENTY": ("entity", "Asks for an entity: an object, animal, product, event, colour, term, or similar."),
    "DESC": ("description", "Asks for a description, definition, reason, or manner."),
    "HUM": ("human", "Asks for a person, a group of people, or an organisation."),
    "LOC": ("location", "Asks for a place: a city, country, mountain, or other location."),
    "NUM": ("numeric", "Asks for a number: a date, count, amount, distance, or other value."),
}

MNLI = [
    ("entailment", "The hypothesis must be true if the premise is true."),
    ("neutral", "The hypothesis might be true or false; the premise does not settle it."),
    ("contradiction", "The hypothesis must be false if the premise is true."),
]

SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
AMAZON_LEVELS = ["1 star: very dissatisfied", "2 stars: dissatisfied", "3 stars: mixed",
                 "4 stars: satisfied", "5 stars: very satisfied"]

BANKING_INTENTS = 26

# Prompt-framing variants. Each is sent as the request's `system` text, which
# the server appends after its fixed framing line.
FRAMINGS = {
    "evidence": "Base the answer only on what the state says. When the state does not settle "
                "the question, choose the label that is most likely given the state.",
    "expert": "You are an expert annotator. Read the state carefully, consider every label, and "
              "answer with the label a careful human annotator would choose.",
    "calib": "Your answer is read as a probability. If you are unsure, it is fine to be unsure; "
             "do not be more confident than the state supports.",
}

SETS = {
    "boolq": dict(dataset="google/boolq", split="validation", pool="train"),
    "sst2": dict(dataset="stanfordnlp/sst2", split="validation", pool="train"),
    "agnews": dict(dataset="fancyzhx/ag_news", split="test", pool="train"),
    "trec": dict(dataset="SetFit/TREC-QC", split="test", pool="train"),
    "mnli": dict(dataset="nyu-mll/glue", config="mnli", split="validation_matched", pool="train"),
    "banking77": dict(dataset="mteb/banking77", split="test", pool="train"),
    "sst5": dict(dataset="SetFit/sst5", split="test", pool="train"),
    "amazon": dict(dataset="SetFit/amazon_reviews_multi_en", split="test", pool="train"),
    "kev_hard": dict(kev=True),
}


# ---------------------------------------------------------------------------
# Download

def http_json(url, attempts=6):
    delay = 2.0
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            print("  retry %s after %s" % (url[:90], error), file=sys.stderr)
            time.sleep(delay)
            delay *= 2


def fetch_rows(dataset, config, split, offset, length):
    query = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split,
                                        offset=offset, length=length))
    return http_json("%s?%s" % (ROWS_API, query))


def fetch_split(dataset, config, split):
    """Every row of a split, cached; the dataset's Hub commit is recorded."""
    path = os.path.join(DATA_DIR, "%s__%s__%s.json" % (dataset.replace("/", "__"), config, split))
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)["rows"]
    first = fetch_rows(dataset, config, split, 0, 100)
    total = first["num_rows_total"]
    rows = [entry["row"] for entry in first["rows"]]
    for offset in range(100, total, 100):
        rows.extend(entry["row"] for entry in fetch_rows(dataset, config, split, offset, 100)["rows"])
    names = {}
    for feature in first["features"]:
        if "names" in feature["type"]:
            names[feature["name"]] = feature["type"]["names"]
    info = http_json("https://huggingface.co/api/datasets/%s" % dataset)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"dataset": dataset, "config": config, "split": split, "sha": info.get("sha"),
                   "names": names, "rows": rows}, handle)
    print("  %s %s/%s: %d rows (sha %s)" % (dataset, config, split, len(rows), info.get("sha")))
    return rows


def fetch_pool(dataset, config, split, rng, pages=5):
    """A few seeded pages of the train split for few-shot examples; downloading
    whole train splits (up to 400k rows) buys nothing."""
    path = os.path.join(DATA_DIR, "%s__%s__%s__pool.json" % (dataset.replace("/", "__"), config, split))
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    total = fetch_rows(dataset, config, split, 0, 1)["num_rows_total"]
    offsets = sorted(rng.sample(range(0, max(1, total - 100)), pages))
    rows = []
    for offset in offsets:
        rows.extend(entry["row"] for entry in fetch_rows(dataset, config, split, offset, 100)["rows"])
    with open(path, "w") as handle:
        json.dump(rows, handle)
    return rows


def fetch_kev(name):
    relative, digest = KEV_FILES[name]
    path = os.path.join(DATA_DIR, "kev_hard_%s.jsonl" % name)
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/jaredpalmer/kev/%s/%s" % (KEV_COMMIT, relative)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
    with open(path, "rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit("kev %s does not match its pinned sha256" % name)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Items

def single(item_id, state, qtype, instructions, criteria, labels, gold):
    body = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        body["criteria"] = criteria
    return {"id": item_id, "state": state,
            "questions": {"q": body}, "gold": {"q": gold}, "labels": {"q": labels}}


def choice_criteria(options):
    return {key: description for key, description in options}


def build_item(name, index, row, names):
    """One item for a row of an ordinary set, or None when the row has no label."""
    item_id = "%s-%d" % (name, index)
    if name == "boolq":
        question = row["question"].strip()
        question = question[0].upper() + question[1:] + "?"
        return single(item_id, row["passage"], "noul", question, None, YES_NO,
                      0 if row["answer"] else 1)
    if name == "sst2":
        if row["label"] < 0:
            return None
        return single(item_id, row["sentence"].strip(), "noul",
                      "Is the sentiment of this movie review positive?", None, YES_NO,
                      0 if row["label"] == 1 else 1)
    if name == "agnews":
        return single(item_id, row["text"], "choice", "Which topic does this news article belong to?",
                      choice_criteria(AG_NEWS), [key for key, _ in AG_NEWS], row["label"])
    if name == "trec":
        order = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        options = [TREC[code] for code in order]
        return single(item_id, row["text"], "choice", "What kind of answer is this question asking for?",
                      choice_criteria(options), [key for key, _ in options],
                      order.index(row["label_coarse_original"]))
    if name == "mnli":
        state = "Premise: %s\nHypothesis: %s" % (row["premise"], row["hypothesis"])
        return single(item_id, state, "choice", "How does the premise relate to the hypothesis?",
                      choice_criteria(MNLI), [key for key, _ in MNLI], row["label"])
    if name == "banking77":
        intents = names["banking_intents"]
        if row["label_text"] not in intents:
            return None
        options = [(intent, "Customer asks about %s." % intent.replace("_", " ")) for intent in intents]
        return single(item_id, row["text"], "choice",
                      "Which banking intent best describes this customer message?",
                      choice_criteria(options), intents, intents.index(row["label_text"]))
    if name == "sst5":
        return single(item_id, row["text"].strip(), "score", "How positive is this movie review?",
                      SST5_LEVELS, SST5_LEVELS, row["label"])
    if name == "amazon":
        return single(item_id, row["text"], "score", "How many stars did the reviewer give?",
                      AMAZON_LEVELS, AMAZON_LEVELS, row["label"])
    raise KeyError(name)


def kev_item(item_id, row):
    """A Kev row keeps all its questions in one request, like System One traffic.

    Kev leaves some option descriptions null; the endpoint needs a string, so a
    null description becomes the option key in words."""
    questions, gold, labels = {}, {}, {}
    for key, question in row["questions"].items():
        body = {"type": question["type"], "instructions": question["instructions"]}
        if question["type"] == "choice":
            criteria = {option: (text if text is not None else option.replace("_", " "))
                        for option, text in question["criteria"].items()}
            body["criteria"] = criteria
            labels[key] = list(criteria)
            gold[key] = labels[key].index(question["label"])
        elif question["type"] == "score":
            body["criteria"] = question["criteria"]
            labels[key] = list(question["criteria"])
            gold[key] = int(question["label"])
        else:
            if question.get("criteria") is not None:
                body["criteria"] = question["criteria"]
            labels[key] = YES_NO
            gold[key] = 0 if question["label"] else 1
        questions[key] = body
    return {"id": item_id, "state": row["state"], "questions": questions,
            "gold": gold, "labels": labels}


def build_set(name):
    rng = random.Random("%d:%s" % (SEED, name))
    spec = SETS[name]
    if spec.get("kev"):
        rows = fetch_kev("test")
        chosen = rng.sample(range(len(rows)), CAL_COUNT + TEST_COUNT)
        items = [kev_item("kev_hard-%d" % index, rows[index]) for index in chosen]
        pool_rows = fetch_kev("development")
        pool = [kev_item("kev_dev-%d" % index, row) for index, row in enumerate(pool_rows)]
        meta = {"source": "github.com/jaredpalmer/kev@%s evals/hard-v1" % KEV_COMMIT}
    else:
        config = spec.get("config", "default")
        rows = fetch_split(spec["dataset"], config, spec["split"])
        names = {}
        if name == "banking77":
            every = sorted(set(row["label_text"] for row in rows))
            names["banking_intents"] = sorted(rng.sample(every, BANKING_INTENTS))
        candidates = []
        for index, row in enumerate(rows):
            item = build_item(name, index, row, names)
            if item is not None:
                candidates.append(item)
        items = rng.sample(candidates, CAL_COUNT + TEST_COUNT)
        pool_rows = fetch_pool(spec["dataset"], config, spec["pool"], rng)
        pool = []
        for index, row in enumerate(pool_rows):
            item = build_item(name, index, row, names)
            if item is not None:
                item["id"] = "%s-pool-%d" % (name, index)
                pool.append(item)
        meta = {"source": "%s %s/%s" % (spec["dataset"], config, spec["split"]), "names": names}
    for position, item in enumerate(items):
        item["split"] = "cal" if position < CAL_COUNT else "test"
        item["set"] = name
    os.makedirs(ITEMS_DIR, exist_ok=True)
    with open(os.path.join(ITEMS_DIR, "%s.json" % name), "w") as handle:
        json.dump({"meta": meta, "items": items, "pool": pool}, handle)
    print("%s: %d items, %d pool" % (name, len(items), len(pool)))


def load_set(name):
    with open(os.path.join(ITEMS_DIR, "%s.json" % name)) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Request variants

def letter(index):
    return chr(65 + index)


def answer_text(item, key):
    gold = item["gold"][key]
    if item["questions"][key]["type"] == "noul":
        return YES_NO[gold]
    return "%s (%s)" % (letter(gold), item["labels"][key][gold])


def fewshot_system(name, pool, count):
    """k labelled pool examples, class-balanced round robin, short states only.

    The question itself follows in the prompt's suffix, so an example is its
    state and its answer; Kev rows vary their question, so theirs are shown too."""
    rng = random.Random("%d:%s:fewshot" % (SEED, name))
    short = [item for item in pool if len(item["state"]) <= 700]
    by_class = {}
    for item in short:
        key = list(item["questions"])[0]
        by_class.setdefault(item["gold"][key], []).append(item)
    for bucket in by_class.values():
        rng.shuffle(bucket)
    classes = sorted(by_class)
    rng.shuffle(classes)
    chosen = []
    while len(chosen) < count and any(by_class.values()):
        for label in classes:
            if by_class[label] and len(chosen) < count:
                chosen.append(by_class[label].pop())
    rng.shuffle(chosen)
    parts = ["Worked examples of states and their correct answers:"]
    for number, item in enumerate(chosen, 1):
        key = list(item["questions"])[0]
        lines = ["<example %d>" % number, "<state>", item["state"], "</state>"]
        if SETS[name].get("kev"):
            lines.append("Question: %s" % item["questions"][key]["instructions"])
        lines.append("Answer: %s" % answer_text(item, key))
        lines.append("</example %d>" % number)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def perm_shifts(option_count):
    """Full cyclic shifts for small option sets; four spread shifts otherwise,
    which keeps a 26-intent question at four suffix prefills."""
    if option_count <= 6:
        return list(range(option_count))
    return [0] + [round(option_count * step / 4) for step in (1, 2, 3)]


def shifted(body, labels, shift):
    """The question with its options rotated so position i shows option (i+shift) % n."""
    count = len(labels)
    order = [labels[(position + shift) % count] for position in range(count)]
    return dict(body, criteria={key: body["criteria"][key] for key in order})


def request_for(item, variant, system_text):
    questions = {}
    plan = {}
    for key, body in item["questions"].items():
        if variant == "perm" and body["type"] == "choice":
            for shift in perm_shifts(len(item["labels"][key])):
                questions["%s__s%d" % (key, shift)] = shifted(body, item["labels"][key], shift)
                plan["%s__s%d" % (key, shift)] = (key, shift)
        else:
            questions[key] = body
            plan[key] = (key, 0)
    payload = {"state": item["state"], "questions": questions}
    if system_text is not None:
        payload["system"] = system_text
    return payload, plan


def post(port, payload, timeout=900):
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/systemone" % port, data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError("HTTP %d: %s" % (error.code, error.read().decode("utf-8")[:400]))
    return body, (time.perf_counter() - started) * 1000.0


def probabilities(answer, body):
    """The answer as a probability list in the order the request listed labels."""
    if body["type"] == "noul":
        return [answer["noul"], 1.0 - answer["noul"]]
    if body["type"] == "choice":
        return [answer["probabilities"][key] for key in body["criteria"]]
    return [answer["probabilities"][str(index)] for index in range(len(body["criteria"]))]


def server_model(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))["data"][0]["id"]


def content_free_items(data):
    """Zhao et al.'s content-free inputs, asked with each distinct question of
    the set (ordinary sets have one)."""
    template = data["items"][0]
    items = []
    for probe in ("N/A", "[MASK]", ""):
        item = dict(template, id="%s-cf-%s" % (template["set"], probe or "empty"),
                    state=probe, split="cf")
        items.append(item)
    return items


def run(args):
    model = server_model(args.port)
    # --save-as stores a run under its own name when the request is the same
    # but the server was started with different flags.
    path = os.path.join(RUNS_DIR, args.tag, "%s.jsonl" % (args.save_as or args.variant))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = set(json.loads(line)["id"] for line in handle if line.strip())
    sets = args.sets or list(SETS)
    # Untimed warm-up: the first requests after a start read experts from a
    # cold SSD and would land in whichever set runs first.
    warm = load_set(sets[0])["items"][0]
    for _ in range(args.warmup):
        payload, _ = request_for(warm, "base", None)
        payload["model"] = model
        post(args.port, payload)
    with open(path, "a") as out:
        for name in sets:
            data = load_set(name)
            if args.variant == "cf":
                if SETS[name].get("kev"):
                    continue  # every Kev row asks its own questions; no shared template
                items = content_free_items(data)
            else:
                items = [item for item in data["items"]
                         if args.split == "all" or item["split"] == args.split]
            if args.limit:
                items = items[:args.limit]
            system_text = None
            if args.variant.startswith("fewshot"):
                system_text = fewshot_system(name, data["pool"], int(args.variant[len("fewshot"):]))
            elif args.variant.startswith("sys-"):
                system_text = FRAMINGS[args.variant[len("sys-"):]]
            started, count = time.time(), 0
            for item in items:
                if item["id"] in done:
                    continue
                payload, plan = request_for(item, "perm" if args.variant == "perm" else "base",
                                            system_text)
                payload["model"] = model
                body, elapsed = post(args.port, payload)
                probs = {}
                for sent_key, (key, shift) in plan.items():
                    values = probabilities(body["answers"][sent_key], payload["questions"][sent_key])
                    if shift:
                        # Back to the original option order: position i showed option (i+shift) % n.
                        count_labels = len(values)
                        original = [0.0] * count_labels
                        for position, value in enumerate(values):
                            original[(position + shift) % count_labels] = value
                        values = original
                    probs.setdefault(key, {})[str(shift)] = values
                record = {"id": item["id"], "set": name, "split": item["split"],
                          "latency_ms": round(elapsed, 1),
                          "input_tokens": body.get("usage", {}).get("input_tokens"),
                          "probs": probs, "gold": item["gold"],
                          "types": {key: item["questions"][key]["type"] for key in item["questions"]}}
                out.write(json.dumps(record) + "\n")
                out.flush()
                count += 1
            if count:
                print("%s %s %s: %d requests in %.0fs" % (args.tag, args.save_as or args.variant, name, count,
                                                          time.time() - started), flush=True)


# ---------------------------------------------------------------------------
# Metrics

def argmax(values):
    best = 0
    for index in range(len(values)):
        if values[index] > values[best]:
            best = index
    return best


def normalise(values):
    total = sum(values)
    return [value / total for value in values]


def power(probs, temperature):
    """Temperature on log-probabilities: p^(1/T), renormalised."""
    logs = [math.log(max(value, 1e-300)) / temperature for value in probs]
    top = max(logs)
    return normalise([math.exp(value - top) for value in logs])


def metrics(records):
    """records: list of (probs, gold). Brier is the multi-class sum over labels."""
    n = len(records)
    correct = 0
    brier = nll = confident_errors = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    gold_counts, pred_counts, hit_counts = {}, {}, {}
    for probs, gold in records:
        pred = argmax(probs)
        confidence = probs[pred]
        hit = pred == gold
        correct += hit
        brier += sum((value - (1.0 if index == gold else 0.0)) ** 2 for index, value in enumerate(probs))
        nll += -math.log(max(probs[gold], 1e-12))
        if confidence >= 0.9 and not hit:
            confident_errors += 1
        slot = min(14, int(confidence * 15))
        bins[slot][0] += 1
        bins[slot][1] += hit
        bins[slot][2] += confidence
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        if hit:
            hit_counts[gold] = hit_counts.get(gold, 0) + 1
    f1s = []
    for label in set(gold_counts) | set(pred_counts):
        tp = hit_counts.get(label, 0)
        precision = tp / pred_counts[label] if pred_counts.get(label) else 0.0
        recall = tp / gold_counts[label] if gold_counts.get(label) else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    ece = sum(abs(hits - conf) for count, hits, conf in bins if count) / n
    return {"n": n, "acc": correct / n, "f1": sum(f1s) / len(f1s), "ece": ece,
            "brier": brier / n, "nll": nll / n, "cerr": confident_errors / n}


def fit_temperature(records):
    """T minimising NLL, by golden-section search on log T in [-3, 3]."""
    if not records:
        return 1.0
    def loss(log_t):
        t = math.exp(log_t)
        return sum(-math.log(max(power(probs, t)[gold], 1e-12)) for probs, gold in records)
    low, high = -3.0, 3.0
    ratio = (math.sqrt(5) - 1) / 2
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = loss(a), loss(b)
    for _ in range(60):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = loss(a)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = loss(b)
    return math.exp((low + high) / 2)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q / 100.0 * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


# ---------------------------------------------------------------------------
# Report

def load_run(tag, variant):
    path = os.path.join(RUNS_DIR, tag, "%s.jsonl" % variant)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_name(record, key):
    """Kev rows mix question types; each type is scored as its own group."""
    if record["set"] == "kev_hard":
        return "kev_%s" % record["types"][key]
    return record["set"]


def flatten(run_records, split):
    """(group, n_labels) -> list of (record id, probs-by-shift, gold)."""
    groups = {}
    for record in run_records:
        if record["split"] != split:
            continue
        for key, by_shift in record["probs"].items():
            base = by_shift["0"]
            groups.setdefault(group_name(record, key), []).append(
                (record["id"], key, by_shift, record["gold"][key], len(base)))
    return groups


def prior_from(rows):
    """Mean probability per position, per option count."""
    sums, counts = {}, {}
    for _, _, by_shift, _, n in rows:
        values = by_shift["0"]
        acc = sums.setdefault(n, [0.0] * n)
        for index in range(n):
            acc[index] += values[index]
        counts[n] = counts.get(n, 0) + 1
    return {n: [value / counts[n] for value in acc] for n, acc in sums.items()}


def pride_prior(rows):
    """PriDe (Zheng et al. 2024): with option o shown at position d under each
    cyclic shift, log p_obs(d) = log prior(d) + log p(o) + c. Averaging the
    observed log-probability of each POSITION over the shifts cancels the
    option term, leaving the position prior; averaged over calibration items."""
    sums, counts = {}, {}
    for _, _, by_shift, _, n in rows:
        shifts = [int(s) for s in by_shift]
        if len(shifts) < 2:
            continue
        # by_shift holds probabilities re-mapped to ORIGINAL option order;
        # position i under shift s showed option (i + s) % n.
        logs = [0.0] * n
        for shift in shifts:
            values = by_shift[str(shift)]
            for position in range(n):
                logs[position] += math.log(max(values[(position + shift) % n], 1e-300))
        logs = [value / len(shifts) for value in logs]
        top = max(logs)
        prior = normalise([math.exp(value - top) for value in logs])
        acc = sums.setdefault(n, [0.0] * n)
        for index in range(n):
            acc[index] += prior[index]
        counts[n] = counts.get(n, 0) + 1
    return {n: [value / counts[n] for value in acc] for n, acc in sums.items()}


def divide(probs, prior):
    if prior is None:
        return probs
    return normalise([value / max(weight, 1e-12) for value, weight in zip(probs, prior)])


def transformed(tag, request_variant, transform, cal, test, extras):
    """Test-split (probs, gold) per group for one transform; None if unavailable."""
    out = {}
    for group, rows in test.items():
        cal_rows = cal.get(group, [])
        if transform == "raw":
            out[group] = [(r[2]["0"], r[3]) for r in rows]
        elif transform == "temp":
            t = fit_temperature([(r[2]["0"], r[3]) for r in cal_rows])
            out[group] = [(power(r[2]["0"], t), r[3]) for r in rows]
        elif transform == "gtemp":
            t = extras["gtemp"]
            out[group] = [(power(r[2]["0"], t), r[3]) for r in rows]
        elif transform in ("bc", "bc+temp"):
            prior = prior_from(cal_rows)
            fitted = [(divide(r[2]["0"], prior.get(r[4])), r[3]) for r in cal_rows]
            t = fit_temperature(fitted) if transform == "bc+temp" else 1.0
            out[group] = [(power(divide(r[2]["0"], prior.get(r[4])), t), r[3]) for r in rows]
        elif transform == "cc":
            # Kev rows ask their own questions, so they have no content-free
            # probe; those groups pass through raw (the report says so).
            cf = extras.get("cf", {}).get(group, {})
            out[group] = [(divide(r[2]["0"], cf.get(r[4])), r[3]) for r in rows]
        elif transform == "permavg":
            if not any(len(r[2]) > 1 for r in rows):
                out[group] = [(r[2]["0"], r[3]) for r in rows]
            else:
                out[group] = [(normalise([sum(v[i] for v in r[2].values()) / len(r[2])
                                          for i in range(r[4])]), r[3]) for r in rows]
        elif transform == "pride":
            perm_cal = extras.get("perm_cal", {}).get(group)
            prior = pride_prior(perm_cal) if perm_cal else {}
            out[group] = [(divide(r[2]["0"], prior.get(r[4])), r[3]) for r in rows]
        else:
            raise KeyError(transform)
    return out


GROUP_ORDER = ["boolq", "sst2", "kev_noul", "agnews", "trec", "mnli", "banking77", "kev_choice",
               "sst5", "amazon", "kev_score"]
COLUMNS = ["acc", "f1", "ece", "brier", "nll", "cerr"]


def latency_by_set(run_records):
    by_set = {}
    for record in run_records:
        if record["split"] == "test":
            by_set.setdefault(record["set"], []).append(record["latency_ms"])
    return by_set


def report(args):
    lines = []
    summary = []
    for tag in args.tags:
        for request_variant in args.variants:
            records = load_run(tag, request_variant)
            if not records:
                continue
            cal = flatten(records if request_variant != "perm" else load_run(tag, "base") or [], "cal")
            test = flatten(records, "test")
            extras = {}
            pooled = [(r[2]["0"], r[3]) for rows in cal.values() for r in rows]
            extras["gtemp"] = fit_temperature(pooled) if pooled else 1.0
            cf_records = load_run(tag, "cf") if request_variant == "base" else None
            if cf_records:
                extras["cf"] = {g: prior_from(rows) for g, rows in flatten(cf_records, "cf").items()}
            perm_records = load_run(tag, "perm")
            if perm_records:
                extras["perm_cal"] = flatten(perm_records, "cal")
            latencies = latency_by_set(records)
            transforms = ["raw"] if request_variant == "perm" else args.transforms
            if request_variant == "perm":
                transforms = ["permavg"]
            for transform in transforms:
                if transform in ("cc",) and not cf_records:
                    continue
                if transform == "pride" and (request_variant != "base" or not perm_records):
                    continue
                if transform == "permavg" and request_variant != "perm":
                    continue
                result = transformed(tag, request_variant, transform, cal, test, extras)
                if result is None:
                    continue
                name = "%s / %s / %s" % (tag, request_variant, transform)
                lines.append("\n#### %s\n" % name)
                if transform == "cc":
                    lines.append("Kev groups have no content-free probe and are reported raw.\n")
                if transform == "gtemp":
                    lines.append("Global temperature T = %.3f (fitted on all calibration items)\n"
                                 % extras["gtemp"])
                lines.append("| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                rows_out = []
                for group in GROUP_ORDER:
                    if group not in result:
                        continue
                    m = metrics(result[group])
                    set_name = "kev_hard" if group.startswith("kev_") else group
                    lat = latencies.get(set_name, [])
                    rows_out.append((group, m))
                    lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f | %.0f |" % (
                        group, m["n"], m["acc"], m["f1"], m["ece"], m["brier"], m["nll"], m["cerr"],
                        percentile(lat, 50), percentile(lat, 95)))
                every = [lat for values in latencies.values() for lat in values]
                mean = {c: sum(m[c] for _, m in rows_out) / len(rows_out) for c in COLUMNS}
                lines.append("| **mean** | | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f | %.0f |" % (
                    mean["acc"], mean["f1"], mean["ece"], mean["brier"], mean["nll"], mean["cerr"],
                    percentile(every, 50), percentile(every, 95)))
                summary.append((tag, request_variant, transform, mean, percentile(every, 50),
                                percentile(every, 95), {g: m for g, m in rows_out}))
    out = ["## Summary (TEST split, unweighted mean over groups)\n",
           "| model | request | transform | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, variant, transform, mean, p50, p95, _ in summary:
        out.append("| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f | %.0f |" % (
            tag, variant, transform, mean["acc"], mean["f1"], mean["ece"], mean["brier"],
            mean["nll"], mean["cerr"], p50, p95))
    text = "\n".join(out) + "\n\n## Per model x variant\n" + "\n".join(lines) + "\n"
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([{"tag": s[0], "request": s[1], "transform": s[2], "mean": s[3],
                        "p50": s[4], "p95": s[5], "groups": s[6]} for s in summary], handle, indent=1)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--sets", nargs="*")
    runner = sub.add_parser("run")
    runner.add_argument("--port", type=int, default=8089)
    runner.add_argument("--tag", required=True, help="model label for the run directory")
    runner.add_argument("--variant", default="base")
    runner.add_argument("--sets", nargs="*")
    runner.add_argument("--split", choices=["cal", "test", "all"], default="all")
    runner.add_argument("--limit", type=int, default=0)
    runner.add_argument("--warmup", type=int, default=2)
    runner.add_argument("--save-as", default=None,
                        help="run file name when server flags, not the request, differ")
    reporter = sub.add_parser("report")
    reporter.add_argument("--tags", nargs="+", required=True)
    reporter.add_argument("--variants", nargs="+", default=["base"])
    reporter.add_argument("--transforms", nargs="+",
                          default=["raw", "temp", "gtemp", "cc", "bc", "bc+temp", "pride"])
    reporter.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        for name in args.sets or list(SETS):
            build_set(name)
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
