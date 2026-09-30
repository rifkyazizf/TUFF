# Project context

Current state only — git is the changelog.

## What this fork is

Fork of [rexmhall09/TUFF](https://github.com/rexmhall09/TUFF) (remote `parent`; `upstream` is
drumih/turbo-fieldfare). TUFF is a native Swift + Metal inference engine for Apple Silicon that streams
mixture-of-experts weights from SSD through a bounded expert cache, so 35B-class MoE models run on 16 GB Macs.

This fork adds a **TypeSafe-Jev-compatible "System One" typed-decision endpoint** to `TUFFServer`
(branch `systemone-endpoint`, not merged or pushed).

## System One endpoint

`POST /v1/systemone` — request `{model, state, questions: {key: {type, instructions, criteria}}, system?}`,
types `noul` (yes/no), `choice` (2–26 options), `score` (2–10 levels). Returns label probabilities; nothing is
generated.

- Prompt: ChatML system turn (fixed framing + optional `system`) → user `<state>…</state>` + question with
  lettered options → assistant with an empty, closed `<think></think>`. Thinking cannot be enabled.
- Readout: FP16 logits at the answer position, full-vocab log-softmax in Double, renormalised over the label
  tokens (`Yes`/`No`, `A`…). Labels must be single tokens or the request fails with a 500 naming the label.
- ChatML models only (Qwen family); other dialects get a 400.
- Shared prefix: `system + state` is prefilled once, then per question the runner restores a checkpoint (KV
  cursor rewind + copy of the Gated-DeltaNet state) and prefills only the question suffix. Falls back to full
  per-question prefills if prefix/suffix tokenise differently at the join or the runner cannot checkpoint.
- Code: `Sources/TUFFServer/Core/SystemOneModels.swift` (types, validator, prompt, maths),
  `ServerInference.swift` (`scoreSystemOne`), `Sources/TUFFEngine/Runtime/Generation/LabelLogProbs.swift`
  (engine readout), `RunnerCheckpoint` in `ModelForwardRunner.swift` / `RealForwardRunner.swift`.

### Server flags added

| Flag | Default | Purpose |
|---|---|---|
| `--systemone-system-prompt <text>` | none | Used when a request sends no `system` |
| `--systemone-prefix-reuse on\|off\|auto` | on | Shared-prefix checkpoint path; `auto` reuses only for multi-question requests |
| `--systemone-temperature <T>` | 1 | Temperature on label log-probabilities (fitted: Apodex 1.484, kat 2.444, ornith 2.218) |
| `--prefill-expert-staging on\|off` | on | Burst-read a layer's routed experts into a staging arena during prefill |
| `--prefill-mbatch-int4 on\|off` | off | Batched small-M INT4 kernel for wide prefill projections (M ≤ 64): ~8% faster, probabilities shift up to ~0.06 |
| `--prefill-fetch-overlap on\|off` | on | With staging, run the shared expert and already-staged tiles on the GPU while later sub-bursts are read |

`--expert-cache-slots` accepts 4–128 (help text still says 8–32).

### Diagnostics

`TUFF_PREFILL_PROFILE=1` prints, per `prefillChunked` call, wall/GPU time at each sync point (GDN layers,
full-attention layers, shared expert, routed-expert fetch and tiles, tail, LM head), expert-cache misses,
command-buffer count with GPU busy vs span, and a per-`pread` latency histogram. With it,
`TUFF_EXPERT_USAGE_FILE=<path>` dumps staged reads per layer and expert. `TUFF_SYSTEMONE_LABEL_MASS=1` logs the
labels' share of the vocabulary distribution. `TUFF_PREFILL_ATTENTION_SIMDGROUP=1` switches full-attention
prefill to a `simdgroup_matrix` kernel (1.6–4.1× per call on M2; fails the 1e-3 end-to-end parity rule, off).
`TUFF_BENCH_ATTENTION=1` runs its microbenchmark test.

## Build and test

Builds and model tests run on the Mac mini (`rifkyaziz@rifkyazizs-Mac-mini.local`, M2 Pro, 16 GB) — the
MacBook Air has no disk for models. Sync with `rsync -a --delete --exclude .build --exclude scratch` to
`~/Developer/TUFF`, then:

- Build: `swift build -c release --product TUFFServer`
- Unit tests: `Scripts/test.sh --filter "SystemOne|HTTPServerTests|OpenAIValidation|ServerArguments|Prefill"`
  (serial by design — engine tests share Metal state)
- Real-model checks: start `.build/release/TUFFServer --model ~/models/<model> --port 8089`, then
  `python3 Scripts/systemone_smoke.py --port 8089` (42 cases). `--dump a.json` / `--compare a.json b.json`
  compare two runs.
- Accuracy/calibration eval: `Scripts/systemone_eval.py` (`fetch`, `run`, `report`), 9 sets × 300 items
  (100 calibration / 200 test), results and method in `docs/systemone-eval.md`. Data and runs live in
  `scratch/systemone_eval/` on the Mac mini.

Models on the Mac mini, all qwen36-family 35B-A3B, text-only (no vision pack): `~/models/apodex-1.1-mini-4bit`,
`kat-coder-4bit`, `ornith-1.5_35B_A3B_4Bit`. The server reports them as `qwen3.6-35b-a3b`.

## Status

- Smoke suite: 42/42 on Apodex. Answers are deterministic and option-order stable; multi-question answers
  match single-question ones.
- Latency on the Mac mini (Apodex, warm OS file cache): 1 question ≈ 1.55 s, 5 questions on one state ≈ 3.9–4.5 s.
  First request after start ≈ 9 s (cold SSD).
- Profile of a warm prefill: staged expert copies from the OS file cache ≈ 40% (now overlapped with GPU work),
  GPU ≈ 45%, sync/CPU ≈ 7%. GDN layers cost ~110–160 ms of GPU time per prefill call regardless of length.
- Tried and dropped: layer-major multi-question prefill (65% fewer expert reads, but no latency gain once reads
  came from the file cache; extra per-layer syncs cancelled it). Expert cache above ~48 slots is slower.
- Tried and dropped: zero-copy experts (mmap + Metal `bytesNoCopy`), per-expert, per-layer and a `mincore` hybrid.
  Only faster when the exact same experts repeat; with a different expert set per request it is 1.6–14 s vs
  ~1 s for the copy path, because Metal wiring faults pages in serially and pins memory.

## Known issues

- Prefix reuse vs full prefill differs by up to ~0.01 in probability on single-question requests (FP16 chunk
  boundaries move); no answer changes. Restore itself is exact.
- The `system` field reaches the model but a one-line policy does not override strong defaults.
- Probabilities are overconfident without `--systemone-temperature` (confident errors 4–9% raw, ~2% with the
  fitted T). Accuracy: 0.735–0.748 mean over test groups; classic sets 0.85–0.96, Kev `hard-v1` choice 0.49–0.57.
- The validator rejects `null` option descriptions, which Jev/Kev accept.
- Expert cache above ~48 slots slows requests on 16 GB (memory pressure); keep the default 16.
- Keep the Mac mini awake during benchmarks (`sudo pmset -a sleep 0`); it dropped off the network once.

## Next

See the ranked list in `docs/systemone-eval.md` §8. Top items: serve with the fitted temperature; decide on
`auto` + 256-token chunks (16–45% faster single questions, probabilities move ≤ 0.03); layer-major prefill
across chunks (a 2,499-token prefix stages 206 GB of experts today).

1. INT4 projection kernels run 6–9× below the M2 Pro's limits at small M (`TUFF_BENCH_INT4=1` benchmark); the
   batched kernel reaches 2–4× but half-precision weight staging costs answer parity, so it ships off.
2. `Scripts/systemone_smoke.py` also runs 14 reasoning-trap cases (car-wash, step-ordering, retraction); Apodex
   passes 11/14 (fails car-wash at 50 m with 0.94 confidence, oil change, and one ambiguous control).
