# /v1/systemone: accuracy, calibration and speed

Measured on the Mac mini (M2 Pro, 16-core GPU, 16 GB, APPLE SSD AP0512Z, macOS 26.5.2) with the three
stock qwen36-family 35B-A3B models in `~/models` (Apodex 1.1 mini, kat-coder, ornith 1.5; all 4-bit, thinking
off, no weight changes). Every number below is from a command whose output is quoted in this file or in
`scratch/systemone_eval/` on the Mac mini.

## Summary

- **Accuracy is already high on classic classification, weak on reasoning.** Test-split accuracy, mean over
  11 question groups: Apodex 0.748, kat 0.735, ornith 0.741. Classic sets land at 0.85–0.96 (SST-2, TREC,
  MNLI, BoolQ, AG News, 26 Banking77 intents); Kev's reasoning suite (`hard-v1`) choice questions at
  0.49–0.57.
- **Kept: one temperature per model** (`--systemone-temperature`, new, default 1). It never changes an answer.
  Paired bootstrap on 1,913 test questions: NLL −0.069 (Apodex, T = 1.48), −0.342 (kat, T = 2.44),
  −0.264 (ornith, T = 2.22), all with 95% intervals clear of zero; confident errors (p ≥ 0.9 and wrong)
  fall from 4.0/8.9/8.2% to 2.2/1.8/1.8%.
- **Rejected** (no significant accuracy gain, or a loss, or a latency cost for a gain temperature gives for
  free): contextual calibration (−2.0 points), batch calibration, PriDe, option-permutation averaging (2–4×
  latency), 4-shot prompts in the shared prefix (2.7× latency, −0.4 points), a framing sentence, and
  summing label surface forms (the canonical label tokens already hold 98.7–99.98% of the probability).
- **Where prefill time goes:** routed-expert I/O. A default-config request stages 12,210 expert reads
  (21.6 GB) on average; the page cache holds 59% of the 18 GB model files, and the SSD delivers the rest at
  1.35 GB/s average against the 6.85 GB/s it can do. For long prompts, 128-token chunks re-read every
  layer's experts per chunk: 206 GB for one 2,499-token Kev prefix.
- **Faster, opt-in:** `--prefill-chunk-tokens 256 --systemone-prefix-reuse auto` (the `auto` mode is new)
  cuts single-question latency 16–45% in an interleaved A/B with no measurable accuracy change (Δacc
  +0.002 [−0.004, +0.007]). A new `simdgroup_matrix` causal attention kernel
  (`TUFF_PREFILL_ATTENTION_SIMDGROUP=1`) is 1.6–4.1× faster per call on the M2 and matches the old kernel
  within one FP16 ulp, but fails the 1e-3 end-to-end parity rule (max |Δp| 0.12) because MoE routing
  amplifies ulp-level changes, so it stays off.
- **Next** (ranked below): layer-major prefill across chunks for long prompts, keeping hot experts resident
  and read in place, and tiled `simdgroup_matrix` kernels for routed experts, INT4 projections and GDN.

## 1. Eval harness

`Scripts/systemone_eval.py` (standard library only; runs on the Mac mini's Python 3.9):

```
python3 Scripts/systemone_eval.py fetch                  # download + freeze items (cached in scratch/)
python3 Scripts/systemone_eval.py run --port 8089 --tag apodex --variant base
python3 Scripts/systemone_eval.py report --tags apodex kat ornith --json out.json
```

- **Sets** (300 items each, seed 20260927, first 100 calibration / last 200 test; only the test split is
  reported; calibration items are used only to fit transforms):

  | set | type | source (Hub commit) | notes |
  |---|---|---|---|
  | boolq | noul | google/boolq validation (35b264d0) | passage as state |
  | sst2 | noul | stanfordnlp/sst2 validation (8d51e7e4) | "Is the sentiment … positive?" |
  | agnews | choice ×4 | fancyzhx/ag_news test (eb185aad) | |
  | trec | choice ×6 | SetFit/TREC-QC test (0a34640b), coarse labels | |
  | mnli | choice ×3 | nyu-mll/glue mnli validation_matched (bcdcba79) | premise + hypothesis as state |
  | banking77 | choice ×26 | mteb/banking77 test (18072d26) | 26 seeded intents |
  | sst5 | score ×5 | SetFit/sst5 test (e51bdcd8) | |
  | amazon | score ×5 | SetFit/amazon_reviews_multi_en test (ec73b665) | stars |
  | kev_hard | mixed | jaredpalmer/kev@5920c5fe `evals/hard-v1/test.jsonl` (sha256 246ee922…) | Apache-2.0, Kev's own synthetic templates |

  Kev rows keep all their questions (1–2) in one request, like real System One traffic, and are scored as
  three groups: `kev_noul` (66 test questions), `kev_choice` (235), `kev_score` (12, too few to read much
  into). Kev leaves some option descriptions `null`; the endpoint requires a string, so the harness sends
  the option key in words. That is a Jev-compatibility gap in the validator.
- **Metrics:** accuracy, macro-F1 (over label index; for Kev choice that is option position), ECE (15 equal
  bins on top-label confidence), multi-class Brier (sum over labels), NLL, confident-error rate (top
  probability ≥ 0.9 and wrong, as a share of all questions), request latency p50/p95 on the test split.
  Means are unweighted over groups.
- **Request variants:** `base`, `cf` (content-free probes), `perm` (cyclic option shifts in the same
  request), `fewshot<k>`, `sys-<framing>`, and `--save-as` for server-flag variants.
- **Eval-side transforms:** `raw`, `temp` (per set), `gtemp` (one T per model), `cc`, `bc`, `bc+temp`,
  `permavg`, `pride`. The maths was checked on synthetic data: a calibrated generator gives ECE 0.009, a
  T = 0.5 sharpening is recovered as T = 2.005, and PriDe recovers an injected [0.5, 0.3, 0.2] prior exactly.

## 2. Baseline: current prompt, three models (test split)

```
| model | request | transform | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apodex | base | raw | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |
| kat | base | raw | 0.735 | 0.739 | 0.175 | 0.395 | 1.068 | 0.089 | 4948 | 10470 |
| ornith | base | raw | 0.741 | 0.735 | 0.171 | 0.405 | 1.005 | 0.082 | 4475 | 10166 |
```

Per-set accuracy (full per-set tables for every model × transform are in the appendix):

| set | Apodex | kat | ornith |
|---|---:|---:|---:|
| boolq | 0.875 | 0.875 | 0.900 |
| sst2 | 0.945 | 0.960 | 0.955 |
| agnews | 0.880 | 0.860 | 0.845 |
| trec | 0.935 | 0.895 | 0.900 |
| mnli | 0.915 | 0.860 | 0.895 |
| banking77 (26) | 0.855 | 0.875 | 0.880 |
| sst5 | 0.500 | 0.470 | 0.515 |
| amazon | 0.580 | 0.530 | 0.510 |
| kev_noul | 0.848 | 0.848 | 0.848 |
| kev_choice | 0.566 | 0.574 | 0.485 |
| kev_score (n=12) | 0.333 | 0.333 | 0.417 |

With 200 items a 95% interval on accuracy is about ±4–5 points; differences between models on one set are
mostly inside that. The classic sets are old and public, so contamination is possible; Kev's `hard-v1`
(2026 templates, test-only template) is the more honest reasoning check.

## 3. No-training accuracy and calibration techniques

### Research

| technique | idea | source |
|---|---|---|
| Temperature scaling | divide logits by one fitted T | Guo et al. 2017, <https://arxiv.org/abs/1706.04599> |
| Contextual calibration | divide by p(label \| "N/A") | Zhao et al. 2021, <https://arxiv.org/abs/2102.09690> |
| Domain-context calibration | content-free input from in-domain random words | Fei et al. 2023, <https://arxiv.org/abs/2305.19148> |
| Batch calibration | divide by mean p(label) over unlabelled inputs | Zhou et al. 2023, <https://arxiv.org/abs/2309.17249> |
| PriDe | estimate the option-ID prior from cyclic permutations of a few items, divide it out | Zheng et al. 2024, <https://arxiv.org/abs/2309.03882> |
| Permutation averaging | average over option orders | same; Tang et al. 2023, <https://arxiv.org/abs/2310.07712> |
| CalibraEval | label-free, order-preserving debiasing for LLM judges | <https://arxiv.org/abs/2410.15393> |
| Surface-form competition / verbalizers | mass spread over label spellings | Holtzman et al. 2021, <https://arxiv.org/abs/2104.08315>; "Semantic Softmax", <https://arxiv.org/abs/2605.09739> (2026) |
| UniBias | mask biased FFN vectors / attention heads at inference | Zhou et al. 2024, <https://arxiv.org/abs/2405.20612> |
| Calibration without ground truth | Bregman projection towards a better-calibrated reference model | <https://arxiv.org/abs/2601.19862> (2026) |
| Few-shot demonstrations | labelled examples in the prompt | Brown et al. 2020; order sensitivity: Lu et al. 2022, <https://arxiv.org/abs/2104.08786> |

Not tried: UniBias (needs per-head masking inside the attention kernels and a search per task),
CalibraEval (built for pairwise judges), calibration without ground truth (needs a second model).

### Results (Apodex unless stated; test split)

Eval-side transforms on the same runs (`report --transforms …`):

```
| model | request | transform | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apodex | base | raw | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |
| apodex | base | temp | 0.748 | 0.750 | 0.091 | 0.332 | 0.673 | 0.021 | 4156 | 9680 |
| apodex | base | gtemp | 0.748 | 0.750 | 0.105 | 0.339 | 0.688 | 0.022 | 4156 | 9680 |
| apodex | base | cc | 0.728 | 0.727 | 0.136 | 0.379 | 0.842 | 0.053 | 4156 | 9680 |
| apodex | base | bc | 0.745 | 0.748 | 0.114 | 0.355 | 0.810 | 0.046 | 4156 | 9680 |
| apodex | base | bc+temp | 0.745 | 0.748 | 0.082 | 0.335 | 0.685 | 0.023 | 4156 | 9680 |
| apodex | base | pride | 0.749 | 0.751 | 0.122 | 0.348 | 0.763 | 0.040 | 4156 | 9680 |
| kat | base | raw | 0.735 | 0.739 | 0.175 | 0.395 | 1.068 | 0.089 | 4948 | 10470 |
| kat | base | temp | 0.735 | 0.739 | 0.096 | 0.344 | 0.716 | 0.016 | 4948 | 10470 |
| kat | base | gtemp | 0.735 | 0.739 | 0.109 | 0.345 | 0.699 | 0.018 | 4948 | 10470 |
| kat | base | bc | 0.742 | 0.744 | 0.176 | 0.410 | 1.157 | 0.083 | 4948 | 10470 |
| kat | base | bc+temp | 0.742 | 0.744 | 0.097 | 0.364 | 0.796 | 0.028 | 4948 | 10470 |
| ornith | base | raw | 0.741 | 0.735 | 0.171 | 0.405 | 1.005 | 0.082 | 4475 | 10166 |
| ornith | base | temp | 0.741 | 0.735 | 0.123 | 0.396 | 1.982 | 0.070 | 4475 | 10166 |
| ornith | base | gtemp | 0.741 | 0.735 | 0.104 | 0.357 | 0.718 | 0.018 | 4475 | 10166 |
| ornith | base | bc | 0.746 | 0.752 | 0.157 | 0.379 | 0.970 | 0.071 | 4475 | 10166 |
| ornith | base | bc+temp | 0.746 | 0.752 | 0.115 | 0.374 | 1.503 | 0.049 | 4475 | 10166 |
```

Request variants (each a separate run on the test split):

```
| apodex | fewshot4 | raw | 0.743 | 0.748 | 0.138 | 0.361 | 0.789 | 0.041 | 11378 | 19097 |
| apodex | sys-evidence | raw | 0.754 | 0.756 | 0.120 | 0.348 | 0.777 | 0.042 | 5408 | 11306 |
| apodex | fast | raw | 0.750 | 0.750 | 0.129 | 0.350 | 0.773 | 0.040 | 4904 | 12800 |
```

Paired bootstrap (2,000 resamples of test questions shared by both runs;
`scratch/systemone_eval/probes/paired.py`):

```
apodex  base global T=1.48       n=1913  d_acc +0.0000 [+0.0000, +0.0000]  d_nll -0.0692 [-0.0922, -0.0488]
kat     base global T=2.44       n=1913  d_acc +0.0000 [+0.0000, +0.0000]  d_nll -0.3418 [-0.4040, -0.2844]
ornith  base global T=2.22       n=1913  d_acc +0.0000 [+0.0000, +0.0000]  d_nll -0.2644 [-0.3155, -0.2172]
apodex  fast                     n=1913  d_acc +0.0016 [-0.0042, +0.0073]  d_nll -0.0003 [-0.0052, +0.0045]
apodex  fewshot4                 n=1913  d_acc -0.0042 [-0.0183, +0.0099]  d_nll +0.0205 [-0.0131, +0.0555]
apodex  sys-evidence             n=1913  d_acc +0.0010 [-0.0073, +0.0099]  d_nll -0.0007 [-0.0124, +0.0104]
apodex  perm                     n= 400  d_acc -0.0025 [-0.0150, +0.0100]  d_nll -0.1063 [-0.1630, -0.0579]
```

Permutation runs covered AG News and TREC only (the rest were stopped once the cost was clear):

```
#### apodex / base / raw
| agnews | 200 | 0.880 | 0.878 | 0.106 | 0.226 | 0.791 | 0.095 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.050 | 0.124 | 0.293 | 0.030 | 4023 | 4451 |
#### apodex / base / pride
| agnews | 200 | 0.880 | 0.878 | 0.109 | 0.235 | 0.796 | 0.105 | 4990 | 5448 |
| trec | 200 | 0.925 | 0.914 | 0.030 | 0.119 | 0.281 | 0.025 | 4023 | 4451 |
#### apodex / perm / permavg
| agnews | 200 | 0.870 | 0.867 | 0.101 | 0.225 | 0.648 | 0.095 | 9252 | 9862 |
| trec | 200 | 0.940 | 0.950 | 0.050 | 0.107 | 0.224 | 0.020 | 16181 | 17005 |
```

Label mass with `TUFF_SYSTEMONE_LABEL_MASS=1`, and the effect of summing surface forms (" Yes", "yes",
" A" …) over 81 questions:

```
labelvar vs base: questions 81 max|dp| 0.0005 mean|dp| 0.00001 flips 0
n 83 canonical mass min 0.9872 median 0.9980; added by variants max 0.0083 median 0.00000
```

### Kept and rejected

| technique | verdict | why |
|---|---|---|
| One temperature per model | **kept**, `--systemone-temperature <T>` | NLL and confident errors fall on all three models with intervals clear of zero; answers never change. Per-set T is not usable: it needs task labels at serve time and blows up on small groups (ornith `kev_score`, 3 calibration items, T = 0.08, NLL 15.5). Fitted T: Apodex 1.484, kat 2.444, ornith 2.218. |
| Contextual calibration | rejected | −2.0 points mean; SST-2 0.945 → 0.825. The instruction-tuned model's answer to "N/A" is itself skewed and dividing by it over-corrects. |
| Batch calibration | rejected | accuracy −0.3 to +0.7 points (noise), NLL worse on its own; with temperature no better than temperature alone. |
| PriDe | rejected | no change (0.748 → 0.749); the model shows little option-ID bias. |
| Permutation averaging | rejected | Δacc −0.3 [−1.5, +1.0] points; NLL gain real but temperature gives a comparable gain for free, and latency is 2–4× (AG News 5.0 → 9.3 s, TREC 4.0 → 16.2 s). |
| 4-shot in the shared prefix | rejected as a default | Δacc −0.4 [−1.8, +1.0]; helps SST-2 (+2.0), MNLI (+2.5), BoolQ (+1.0), hurts Kev choice (−5.1) and Amazon (−4.0); p50 4.2 → 11.4 s because nothing caches a prefix across requests. Clients can still send examples in `system`. |
| Framing sentence ("evidence") | rejected | Δacc +0.1 [−0.7, +1.0]. |
| Label surface forms | rejected, flag removed | the canonical tokens already hold ≥ 98.7% of the mass; max \|Δp\| 0.0005. The mass log (`TUFF_SYSTEMONE_LABEL_MASS=1`) stays as a diagnostic. |

## 4. Where prefill time goes

### Instrumentation added (all opt-in)

- `TUFF_PREFILL_PROFILE=1|2` (existing) now also prints, per prefill call, the command-buffer count, GPU busy
  time against the first-start-to-last-end span (idle = encode, commit and CPU waits between buffers), and a
  per-`pread` latency histogram with bytes read.
- `TUFF_EXPERT_USAGE_FILE=<path>` (with `TUFF_PREFILL_PROFILE`) writes cumulative staged reads per layer and
  expert.
- `TUFF_SYSTEMONE_LABEL_MASS=1` logs the label tokens' share of the full-vocabulary distribution.
- Metal System Trace / `xctrace` was not available (the Mac mini has only the command-line tools, no Xcode),
  so GPU times are command-buffer timestamps and I/O is `iostat` plus the engine's counters.

### One request, split (Apodex, default config, `server-1.log`)

A single-question SST-2 request is two prefill calls (48-token shared prefix, 26-token question). First
request after start, then warm:

```
[prefill profile] 48 tok in 1 chunk(s):   6214.9 ms
  GDN layers  attn+router     1137.5 ms  18.3%  gpu    206.5 ms
  full layers attn+router      218.2 ms   3.5%  gpu     54.5 ms
  shared expert                  0.0 ms   0.0%  gpu     62.8 ms
  routed experts (wall)       4637.3 ms  74.6%
    expert fetch (SSD/cache)   293.9 ms   4.7%  4198 missed / 4198 used in 540 tiles
    tile GPU wait              134.0 ms   2.2%  gpu    256.9 ms
  LM head                      195.7 ms   3.1%
  command buffers           661, GPU busy 588.1 ms of 6174.5 ms span (idle between buffers 5586.4 ms)
  expert preads            4198 reads, 7428.2 MB, summed 2159.0 ms; p50 0.39 p90 0.87 p99 1.65 max 6.80 ms
...
[prefill profile] 48 tok in 1 chunk(s):    537.8 ms   (warm)
  GDN gpu 150.0 | full gpu 35.6 | fetch 258.3 ms, 4198 experts | tile GPU 169.1 | CBs 661, busy 406.9 of 537.8, idle 130.2
[prefill profile] 26 tok in 1 chunk(s):    440.2 ms   (warm)
  GDN gpu 135.8 | full gpu 34.9 | fetch 176.6 ms, 2767 experts | tile GPU 93.5  | CBs 483, busy 292.3 of 431.5, idle 147.2
```

Per-stage GPU time of the GDN layers (`TUFF_PREFILL_PROFILE=2`, `server-2.log`):

```
== 48 tok in 1 chunk(s):    596.3 ms
  GDN stage in_proj (qkv, z, a, b)  gpu     92.7 ms
  GDN stage conv + tail + qk norm   gpu      1.3 ms
  GDN stage delta step              gpu     30.5 ms
  GDN stage out_proj                gpu     24.2 ms
  full layers attn+router       39.0 ms   6.5%  gpu     36.9 ms
    tile GPU wait               55.8 ms   9.4%  gpu    176.4 ms
== 2499 tok in 20 chunk(s):  42248.7 ms
  GDN stage in_proj (qkv, z, a, b)  gpu   3399.5 ms
  GDN stage conv + tail + qk norm   gpu     71.6 ms
  GDN stage delta step              gpu   2000.3 ms
  GDN stage out_proj                gpu   1072.9 ms
  full layers attn+router     7350.8 ms  17.4%  gpu   7221.5 ms
    tile GPU wait             2265.7 ms   5.4%  gpu  10669.7 ms
  command buffers           20923, GPU busy 28775.3 ms of 42251.7 ms span (idle between buffers 13476.4 ms)
```

What this says:

- **Short prompts (the common case, 50–500 tokens):** warm, a prefill call is ~40–60% expert fetch and
  ~60–75% GPU with partial overlap. GPU work is small in absolute terms (0.3–0.75 s per call) but far below
  the hardware: the GDN input projection reads ~0.43 GB of INT4 weights and scales (estimated from the
  layer shapes) in 92.7 ms, ≈5 GB/s of a 200 GB/s part, for 48 tokens. A call issues 480–820 command buffers.
- **The shared-prefix split doubles expert reads for one-question requests**: the 48-token prefix and the
  26-token question each stage a full per-layer union (4,198 + 2,767 experts).
- **Long prompts:** 2,499 tokens in 20 chunks read 116,507 experts (206 GB) because each chunk re-stages
  every layer's union; expert fetch is 23 s of 40–50 s. On the GPU side, full attention (7.2 s) and routed
  expert tiles (10.7 s) dominate; the GDN recurrence is 2.0 s.
- **Repeat variance is large:** the same 2,499-token request took 40.3 s and 50.3 s back to back, with GDN
  GPU time 6.8 s vs 11.3 s. Short-set wall times shift ±30% between runs hours apart (page-cache contents).
  Only interleaved A/B runs are used for latency claims below.

### I/O (measure first)

Raw SSD, expert-sized random reads with `F_NOCACHE` on the least-resident layer files
(`scratch/systemone_eval/probes/ssdread.c`):

```
chunk=1769472 qd=1 n=120 nocache=1: 212.3 MB in 0.068 s = 3135 MB/s; lat p50 0.59 p90 0.94 p99 1.03 max 1.53 ms
chunk=1769472 qd=2 n=120 nocache=1: 212.3 MB in 0.043 s = 4964 MB/s; lat p50 0.76 p90 1.24 p99 1.45 max 1.52 ms
chunk=1769472 qd=4 n=120 nocache=1: 212.3 MB in 0.040 s = 5327 MB/s; lat p50 1.31 p90 2.04 p99 2.37 max 2.44 ms
chunk=1769472 qd=8 n=120 nocache=1: 212.3 MB in 0.034 s = 6315 MB/s; lat p50 2.29 p90 3.26 p99 3.92 max 3.95 ms
chunk=1769472 qd=16 n=120 nocache=1: 212.3 MB in 0.031 s = 6854 MB/s; lat p50 4.33 p90 5.45 p99 5.74 max 5.78 ms
chunk=1769472 qd=32 n=120 nocache=1: 212.3 MB in 0.033 s = 6525 MB/s; lat p50 9.48 p90 10.89 p99 11.36 max 11.56 ms
chunk=1769472 qd=64 n=120 nocache=1: 212.3 MB in 0.042 s = 5069 MB/s; lat p50 17.96 p90 25.41 p99 26.67 max 26.88 ms
chunk=8388608 qd=8 n=50 nocache=1: 419.4 MB in 0.076 s = 5494 MB/s; lat p50 11.91 p90 13.33 p99 13.69 max 13.69 ms
chunk=1769472 qd=8 n=256 nocache=0: 453.0 MB in 0.008 s = 53344 MB/s; lat p50 0.21 p90 0.36 p99 0.86 max 0.88 ms
```

(The files were 0–68% resident, so the low-QD rows may include some cache hits; the QD 16–32 plateau is the
drive.) Page-cache residency of the model (`resident.c`, `mincore`): `TOTAL resident 10.71 GB of 18.12 GB
(59.1%)`.

During the profiled requests (`iostat -d -w 1 disk0` against the engine's own byte count):

```
iostat-1.txt seconds 164 total read 224.3 GB peak 2823 MB/s mean while >50MB/s 1376 MB/s busy secs 163
server-1.log expert bytes requested 787.6 GB
```

So 28% of expert bytes came from the SSD and 72% from the page cache; the SSD ran at ~20% of its measured
ceiling on average. The staging burst uses 32 `pread`s in flight, which is on the plateau. Per-`pread`
latency inside the engine is 0.4–0.5 ms p50 when warm and 2–5 ms p50 when reads miss the cache (see the
`expert preads` lines above). Overlap: the existing `--prefill-fetch-overlap` runs shared-expert and staged
tiles while later sub-bursts read; warm, GPU idle between buffers is 130–150 ms of a 440–540 ms call, cold
90% of the span.

Expert concentration over 200 test requests (8 classic sets × 25, `TUFF_EXPERT_USAGE_FILE`):

```
layers 40 staged expert reads 2442052 per request 12210.26
top-32 experts/layer cover 26.3% of reads (min layer 18.4%) -> pinned bytes 2.3 GB
top-64 experts/layer cover 47.8% of reads (min layer 35.5%) -> pinned bytes 4.5 GB
top-128 experts/layer cover 78.4% of reads (min layer 64.9%) -> pinned bytes 9.1 GB
top-192 experts/layer cover 94.9% of reads (min layer 86.3%) -> pinned bytes 13.6 GB
```

Usage is only mildly skewed: a 9 GB pinned set covers 78% of reads, about what the 10.7 GB page cache already
holds.

## 5. Speed changes made (opt-in, measured)

### Prefix reuse `auto` + 256-token chunks

`--systemone-prefix-reuse auto` (new) reuses the shared prefix only when a request has more than one question;
a single question is prefilled in one call. Interleaved A/B on 15 test items per set (median ms; Kev column
is the sum over 15 items; probabilities compared with the first run):

```
ab-reuse-on      p50 ms: sst2  1275 agnews  3076 boolq  4895 bankin  8498 kev_ha  8244 | kev total 306s | max|dp| vs reuse-on 0.0000, argmax flips 0
ab-reuse-off     p50 ms: sst2  1254 agnews  3641 boolq  4047 bankin  8954 kev_ha  7462 | kev total 541s | max|dp| vs reuse-on 0.0319, argmax flips 0
ab-reuse-on2     p50 ms: sst2  1287 agnews  3127 boolq  4820 bankin  8543 kev_ha  8364 | kev total 306s | max|dp| vs reuse-on 0.0000, argmax flips 0
ab-chunk256      p50 ms: sst2  1288 agnews  3069 boolq  4792 bankin  7651 kev_ha  7460 | kev total 306s | max|dp| vs reuse-on 0.0164, argmax flips 0
ab-chunk256-off  p50 ms: sst2  1265 agnews  2574 boolq  2737 bankin  5973 kev_ha  6614 | kev total 510s | max|dp| vs reuse-on 0.0319, argmax flips 0
```

and the combined mode in a second interleaved run:

```
ab2-attn-off   p50 ms: sst2  1266 agnews  4301 boolq  5049 bankin  8727 kev_ha  8356 | kev sum 297s
ab2-fast       p50 ms: sst2  1247 agnews  3066 boolq  2764 bankin  5768 kev_ha  6575 | kev sum 277s | vs ab2-attn-off: max|dp| 0.02424 flips 0
```

Single-question requests get 16–45% faster (BoolQ 4.9 → 2.7 s, Banking77 8.5 → 6.0 s); multi-question Kev
rows keep reuse. Probabilities move by up to 0.03 (FP16 chunk boundaries, the same class of difference the
prefix-reuse path already has); on the full test split accuracy is unchanged (paired Δacc +0.002
[−0.004, +0.007]). The full-split run's own latency (p50 4.9 s) was taken a day after the baseline and is
not comparable; the interleaved numbers above are. It stays opt-in because it moves probabilities by more
than the 1e-3 parity rule.

Chunks above 256 are the obvious extension; `PrefillRuntimeConfig.maxChunkTokens` documents why sizes past
280 need a KV-ring layout test first.

### `simdgroup_matrix` causal attention (`TUFF_PREFILL_ATTENTION_SIMDGROUP=1`)

`attention_prefill_causal_simdgroup` (prefill.metal): one simdgroup per (query row, KV head); the 8 query
heads sharing a KV head form the rows of each 8×8 tile, so a tile has one causal limit; keys advance 8 at a
time with an online softmax; FP32 scores and accumulators. Used only for 256-wide heads with 8:1 GQA and
full visibility (the runner passes full layers a window spanning every key); everything else keeps the tiled
kernel. Works on M1/M2 (Apple7+), no tensor cores needed.

Kernel parity (`PrefillAttentionTests.simdgroupCausalMatchesReferenceAndTiled`, M2 Pro):

```
simdgroup attention start=0 chunk=13: max|d| vs tiled 0.00012207031, vs reference 0.00021520257
simdgroup attention start=300 chunk=128: max|d| vs tiled 3.0517578e-05, vs reference 2.7883798e-05
simdgroup attention start=1021 chunk=37: max|d| vs tiled 1.5258789e-05, vs reference 1.4565885e-05
```

and with peaked scores (queries ×24; M3):

```
simdgroup attention start=0 chunk=74 runnerWindow=true sharp=true: max|d| vs tiled 0.00024414062, vs reference 0.0024516848
simdgroup attention start=210 chunk=256 runnerWindow=true sharp=true: max|d| vs tiled 0.00024414062, vs reference 0.0034310892
simdgroup attention start=2371 chunk=128 runnerWindow=true sharp=true: max|d| vs tiled 0.00024414062, vs reference 0.002653893
```

Speed (`TUFF_BENCH_ATTENTION=1`, GPU time per dispatch, median of 20):

```
prefill attention microbenchmark, device: Apple M2 Pro
  start chunk   tiled ms  simdgroup ms  speedup
      0    74      1.035         0.643     1.61x
      0   192      2.064         0.674     3.06x
    210   256      9.472         2.340     4.05x
      0   256      3.626         1.011     3.59x
   1024   128     15.442         4.364     3.54x
   2371   128     36.254         9.620     3.77x
```

In the model, full-layer GPU time for the 2,499-token prefix went from 4.6–10.1 s to 2.8–3.5 s. End to end
(interleaved, 15 items per set) it does **not** pass parity:

```
probabilities: max|dp| 1.21e-01 mean 3.04e-03 nonzero 598/598 flips 0
kev_hard   n=21 max 0.1205 p90 0.0431 median 0.01061  >1e-3: 19
reference: chunk128 vs chunk256 max 0.0164 median 0.00000 >1e-3: 9/81
```

The kernel agrees with the old one to one FP16 ulp, but the difference changes top-8 expert choices
(116,507 vs 116,500 experts staged for the same prompt), and routing is discontinuous. It stays off by
default; wall-clock differences on short prompts were inside run-to-run noise.

## 6. Speed research

### a. Compute

| idea | evidence | measured baseline | expected gain | effort | notes |
|---|---|---|---|---|---|
| Layer-major prefill across chunks | read each layer's experts once per request, not once per chunk; "LLM in a Flash" windowing reuses loaded weights across tokens, <https://arxiv.org/abs/2312.11514>; Klotski pipelines across batches to amortise loads, <https://arxiv.org/abs/2502.06888> | 2,499-token prefix: 206 GB staged, 23 s fetch | fetch → ≤ 18.1 GB (at most one full union per layer), ~20 s off a 40 s request; nothing for ≤ 256-token prompts | L | the chunk executor is ~1,600 lines; attention is causal within a layer, GDN state carries chunk to chunk in the same layer, so the math is the same per chunk |
| Batch question suffixes over one forked state | one suffix pass for all questions: GDN state copied per branch, cascade/shared-prefix attention for the 10 full layers (Hydragen, <https://arxiv.org/abs/2402.05099>; cascade inference in FlashInfer, <https://arxiv.org/abs/2501.01005>) | 5 questions ≈ 3.9–4.5 s (PROJECT_CONTEXT); each suffix restages its union | one expert pass instead of k; tried before as layer-major multi-question and dropped when reads came from the page cache | M | worth re-testing with `auto` and chunk 256, since 28% of bytes still come from the SSD |
| Chunkwise-parallel GDN prefill (WY/UT form) | Yang et al. "Gated Delta Networks", <https://arxiv.org/abs/2412.06464>; MLX `gated_delta_update.h` (8-token chunks, Neumann-series inverse on 8×8 `simdgroup_matrix`); uzu `gdn/chunked/` (Gram, block-diagonal inverse, forward substitution; notes its matrix-unit path did not beat `simdgroup_matrix`) | delta step 30.5 ms (48 tok), 2.0 s (2,499 tok) | 2–4× on the recurrence; 20–60 ms on short prompts, ~1.5 s on long | M | llama.cpp's Metal `gated_delta_net.metal` is sequential like TUFF's |
| Fused projections | TUFF already fuses q/k/v/z/a/b into one GDN in_proj dispatch; the cost is the kernel, not the count | in_proj 92.7 ms for 48 tokens (~5 GB/s effective) | see §6d INT4 GEMM | — | |

### b. Storage and I/O

| idea | evidence | measured baseline | expected gain | effort | format change |
|---|---|---|---|---|---|
| Keep hot experts resident and read them in place | PowerInfer hot/cold split, <https://arxiv.org/abs/2312.12456>; PowerInfer-2, <https://arxiv.org/abs/2406.06282>; MoE-Infinity activation-aware caching, <https://arxiv.org/abs/2401.14361>; Apple `MTLResidencySet` (macOS 15), <https://developer.apple.com/documentation/metal/mtlresidencyset> | top-128/layer = 9.1 GB covers 78% of reads; page cache 10.7 GB | on 16 GB only the memcpy out of the page cache is saved (copies were ~40% of warm prefill, PROJECT_CONTEXT); zero-copy by `mmap`+`bytesNoCopy` was already tried and was slower when the expert set changes, so the resident set must be wired once at load, not per request | M | no |
| Layer-major prefill (above) | | 206 GB for 2.5k tokens | largest single I/O win for long states | L | no |
| `auto` + 256-token chunks | measured here | 12.2k expert reads/request | 16–45% latency on single questions | done (opt-in) | no |
| Lower-bit storage for streamed experts only | HOBBIT mixed-precision experts, <https://arxiv.org/abs/2411.01433>; EdgeMoE, <https://arxiv.org/abs/2308.14352> | 4-bit, 1.77 MB/expert | 3-bit ≈ −25% bytes, 2-bit ≈ −50%; dequant is cheap at 200 GB/s, but it changes outputs, so it is a weight change by this brief's rules | M | yes (repack) |
| Next-layer expert prediction / prefetch | Pre-gated MoE, <https://arxiv.org/abs/2308.12066>; ExpertFlow, <https://arxiv.org/abs/2410.17954>; SpecPrefetch (native routing kept), <https://arxiv.org/abs/2607.24787>; Edge0 (35B MoE from SSD, trained prerouter replaces routing plus a recovery LoRA), <https://arxiv.org/abs/2609.18063> | prefill already knows each layer's full union after that layer's router; the gap is between layers | small for prefill (a chunk's union is ~40–60% of a layer's experts, so there is little to predict); matters for decode and the "thinking budget" idea | M | no |
| `MTLIOCommandQueue` (read straight into Metal buffers, optional LZFSE/LZBITMAP) | WWDC22 "Load resources faster with Metal 3", <https://developer.apple.com/videos/play/wwdc2022/10104/>; <https://developer.apple.com/documentation/metal/mtliocommandqueue> | `pread` into a staging arena, 32 in flight, already on the SSD plateau | saves the CPU copy for SSD-sourced bytes (28%) only; page-cache hits still need a copy or a mapping; compression trades GPU/CPU time for fewer bytes | M | optional (compressed packs) |
| `F_NOCACHE` for streamed reads | fcntl(2); Apple File System Programming Guide performance tips | 72% of bytes come from the page cache at ~53 GB/s | negative on 16 GB: every read would pay the SSD's 6.8 GB/s | — | no |
| `F_RDADVISE` vs `F_RDAHEAD` | fcntl(2) | TUFF has `--rdadvise` (off by default); staging issues whole-expert reads at QD 32 | little: requests are already large and concurrent | S | no |
| Co-locate co-activated experts, bigger extents | "LLM in a Flash" row-column bundling | an expert is already one contiguous 1.77 MB extent | none measurable: 1.77 MB reads already reach 6.3–6.9 GB/s at QD 8–16 | M | yes |

### c. Model-side

- Small models only do well here after training. Kev (LoRA + pointer head, Apache-2.0,
  <https://github.com/jaredpalmer/kev>) reports new-source accuracy 0.697 (0.8B), 0.838 (4B), 0.852 (9B) and
  fitted temperatures 1.38–2.41 — the same correction measured here for the stock 35B. Untrained
  Qwen3.5-4B scores 0.637 balanced accuracy on WANLI (reported by SemIf,
  <https://github.com/TheoLeeCJ/SemIf>). Under "no weight changes" a small-model-first cascade would route
  easy cases through a weaker model; not recommended.
- Core ML / ANE: feasible only for a small model that fits next to the 35B's resident weights (a 4B at 4 bits
  is ~2.5 GB); weak untrained, as above. uzu routes some layers through MPSGraph (the path to the ANE),
  <https://github.com/trymirai/uzu>.
- Short thinking budget ("low reasoning effort"): budget forcing (s1, <https://arxiv.org/abs/2501.19393>)
  or Soft Thinking (<https://arxiv.org/abs/2505.15778>), confidence-gated to the ~10–15% of questions below
  a threshold. Needs a decode loop before the readout; worth measuring on `kev_choice` (0.566).

### d. Kernels

| kernel | measured | technique and reference | expected gain | effort | parity risk | M3+/M5 only? |
|---|---|---|---|---|---|---|
| Routed experts (`prefill_grouped_routed_moe_batched_*`) | 176 ms tile GPU (48 tok), 10.7 s (2,499 tok); one thread per (output row, token–expert pair), weights re-decoded per pair | bucket tokens per expert, 16-row tiles, gate+up in one `simdgroup_matrix` GEMM, GPU-written tile map + indirect dispatch (uzu `moe/experts_two_pass_prefill.metal`, `tiles_map.metal`); "decode once, apply to 1–8 rows" (mlx-serve ternary GEMV, <https://github.com/ddalcu/mlx-serve/pull/530>, 1.05–1.38×) | 2–4× on tiles | M | moderate (summation order) | no |
| INT4 projections (`prefill_dequant_int4_qmm_f16_block`) | 8×8 threads, one scalar output each, no threadgroup memory; in_proj ~5 GB/s effective at 48 tokens | MLX `quantized.metal` qmm (dequant into threadgroup tiles, `simdgroup_matrix`); llama.cpp `kernel_mul_mm` (64×32 tiles); split-K for small M (uzu `gemm_split_k_reduce.metal`); existing `--prefill-mbatch-int4` is 2–4× but moves probabilities 0.06 because it stages weights in half | 2–4× | M | the existing batched kernel shows the risk; keep FP32 accumulate and exact dequant | no |
| Attention prefill | 7.2 s GPU for 2,499 tokens (old kernel) | done: `simdgroup_matrix` kernel, 1.6–4.1×; next: share K/V tiles across query rows in threadgroup memory (uzu `attention_gemm_grouped.metal`, MLX `steel/attn`, FlashAttention-2, <https://arxiv.org/abs/2307.08691>; metal-flash-attention, <https://github.com/philipturner/metal-flash-attention>) | further ~2× on long prompts | S–M | fails end-to-end 1e-3 (see §5) | no |
| GDN recurrence (`gdn_delta_step_prefill`) | sequential over tokens, state in registers; 30.5 ms / 2.0 s | chunkwise UT form (§6a) | 2–4× | M | moderate | no |
| Command buffers | 480–820 per short call, 17k–21k for 2.5k tokens; warm idle 130–150 ms per call | fewer sync points; encoder reuse; indirect dispatch so tile counts need no CPU round trip (uzu); `MTLSharedEvent` to let the GPU wait on staged reads instead of the CPU | 10–25% on short prompts | M | none | no |
| Metal 4 tensor ops / MPP | M2 has none; TUFF gates them behind `mppTensorOpsAvailable` | MLX `*_nax` kernels, uzu MXU paths | M5 "neural accelerators" only | — | — | M5 (M3/M4 have no MPP matmul either) |

### e. The mlx.fast campaign (Yukon)

Yukon's mlx.fast challenge (<https://www.yukon.org/mlxfast>) runs AI-agent-driven speed submissions against
Layr-Labs engine repositories, scored on a ranked M5 box against a serial control. Its Ternary Bonsai 2 27B
track scores `prefill^0.25 · decode^0.75`; the record (winglock, Claude Opus 5.5) is 505.4% faster than the
control, 580.0 tok/s decode and 1901.4 tok/s prefill. Most of that is decode: DFlash speculative decoding
with 12-token drafts, which does nothing for a prefill-only endpoint. The prefill and MoE pieces, read from
the record trees (Bonsai: `Layr-Labs/mlxfast-bonsai2-27b-engine` PR #621, commit 831fae7,
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`; MoE: `Layr-Labs/mlxfast-qwen38-125b-a6b-engine`
PR #2977, `Runner/FastModel/TrackPrefill*.swift`, `TrackP12Prefill.swift`), map onto TUFF like this:

| technique (record tree) | what it does | TUFF today | fit |
|---|---|---|---|
| Bit-identical multi-row GDN prefill kernel (`Qwen35GDNPrefillKernel`, `MLXFAST_GDN_PREFILL_KERNEL`) | one simdgroup carries four value rows, sharing each step's q/k/g/beta loads; the four rows' reductions are interleaved so a step issues 13 shuffles instead of 40; same pairwise sums, so outputs are bit-identical, checked at process start against the stock kernel with fallback | `gdn_delta_step_prefill`: one simdgroup per value row, two `simd_sum`s per step (30.5 ms / 48 tokens, 2.0 s / 2,499 tokens) | **implemented and rejected** (below): bit-identical on M2 and M3, but slower than TUFF's stock kernel |
| Chunkwise GDN (`Qwen35GatedDeltaChunked`, chunk 8 or 16) | `prep` builds K Kᵀ, Q Kᵀ and the decay factors once per key head and chunk, shared by the value heads on that key head; `scan` carries state across chunks with each simdgroup holding 8 state rows in registers; FP32 8×8 MMAs; not bit-identical | sequential | same as §6a; would need the parity budget the fast mode needs |
| Counting sort of routed assignments (`TrackPrefillSort`) | stable counting sort over expert ids in 2 launches instead of MLX's 7-launch merge sort per `argSort` (672 launches per 1,024-token chunk); yields sorted ids, token rows and the inverse permutation directly | routing and pair sorting on the CPU between command buffers ("route CPU + shared encode") | removes a CPU round trip per layer if routing moves to the GPU |
| Tile table + fused gate/up/SwiGLU GEMM (`TrackPrefillIndirect`) | one 32-row tile per expert run slice, deterministic scans, no atomics; gate and up weights streamed together with the activation staged once and the SiLU product in the epilogue; down projection in 128×32 weight blocks | one thread per (output row, pair), weights re-decoded per pair (176 ms / 10.7 s tile GPU) | the routed-expert rewrite in §6d, #4 in the ranked list; the same design uzu uses |
| Copy removal by re-addressing (`TrackP12Prefill`) | consumers read split projections and sorted expert rows in place (through the inverse permutation) instead of reading a concatenated or scattered copy | TUFF's MoE writes `route_partials` per (token, rank) then reduces | small |
| Prefill submission pipeline (`MLXFAST_PREFILL_PIPELINE`) | the GPU starts the first layers while the CPU encodes the rest; boundaries chosen because every extra command buffer measured as a cost | 480–820 command buffers per short call, 130–150 ms warm GPU idle between them | fewer, earlier commits; M |
| Last-row LM head (`MLXFAST_PROMPT_LAST_ROW`) | prompt forwards compute the head only at the final position | already the case (`writeFinalHead`; head 2–8 ms warm) | — |

**The bit-identical GDN kernel, tried.** Implemented as `gdn_delta_step_prefill_rows` with R = 1, 2 or 4
value rows per simdgroup (function constant), the transposed butterfly, next-step loads issued early, and a
load-time bitwise self-check against the stock kernel. It matched bit for bit on both chips (27 cases: R =
1/2/4 × 1–393 rows × qwen36 and a small ragged shape, including a chunk continuing from a prior state), but was
slower (GPU time per dispatch, median of 20, qwen36 geometry):

```
GDN prefill recurrence microbenchmark, device: Apple M2 Pro, active: 1,2,4
   rows   stock ms    R=1 ms    R=2 ms    R=4 ms
     26      0.792     1.065     0.822     0.715
     48      0.914     1.950     1.501     1.296
    128      2.411     5.171     3.972     3.409
    256      4.803    10.325     7.924     6.789
```

(On an M3, a first version without the early loads ran at 0.62–0.80× of stock, and a variant keeping the
hardware `simd_sum` per row was no better.) TUFF's stock kernel already uses the hardware `simd_sum` and keeps
four times as many simdgroups in flight; the recurrence is bound by its per-step serial latency, and the extra
shuffles, broadcast and register traffic cost more than the shared loads save. The record's gain was against
MLX's Kahan-compensated kernel on an M5, a different baseline. The kernel was removed.

All of this was built and tuned on M5-class boxes; the tensor-unit bodies in the Bonsai record (the "Z" verify
bodies, the plane kernel) need the M5's matrix units, while the GDN, sort, tile-table and copy-removal pieces
are plain Metal and apply to the M2.

## 7. What changes on a larger Mac (32–64 GB)

- The expert pool (17 GB) fits: after the first request every expert is in the page cache, or better, loaded
  once into resident Metal buffers, and fetch drops from 23 s to zero for the long prompt and from ~0.2–1.2 s
  per call to zero for short ones. Layer-major prefill, hot-expert pinning, lower-bit streaming and
  `MTLIOCommandQueue` stop mattering.
- What remains is GPU work (0.3–0.75 s per short call on M2 Pro, ~27 s for 2,499 tokens), so the kernel list in
  §6d becomes the whole story, together with batching question suffixes.
- Accuracy and calibration findings do not depend on memory.
- An M3/M4 GPU (same `simdgroup_matrix` model, faster clocks, dynamic caching) scales the same kernels; only
  M5 adds matrix units that need separate MPP paths.

## 8. Ranked next steps

| # | step | expected gain | effort | first step |
|---|---|---|---|---|
| 1 | Serve with `--systemone-temperature` per model (1.48 / 2.44 / 2.22) | confident errors 4–9% → ~2%; NLL −0.07 to −0.34 | S | refit T on a larger labelled set that matches production traffic, then set it in the launch config |
| 2 | Make `auto` + chunk 256 the System One default after a parity decision | 16–45% on single questions | S | decide whether the 0.03 probability movement is acceptable (it is the same class as the existing prefix-reuse vs full-prefill difference) |
| 3 | Layer-major prefill across chunks | ~20 s off a 2,500-token request; ~10× fewer expert bytes | L | prototype for GDN + full layers with the existing per-chunk kernels, compare logits bit-for-bit against chunk-major |
| 4 | Routed-expert grouped GEMM (counting sort + tile table + fused gate/up/SwiGLU, indirect dispatch) | 2–4× on tile GPU time | M | port the mlx.fast MoE track's `TrackPrefillSort`/`TrackPrefillIndirect` structure (also uzu's) to TUFF's staged blobs behind a flag, with the 1e-3 parity test |
| 5 | Tiled INT4 GEMM with exact dequant and FP32 accumulate | 2–4× on projections | M | start from the existing `PrefillInt4MBatchQMM` but keep BF16/FP32 staging to recover parity |
| 6 | Confidence-gated short thinking budget | accuracy on `kev_choice` (0.566) | M | add a budgeted decode before the readout; measure decode tok/s on the M2 |
| 7 | Chunkwise GDN prefill | 2–4× on the recurrence | M | port MLX's 8-token chunk kernel |
| 8 | Resident hot experts read in place | saves the page-cache copy for ~78% of reads | M | measure the copy share with the new histograms, then wire a 9 GB set at load |
| 9 | Shared K/V tiles in the simdgroup attention kernel | ~2× on long-prompt attention | S | 8–16 query rows per threadgroup with K/V blocks in threadgroup memory |
| 10 | Accept `null` option descriptions (Jev compatibility) | correctness | S | render `A. key` when the description is null |

## 9. Flags and diagnostics added

| name | default | purpose |
|---|---|---|
| `--systemone-temperature <T>` | 1 | temperature on label log-probabilities |
| `--systemone-prefix-reuse auto` | (`on`) | reuse the shared prefix only for multi-question requests |
| `TUFF_PREFILL_ATTENTION_SIMDGROUP=1` | off | simdgroup causal attention for 256/8:1 full layers |
| `TUFF_PREFILL_PROFILE=1\|2` | off | now also: command buffers, GPU busy vs span, `pread` histogram |
| `TUFF_EXPERT_USAGE_FILE=<path>` | unset | per-layer, per-expert staged reads (with `TUFF_PREFILL_PROFILE`) |
| `TUFF_SYSTEMONE_LABEL_MASS=1` | off | label tokens' share of the vocabulary distribution |
| `TUFF_BENCH_ATTENTION=1` | off | attention microbenchmark test |

Build and tests on the Mac mini: `swift build -c release --product TUFFServer` →
`Build of product 'TUFFServer' complete!`; `Scripts/test.sh --filter
"SystemOne|HTTPServerTests|OpenAIValidation|ServerArguments|Prefill"` → `Test run with 279 tests in 58 suites
passed`. Serving with `--systemone-temperature 1.484` reproduces the eval-side transform:
`server T=1.484 vs eval-side power(base, 1.484): questions 12 max|dp| 4.44e-16`.

## Appendix: per-set tables

Generated by `python3 Scripts/systemone_eval.py report --tags apodex kat ornith --variants base --transforms
raw temp gtemp cc bc bc+temp pride` and `… --tags apodex --variants base perm fast fewshot4 sys-evidence
--transforms raw gtemp` on the Mac mini (`scratch/systemone_eval/report-base.md`, `report-variants.md`).
Test-only variants have no calibration split, so their `gtemp` rows equal `raw`.


#### Summary (report-base.md)

| model | request | transform | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apodex | base | raw | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |
| apodex | base | temp | 0.748 | 0.750 | 0.091 | 0.332 | 0.673 | 0.021 | 4156 | 9680 |
| apodex | base | gtemp | 0.748 | 0.750 | 0.105 | 0.339 | 0.688 | 0.022 | 4156 | 9680 |
| apodex | base | cc | 0.728 | 0.727 | 0.136 | 0.379 | 0.842 | 0.053 | 4156 | 9680 |
| apodex | base | bc | 0.745 | 0.748 | 0.114 | 0.355 | 0.810 | 0.046 | 4156 | 9680 |
| apodex | base | bc+temp | 0.745 | 0.748 | 0.082 | 0.335 | 0.685 | 0.023 | 4156 | 9680 |
| apodex | base | pride | 0.749 | 0.751 | 0.122 | 0.348 | 0.763 | 0.040 | 4156 | 9680 |
| kat | base | raw | 0.735 | 0.739 | 0.175 | 0.395 | 1.068 | 0.089 | 4948 | 10470 |
| kat | base | temp | 0.735 | 0.739 | 0.096 | 0.344 | 0.716 | 0.016 | 4948 | 10470 |
| kat | base | gtemp | 0.735 | 0.739 | 0.109 | 0.345 | 0.699 | 0.018 | 4948 | 10470 |
| kat | base | bc | 0.742 | 0.744 | 0.176 | 0.410 | 1.157 | 0.083 | 4948 | 10470 |
| kat | base | bc+temp | 0.742 | 0.744 | 0.097 | 0.364 | 0.796 | 0.028 | 4948 | 10470 |
| ornith | base | raw | 0.741 | 0.735 | 0.171 | 0.405 | 1.005 | 0.082 | 4475 | 10166 |
| ornith | base | temp | 0.741 | 0.735 | 0.123 | 0.396 | 1.982 | 0.070 | 4475 | 10166 |
| ornith | base | gtemp | 0.741 | 0.735 | 0.104 | 0.357 | 0.718 | 0.018 | 4475 | 10166 |
| ornith | base | bc | 0.746 | 0.752 | 0.157 | 0.379 | 0.970 | 0.071 | 4475 | 10166 |
| ornith | base | bc+temp | 0.746 | 0.752 | 0.115 | 0.374 | 1.503 | 0.049 | 4475 | 10166 |


#### apodex / base / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.059 | 0.197 | 0.333 | 0.040 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.043 | 0.092 | 0.175 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.126 | 0.236 | 0.398 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.106 | 0.226 | 0.791 | 0.095 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.050 | 0.124 | 0.293 | 0.030 | 4023 | 4451 |
| mnli | 200 | 0.915 | 0.911 | 0.049 | 0.137 | 0.290 | 0.015 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.125 | 0.256 | 0.922 | 0.095 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.122 | 0.537 | 1.068 | 0.021 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.206 | 0.683 | 1.321 | 0.045 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.144 | 0.570 | 1.124 | 0.045 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.338 | 0.777 | 1.726 | 0.000 | 6775 | 83456 |
| **mean** | | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |

#### apodex / base / temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.052 | 0.201 | 0.332 | 0.010 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.037 | 0.093 | 0.175 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.148 | 0.255 | 0.411 | 0.015 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.093 | 0.222 | 0.565 | 0.085 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.053 | 0.124 | 0.303 | 0.030 | 4023 | 4451 |
| mnli | 200 | 0.915 | 0.911 | 0.039 | 0.134 | 0.292 | 0.020 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.092 | 0.249 | 0.688 | 0.035 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.083 | 0.530 | 1.035 | 0.000 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.082 | 0.630 | 1.154 | 0.005 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.083 | 0.546 | 1.031 | 0.010 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.239 | 0.671 | 1.420 | 0.000 | 6775 | 83456 |
| **mean** | | 0.748 | 0.750 | 0.091 | 0.332 | 0.673 | 0.021 | 4156 | 9680 |

#### apodex / base / gtemp

Global temperature T = 1.484 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.030 | 0.197 | 0.325 | 0.010 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.058 | 0.102 | 0.190 | 0.015 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.170 | 0.258 | 0.414 | 0.015 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.098 | 0.223 | 0.579 | 0.085 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.066 | 0.134 | 0.301 | 0.025 | 4023 | 4451 |
| mnli | 200 | 0.915 | 0.911 | 0.096 | 0.156 | 0.321 | 0.005 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.106 | 0.251 | 0.715 | 0.060 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.078 | 0.531 | 1.036 | 0.000 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.104 | 0.637 | 1.169 | 0.005 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.082 | 0.547 | 1.034 | 0.020 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.266 | 0.698 | 1.490 | 0.000 | 6775 | 83456 |
| **mean** | | 0.748 | 0.750 | 0.105 | 0.339 | 0.688 | 0.022 | 4156 | 9680 |

#### apodex / base / cc

Kev groups have no content-free probe and are reported raw.

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.845 | 0.845 | 0.098 | 0.271 | 0.469 | 0.065 | 4538 | 7298 |
| sst2 | 200 | 0.825 | 0.823 | 0.100 | 0.248 | 0.395 | 0.060 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.126 | 0.236 | 0.398 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.875 | 0.872 | 0.114 | 0.239 | 0.914 | 0.105 | 4990 | 5448 |
| trec | 200 | 0.925 | 0.936 | 0.044 | 0.115 | 0.268 | 0.025 | 4023 | 4451 |
| mnli | 200 | 0.875 | 0.864 | 0.049 | 0.172 | 0.302 | 0.010 | 3997 | 5281 |
| banking77 | 200 | 0.865 | 0.845 | 0.110 | 0.242 | 0.888 | 0.110 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.122 | 0.537 | 1.068 | 0.021 | 6775 | 83456 |
| sst5 | 200 | 0.530 | 0.490 | 0.180 | 0.677 | 1.513 | 0.070 | 2884 | 3526 |
| amazon | 200 | 0.525 | 0.507 | 0.221 | 0.658 | 1.323 | 0.085 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.338 | 0.777 | 1.726 | 0.000 | 6775 | 83456 |
| **mean** | | 0.728 | 0.727 | 0.136 | 0.379 | 0.842 | 0.053 | 4156 | 9680 |

#### apodex / base / bc

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.870 | 0.869 | 0.048 | 0.200 | 0.338 | 0.050 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.037 | 0.089 | 0.167 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.848 | 0.060 | 0.237 | 0.381 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.885 | 0.884 | 0.114 | 0.220 | 0.749 | 0.090 | 4990 | 5448 |
| trec | 200 | 0.920 | 0.929 | 0.059 | 0.132 | 0.303 | 0.030 | 4023 | 4451 |
| mnli | 200 | 0.930 | 0.928 | 0.045 | 0.133 | 0.279 | 0.015 | 3997 | 5281 |
| banking77 | 200 | 0.870 | 0.861 | 0.112 | 0.238 | 0.764 | 0.090 | 9456 | 9800 |
| kev_choice | 235 | 0.579 | 0.643 | 0.087 | 0.544 | 1.080 | 0.026 | 6775 | 83456 |
| sst5 | 200 | 0.510 | 0.522 | 0.176 | 0.659 | 1.255 | 0.040 | 2884 | 3526 |
| amazon | 200 | 0.585 | 0.571 | 0.127 | 0.560 | 1.104 | 0.030 | 4041 | 6628 |
| kev_score | 12 | 0.250 | 0.233 | 0.383 | 0.893 | 2.489 | 0.083 | 6775 | 83456 |
| **mean** | | 0.745 | 0.748 | 0.114 | 0.355 | 0.810 | 0.046 | 4156 | 9680 |

#### apodex / base / bc+temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.870 | 0.869 | 0.051 | 0.204 | 0.336 | 0.010 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.041 | 0.090 | 0.167 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.848 | 0.122 | 0.239 | 0.382 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.885 | 0.884 | 0.107 | 0.214 | 0.567 | 0.080 | 4990 | 5448 |
| trec | 200 | 0.920 | 0.929 | 0.055 | 0.132 | 0.310 | 0.030 | 4023 | 4451 |
| mnli | 200 | 0.930 | 0.928 | 0.042 | 0.128 | 0.281 | 0.020 | 3997 | 5281 |
| banking77 | 200 | 0.870 | 0.861 | 0.063 | 0.227 | 0.603 | 0.030 | 9456 | 9800 |
| kev_choice | 235 | 0.579 | 0.643 | 0.060 | 0.539 | 1.057 | 0.004 | 6775 | 83456 |
| sst5 | 200 | 0.510 | 0.522 | 0.097 | 0.617 | 1.132 | 0.005 | 2884 | 3526 |
| amazon | 200 | 0.585 | 0.571 | 0.069 | 0.543 | 1.032 | 0.020 | 4041 | 6628 |
| kev_score | 12 | 0.250 | 0.233 | 0.199 | 0.755 | 1.670 | 0.000 | 6775 | 83456 |
| **mean** | | 0.745 | 0.748 | 0.082 | 0.335 | 0.685 | 0.023 | 4156 | 9680 |

#### apodex / base / pride

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.059 | 0.197 | 0.333 | 0.040 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.043 | 0.092 | 0.175 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.126 | 0.236 | 0.398 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.109 | 0.235 | 0.796 | 0.105 | 4990 | 5448 |
| trec | 200 | 0.925 | 0.914 | 0.030 | 0.119 | 0.281 | 0.025 | 4023 | 4451 |
| mnli | 200 | 0.930 | 0.927 | 0.039 | 0.122 | 0.252 | 0.010 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.125 | 0.256 | 0.922 | 0.095 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.122 | 0.537 | 1.068 | 0.021 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.206 | 0.683 | 1.321 | 0.045 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.144 | 0.570 | 1.124 | 0.045 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.338 | 0.777 | 1.726 | 0.000 | 6775 | 83456 |
| **mean** | | 0.749 | 0.751 | 0.122 | 0.348 | 0.763 | 0.040 | 4156 | 9680 |

#### kat / base / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.873 | 0.096 | 0.191 | 0.371 | 0.050 | 4826 | 7497 |
| sst2 | 200 | 0.960 | 0.960 | 0.029 | 0.059 | 0.133 | 0.015 | 2019 | 2928 |
| kev_noul | 66 | 0.848 | 0.841 | 0.125 | 0.201 | 0.315 | 0.030 | 6880 | 80534 |
| agnews | 200 | 0.860 | 0.857 | 0.125 | 0.254 | 1.131 | 0.105 | 4882 | 5253 |
| trec | 200 | 0.895 | 0.907 | 0.068 | 0.165 | 0.416 | 0.035 | 5127 | 5334 |
| mnli | 200 | 0.860 | 0.857 | 0.098 | 0.217 | 0.601 | 0.070 | 4946 | 5285 |
| banking77 | 200 | 0.875 | 0.863 | 0.108 | 0.234 | 0.976 | 0.095 | 10026 | 10813 |
| kev_choice | 235 | 0.574 | 0.592 | 0.252 | 0.659 | 1.595 | 0.132 | 6880 | 80534 |
| sst5 | 200 | 0.470 | 0.447 | 0.342 | 0.812 | 1.887 | 0.210 | 2661 | 3282 |
| amazon | 200 | 0.530 | 0.507 | 0.299 | 0.715 | 1.748 | 0.150 | 4687 | 6430 |
| kev_score | 12 | 0.333 | 0.431 | 0.384 | 0.843 | 2.574 | 0.083 | 6880 | 80534 |
| **mean** | | 0.735 | 0.739 | 0.175 | 0.395 | 1.068 | 0.089 | 4948 | 10470 |

#### kat / base / temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.873 | 0.058 | 0.171 | 0.286 | 0.015 | 4826 | 7497 |
| sst2 | 200 | 0.960 | 0.960 | 0.033 | 0.062 | 0.119 | 0.010 | 2019 | 2928 |
| kev_noul | 66 | 0.848 | 0.841 | 0.124 | 0.212 | 0.335 | 0.000 | 6880 | 80534 |
| agnews | 200 | 0.860 | 0.857 | 0.104 | 0.240 | 0.597 | 0.085 | 4882 | 5253 |
| trec | 200 | 0.895 | 0.907 | 0.068 | 0.164 | 0.413 | 0.035 | 5127 | 5334 |
| mnli | 200 | 0.860 | 0.857 | 0.058 | 0.194 | 0.385 | 0.025 | 4946 | 5285 |
| banking77 | 200 | 0.875 | 0.863 | 0.061 | 0.219 | 0.605 | 0.010 | 10026 | 10813 |
| kev_choice | 235 | 0.574 | 0.592 | 0.061 | 0.552 | 1.070 | 0.000 | 6880 | 80534 |
| sst5 | 200 | 0.470 | 0.447 | 0.109 | 0.653 | 1.209 | 0.000 | 2661 | 3282 |
| amazon | 200 | 0.530 | 0.507 | 0.062 | 0.573 | 1.066 | 0.000 | 4687 | 6430 |
| kev_score | 12 | 0.333 | 0.431 | 0.323 | 0.741 | 1.795 | 0.000 | 6880 | 80534 |
| **mean** | | 0.735 | 0.739 | 0.096 | 0.344 | 0.716 | 0.016 | 4948 | 10470 |

#### kat / base / gtemp

Global temperature T = 2.444 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.873 | 0.067 | 0.171 | 0.284 | 0.015 | 4826 | 7497 |
| sst2 | 200 | 0.960 | 0.960 | 0.047 | 0.069 | 0.135 | 0.005 | 2019 | 2928 |
| kev_noul | 66 | 0.848 | 0.841 | 0.126 | 0.209 | 0.329 | 0.000 | 6880 | 80534 |
| agnews | 200 | 0.860 | 0.857 | 0.101 | 0.237 | 0.553 | 0.070 | 4882 | 5253 |
| trec | 200 | 0.895 | 0.907 | 0.075 | 0.163 | 0.360 | 0.015 | 5127 | 5334 |
| mnli | 200 | 0.860 | 0.857 | 0.058 | 0.194 | 0.384 | 0.025 | 4946 | 5285 |
| banking77 | 200 | 0.875 | 0.863 | 0.069 | 0.220 | 0.610 | 0.010 | 10026 | 10813 |
| kev_choice | 235 | 0.574 | 0.592 | 0.080 | 0.558 | 1.081 | 0.009 | 6880 | 80534 |
| sst5 | 200 | 0.470 | 0.447 | 0.132 | 0.663 | 1.227 | 0.000 | 2661 | 3282 |
| amazon | 200 | 0.530 | 0.507 | 0.123 | 0.594 | 1.097 | 0.045 | 4687 | 6430 |
| kev_score | 12 | 0.333 | 0.431 | 0.316 | 0.712 | 1.630 | 0.000 | 6880 | 80534 |
| **mean** | | 0.735 | 0.739 | 0.109 | 0.345 | 0.699 | 0.018 | 4948 | 10470 |

#### kat / base / bc

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.870 | 0.868 | 0.092 | 0.197 | 0.388 | 0.055 | 4826 | 7497 |
| sst2 | 200 | 0.970 | 0.970 | 0.035 | 0.058 | 0.131 | 0.015 | 2019 | 2928 |
| kev_noul | 66 | 0.879 | 0.876 | 0.119 | 0.196 | 0.293 | 0.030 | 6880 | 80534 |
| agnews | 200 | 0.850 | 0.847 | 0.135 | 0.248 | 1.084 | 0.105 | 4882 | 5253 |
| trec | 200 | 0.895 | 0.906 | 0.064 | 0.173 | 0.441 | 0.045 | 5127 | 5334 |
| mnli | 200 | 0.870 | 0.867 | 0.093 | 0.215 | 0.592 | 0.075 | 4946 | 5285 |
| banking77 | 200 | 0.880 | 0.867 | 0.100 | 0.223 | 0.755 | 0.095 | 10026 | 10813 |
| kev_choice | 235 | 0.562 | 0.582 | 0.251 | 0.656 | 1.605 | 0.128 | 6880 | 80534 |
| sst5 | 200 | 0.500 | 0.516 | 0.269 | 0.738 | 1.619 | 0.140 | 2661 | 3282 |
| amazon | 200 | 0.555 | 0.544 | 0.264 | 0.673 | 1.617 | 0.145 | 4687 | 6430 |
| kev_score | 12 | 0.333 | 0.338 | 0.517 | 1.135 | 4.198 | 0.083 | 6880 | 80534 |
| **mean** | | 0.742 | 0.744 | 0.176 | 0.410 | 1.157 | 0.083 | 4948 | 10470 |

#### kat / base / bc+temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.870 | 0.868 | 0.032 | 0.176 | 0.294 | 0.015 | 4826 | 7497 |
| sst2 | 200 | 0.970 | 0.970 | 0.027 | 0.061 | 0.117 | 0.010 | 2019 | 2928 |
| kev_noul | 66 | 0.879 | 0.876 | 0.113 | 0.200 | 0.315 | 0.000 | 6880 | 80534 |
| agnews | 200 | 0.850 | 0.847 | 0.123 | 0.235 | 0.595 | 0.085 | 4882 | 5253 |
| trec | 200 | 0.895 | 0.906 | 0.059 | 0.170 | 0.417 | 0.045 | 5127 | 5334 |
| mnli | 200 | 0.870 | 0.867 | 0.046 | 0.193 | 0.381 | 0.025 | 4946 | 5285 |
| banking77 | 200 | 0.880 | 0.867 | 0.069 | 0.197 | 0.502 | 0.035 | 10026 | 10813 |
| kev_choice | 235 | 0.562 | 0.582 | 0.058 | 0.554 | 1.077 | 0.000 | 6880 | 80534 |
| sst5 | 200 | 0.500 | 0.516 | 0.091 | 0.631 | 1.167 | 0.000 | 2661 | 3282 |
| amazon | 200 | 0.555 | 0.544 | 0.087 | 0.561 | 1.040 | 0.005 | 4687 | 6430 |
| kev_score | 12 | 0.333 | 0.338 | 0.360 | 1.027 | 2.849 | 0.083 | 6880 | 80534 |
| **mean** | | 0.742 | 0.744 | 0.097 | 0.364 | 0.796 | 0.028 | 4948 | 10470 |

#### ornith / base / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.900 | 0.898 | 0.056 | 0.163 | 0.322 | 0.040 | 4766 | 8022 |
| sst2 | 200 | 0.955 | 0.955 | 0.028 | 0.074 | 0.192 | 0.020 | 1931 | 2633 |
| kev_noul | 66 | 0.848 | 0.843 | 0.097 | 0.243 | 0.428 | 0.045 | 6568 | 81515 |
| agnews | 200 | 0.845 | 0.840 | 0.138 | 0.271 | 1.067 | 0.120 | 4800 | 5209 |
| trec | 200 | 0.900 | 0.908 | 0.070 | 0.171 | 0.477 | 0.060 | 4204 | 5276 |
| mnli | 200 | 0.895 | 0.893 | 0.064 | 0.184 | 0.424 | 0.050 | 4239 | 5309 |
| banking77 | 200 | 0.880 | 0.857 | 0.096 | 0.221 | 0.984 | 0.090 | 9772 | 10595 |
| kev_choice | 235 | 0.485 | 0.509 | 0.224 | 0.715 | 1.553 | 0.060 | 6568 | 81515 |
| sst5 | 200 | 0.515 | 0.507 | 0.341 | 0.791 | 1.751 | 0.235 | 2801 | 3376 |
| amazon | 200 | 0.510 | 0.484 | 0.330 | 0.748 | 1.633 | 0.185 | 4261 | 6569 |
| kev_score | 12 | 0.417 | 0.389 | 0.442 | 0.874 | 2.223 | 0.000 | 6568 | 81515 |
| **mean** | | 0.741 | 0.735 | 0.171 | 0.405 | 1.005 | 0.082 | 4475 | 10166 |

#### ornith / base / temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.900 | 0.898 | 0.048 | 0.168 | 0.291 | 0.010 | 4766 | 8022 |
| sst2 | 200 | 0.955 | 0.955 | 0.032 | 0.081 | 0.166 | 0.010 | 1931 | 2633 |
| kev_noul | 66 | 0.848 | 0.843 | 0.139 | 0.277 | 0.443 | 0.000 | 6568 | 81515 |
| agnews | 200 | 0.845 | 0.840 | 0.088 | 0.252 | 0.568 | 0.095 | 4800 | 5209 |
| trec | 200 | 0.900 | 0.908 | 0.046 | 0.162 | 0.377 | 0.030 | 4204 | 5276 |
| mnli | 200 | 0.895 | 0.893 | 0.059 | 0.186 | 0.369 | 0.025 | 4239 | 5309 |
| banking77 | 200 | 0.880 | 0.857 | 0.095 | 0.224 | 0.673 | 0.015 | 9772 | 10595 |
| kev_choice | 235 | 0.485 | 0.509 | 0.058 | 0.640 | 1.234 | 0.000 | 6568 | 81515 |
| sst5 | 200 | 0.515 | 0.507 | 0.088 | 0.620 | 1.139 | 0.000 | 2801 | 3376 |
| amazon | 200 | 0.510 | 0.484 | 0.115 | 0.578 | 1.075 | 0.000 | 4261 | 6569 |
| kev_score | 12 | 0.417 | 0.389 | 0.583 | 1.167 | 15.464 | 0.583 | 6568 | 81515 |
| **mean** | | 0.741 | 0.735 | 0.123 | 0.396 | 1.982 | 0.070 | 4475 | 10166 |

#### ornith / base / gtemp

Global temperature T = 2.218 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.900 | 0.898 | 0.057 | 0.170 | 0.295 | 0.010 | 4766 | 8022 |
| sst2 | 200 | 0.955 | 0.955 | 0.043 | 0.084 | 0.170 | 0.010 | 1931 | 2633 |
| kev_noul | 66 | 0.848 | 0.843 | 0.091 | 0.251 | 0.401 | 0.015 | 6568 | 81515 |
| agnews | 200 | 0.845 | 0.840 | 0.090 | 0.253 | 0.571 | 0.095 | 4800 | 5209 |
| trec | 200 | 0.900 | 0.908 | 0.079 | 0.159 | 0.371 | 0.015 | 4204 | 5276 |
| mnli | 200 | 0.895 | 0.893 | 0.088 | 0.197 | 0.388 | 0.015 | 4239 | 5309 |
| banking77 | 200 | 0.880 | 0.857 | 0.096 | 0.227 | 0.683 | 0.010 | 9772 | 10595 |
| kev_choice | 235 | 0.485 | 0.509 | 0.072 | 0.641 | 1.240 | 0.000 | 6568 | 81515 |
| sst5 | 200 | 0.515 | 0.507 | 0.141 | 0.639 | 1.164 | 0.000 | 2801 | 3376 |
| amazon | 200 | 0.510 | 0.484 | 0.155 | 0.606 | 1.098 | 0.025 | 4261 | 6569 |
| kev_score | 12 | 0.417 | 0.389 | 0.231 | 0.699 | 1.521 | 0.000 | 6568 | 81515 |
| **mean** | | 0.741 | 0.735 | 0.104 | 0.357 | 0.718 | 0.018 | 4475 | 10166 |

#### ornith / base / bc

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.905 | 0.903 | 0.064 | 0.171 | 0.340 | 0.045 | 4766 | 8022 |
| sst2 | 200 | 0.960 | 0.960 | 0.034 | 0.072 | 0.186 | 0.020 | 1931 | 2633 |
| kev_noul | 66 | 0.848 | 0.846 | 0.123 | 0.243 | 0.415 | 0.061 | 6568 | 81515 |
| agnews | 200 | 0.860 | 0.857 | 0.131 | 0.257 | 0.986 | 0.105 | 4800 | 5209 |
| trec | 200 | 0.905 | 0.912 | 0.069 | 0.164 | 0.464 | 0.050 | 4204 | 5276 |
| mnli | 200 | 0.895 | 0.893 | 0.067 | 0.181 | 0.415 | 0.050 | 4239 | 5309 |
| banking77 | 200 | 0.875 | 0.859 | 0.091 | 0.221 | 0.736 | 0.085 | 9772 | 10595 |
| kev_choice | 235 | 0.489 | 0.515 | 0.219 | 0.718 | 1.500 | 0.068 | 6568 | 81515 |
| sst5 | 200 | 0.500 | 0.504 | 0.306 | 0.752 | 1.620 | 0.145 | 2801 | 3376 |
| amazon | 200 | 0.555 | 0.541 | 0.276 | 0.690 | 1.491 | 0.150 | 4261 | 6569 |
| kev_score | 12 | 0.417 | 0.484 | 0.343 | 0.697 | 2.512 | 0.000 | 6568 | 81515 |
| **mean** | | 0.746 | 0.752 | 0.157 | 0.379 | 0.970 | 0.071 | 4475 | 10166 |

#### ornith / base / bc+temp

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.905 | 0.903 | 0.055 | 0.174 | 0.302 | 0.020 | 4766 | 8022 |
| sst2 | 200 | 0.960 | 0.960 | 0.038 | 0.079 | 0.162 | 0.010 | 1931 | 2633 |
| kev_noul | 66 | 0.848 | 0.846 | 0.124 | 0.264 | 0.424 | 0.000 | 6568 | 81515 |
| agnews | 200 | 0.860 | 0.857 | 0.092 | 0.240 | 0.551 | 0.090 | 4800 | 5209 |
| trec | 200 | 0.905 | 0.912 | 0.059 | 0.157 | 0.380 | 0.040 | 4204 | 5276 |
| mnli | 200 | 0.895 | 0.893 | 0.043 | 0.182 | 0.364 | 0.020 | 4239 | 5309 |
| banking77 | 200 | 0.875 | 0.859 | 0.048 | 0.214 | 0.584 | 0.015 | 9772 | 10595 |
| kev_choice | 235 | 0.489 | 0.515 | 0.073 | 0.640 | 1.224 | 0.000 | 6568 | 81515 |
| sst5 | 200 | 0.500 | 0.504 | 0.113 | 0.618 | 1.143 | 0.000 | 2801 | 3376 |
| amazon | 200 | 0.555 | 0.541 | 0.098 | 0.564 | 1.043 | 0.010 | 4261 | 6569 |
| kev_score | 12 | 0.417 | 0.484 | 0.518 | 0.978 | 10.354 | 0.333 | 6568 | 81515 |
| **mean** | | 0.746 | 0.752 | 0.115 | 0.374 | 1.503 | 0.049 | 4475 | 10166 |

#### Summary (report-variants.md)

| model | request | transform | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apodex | base | raw | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |
| apodex | base | gtemp | 0.748 | 0.750 | 0.105 | 0.339 | 0.688 | 0.022 | 4156 | 9680 |
| apodex | perm | permavg | 0.905 | 0.908 | 0.075 | 0.166 | 0.436 | 0.058 | 13914 | 16830 |
| apodex | fast | raw | 0.750 | 0.750 | 0.129 | 0.350 | 0.773 | 0.040 | 4904 | 12800 |
| apodex | fast | gtemp | 0.750 | 0.750 | 0.129 | 0.350 | 0.773 | 0.040 | 4904 | 12800 |
| apodex | fewshot4 | raw | 0.743 | 0.748 | 0.138 | 0.361 | 0.789 | 0.041 | 11378 | 19097 |
| apodex | fewshot4 | gtemp | 0.743 | 0.748 | 0.138 | 0.361 | 0.789 | 0.041 | 11378 | 19097 |
| apodex | sys-evidence | raw | 0.754 | 0.756 | 0.120 | 0.348 | 0.777 | 0.042 | 5408 | 11306 |
| apodex | sys-evidence | gtemp | 0.754 | 0.756 | 0.120 | 0.348 | 0.777 | 0.042 | 5408 | 11306 |


#### apodex / base / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.059 | 0.197 | 0.333 | 0.040 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.043 | 0.092 | 0.175 | 0.020 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.126 | 0.236 | 0.398 | 0.030 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.106 | 0.226 | 0.791 | 0.095 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.050 | 0.124 | 0.293 | 0.030 | 4023 | 4451 |
| mnli | 200 | 0.915 | 0.911 | 0.049 | 0.137 | 0.290 | 0.015 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.125 | 0.256 | 0.922 | 0.095 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.122 | 0.537 | 1.068 | 0.021 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.206 | 0.683 | 1.321 | 0.045 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.144 | 0.570 | 1.124 | 0.045 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.338 | 0.777 | 1.726 | 0.000 | 6775 | 83456 |
| **mean** | | 0.748 | 0.750 | 0.124 | 0.349 | 0.767 | 0.040 | 4156 | 9680 |

#### apodex / base / gtemp

Global temperature T = 1.484 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.875 | 0.874 | 0.030 | 0.197 | 0.325 | 0.010 | 4538 | 7298 |
| sst2 | 200 | 0.945 | 0.945 | 0.058 | 0.102 | 0.190 | 0.015 | 2201 | 2751 |
| kev_noul | 66 | 0.848 | 0.841 | 0.170 | 0.258 | 0.414 | 0.015 | 6775 | 83456 |
| agnews | 200 | 0.880 | 0.878 | 0.098 | 0.223 | 0.579 | 0.085 | 4990 | 5448 |
| trec | 200 | 0.935 | 0.922 | 0.066 | 0.134 | 0.301 | 0.025 | 4023 | 4451 |
| mnli | 200 | 0.915 | 0.911 | 0.096 | 0.156 | 0.321 | 0.005 | 3997 | 5281 |
| banking77 | 200 | 0.855 | 0.844 | 0.106 | 0.251 | 0.715 | 0.060 | 9456 | 9800 |
| kev_choice | 235 | 0.566 | 0.627 | 0.078 | 0.531 | 1.036 | 0.000 | 6775 | 83456 |
| sst5 | 200 | 0.500 | 0.505 | 0.104 | 0.637 | 1.169 | 0.005 | 2884 | 3526 |
| amazon | 200 | 0.580 | 0.561 | 0.082 | 0.547 | 1.034 | 0.020 | 4041 | 6628 |
| kev_score | 12 | 0.333 | 0.343 | 0.266 | 0.698 | 1.490 | 0.000 | 6775 | 83456 |
| **mean** | | 0.748 | 0.750 | 0.105 | 0.339 | 0.688 | 0.022 | 4156 | 9680 |

#### apodex / perm / permavg

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agnews | 200 | 0.870 | 0.867 | 0.101 | 0.225 | 0.648 | 0.095 | 9252 | 9862 |
| trec | 200 | 0.940 | 0.950 | 0.050 | 0.107 | 0.224 | 0.020 | 16181 | 17005 |
| **mean** | | 0.905 | 0.908 | 0.075 | 0.166 | 0.436 | 0.058 | 13914 | 16830 |

#### apodex / fast / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.865 | 0.864 | 0.052 | 0.201 | 0.341 | 0.045 | 4134 | 7716 |
| sst2 | 200 | 0.945 | 0.945 | 0.036 | 0.089 | 0.166 | 0.020 | 1759 | 2106 |
| kev_noul | 66 | 0.848 | 0.841 | 0.096 | 0.236 | 0.389 | 0.030 | 7442 | 103415 |
| agnews | 200 | 0.880 | 0.878 | 0.108 | 0.224 | 0.784 | 0.095 | 6041 | 6632 |
| trec | 200 | 0.920 | 0.909 | 0.046 | 0.128 | 0.298 | 0.030 | 2793 | 6213 |
| mnli | 200 | 0.920 | 0.916 | 0.029 | 0.139 | 0.286 | 0.015 | 4990 | 6198 |
| banking77 | 200 | 0.865 | 0.849 | 0.120 | 0.251 | 0.924 | 0.095 | 11568 | 12993 |
| kev_choice | 235 | 0.583 | 0.643 | 0.108 | 0.537 | 1.066 | 0.021 | 7442 | 103415 |
| sst5 | 200 | 0.495 | 0.494 | 0.212 | 0.687 | 1.323 | 0.060 | 1943 | 2583 |
| amazon | 200 | 0.590 | 0.574 | 0.149 | 0.571 | 1.124 | 0.030 | 5192 | 6826 |
| kev_score | 12 | 0.333 | 0.343 | 0.460 | 0.780 | 1.807 | 0.000 | 7442 | 103415 |
| **mean** | | 0.750 | 0.750 | 0.129 | 0.350 | 0.773 | 0.040 | 4904 | 12800 |

#### apodex / fast / gtemp

Global temperature T = 1.000 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.865 | 0.864 | 0.052 | 0.201 | 0.341 | 0.045 | 4134 | 7716 |
| sst2 | 200 | 0.945 | 0.945 | 0.036 | 0.089 | 0.166 | 0.020 | 1759 | 2106 |
| kev_noul | 66 | 0.848 | 0.841 | 0.096 | 0.236 | 0.389 | 0.030 | 7442 | 103415 |
| agnews | 200 | 0.880 | 0.878 | 0.108 | 0.224 | 0.784 | 0.095 | 6041 | 6632 |
| trec | 200 | 0.920 | 0.909 | 0.046 | 0.128 | 0.298 | 0.030 | 2793 | 6213 |
| mnli | 200 | 0.920 | 0.916 | 0.029 | 0.139 | 0.286 | 0.015 | 4990 | 6198 |
| banking77 | 200 | 0.865 | 0.849 | 0.120 | 0.251 | 0.924 | 0.095 | 11568 | 12993 |
| kev_choice | 235 | 0.583 | 0.643 | 0.108 | 0.537 | 1.066 | 0.021 | 7442 | 103415 |
| sst5 | 200 | 0.495 | 0.494 | 0.212 | 0.687 | 1.323 | 0.060 | 1943 | 2583 |
| amazon | 200 | 0.590 | 0.574 | 0.149 | 0.571 | 1.124 | 0.030 | 5192 | 6826 |
| kev_score | 12 | 0.333 | 0.343 | 0.460 | 0.780 | 1.807 | 0.000 | 7442 | 103415 |
| **mean** | | 0.750 | 0.750 | 0.129 | 0.350 | 0.773 | 0.040 | 4904 | 12800 |

#### apodex / fewshot4 / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.885 | 0.884 | 0.048 | 0.202 | 0.344 | 0.030 | 17148 | 20397 |
| sst2 | 200 | 0.965 | 0.965 | 0.030 | 0.056 | 0.112 | 0.010 | 6523 | 7750 |
| kev_noul | 66 | 0.803 | 0.789 | 0.116 | 0.271 | 0.430 | 0.015 | 16237 | 95658 |
| agnews | 200 | 0.875 | 0.874 | 0.107 | 0.219 | 0.672 | 0.075 | 13622 | 15377 |
| trec | 200 | 0.925 | 0.935 | 0.049 | 0.117 | 0.270 | 0.025 | 9511 | 9751 |
| mnli | 200 | 0.940 | 0.938 | 0.067 | 0.129 | 0.259 | 0.015 | 11331 | 13917 |
| banking77 | 200 | 0.855 | 0.832 | 0.113 | 0.257 | 0.936 | 0.080 | 14045 | 14579 |
| kev_choice | 235 | 0.515 | 0.568 | 0.146 | 0.603 | 1.230 | 0.017 | 16237 | 95658 |
| sst5 | 200 | 0.535 | 0.523 | 0.238 | 0.685 | 1.310 | 0.085 | 9976 | 10822 |
| amazon | 200 | 0.540 | 0.541 | 0.254 | 0.688 | 1.338 | 0.100 | 10480 | 11822 |
| kev_score | 12 | 0.333 | 0.381 | 0.347 | 0.741 | 1.777 | 0.000 | 16237 | 95658 |
| **mean** | | 0.743 | 0.748 | 0.138 | 0.361 | 0.789 | 0.041 | 11378 | 19097 |

#### apodex / fewshot4 / gtemp

Global temperature T = 1.000 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.885 | 0.884 | 0.048 | 0.202 | 0.344 | 0.030 | 17148 | 20397 |
| sst2 | 200 | 0.965 | 0.965 | 0.030 | 0.056 | 0.112 | 0.010 | 6523 | 7750 |
| kev_noul | 66 | 0.803 | 0.789 | 0.116 | 0.271 | 0.430 | 0.015 | 16237 | 95658 |
| agnews | 200 | 0.875 | 0.874 | 0.107 | 0.219 | 0.672 | 0.075 | 13622 | 15377 |
| trec | 200 | 0.925 | 0.935 | 0.049 | 0.117 | 0.270 | 0.025 | 9511 | 9751 |
| mnli | 200 | 0.940 | 0.938 | 0.067 | 0.129 | 0.259 | 0.015 | 11331 | 13917 |
| banking77 | 200 | 0.855 | 0.832 | 0.113 | 0.257 | 0.936 | 0.080 | 14045 | 14579 |
| kev_choice | 235 | 0.515 | 0.568 | 0.146 | 0.603 | 1.230 | 0.017 | 16237 | 95658 |
| sst5 | 200 | 0.535 | 0.523 | 0.238 | 0.685 | 1.310 | 0.085 | 9976 | 10822 |
| amazon | 200 | 0.540 | 0.541 | 0.254 | 0.688 | 1.338 | 0.100 | 10480 | 11822 |
| kev_score | 12 | 0.333 | 0.381 | 0.347 | 0.741 | 1.777 | 0.000 | 16237 | 95658 |
| **mean** | | 0.743 | 0.748 | 0.138 | 0.361 | 0.789 | 0.041 | 11378 | 19097 |

#### apodex / sys-evidence / raw

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.855 | 0.854 | 0.076 | 0.209 | 0.354 | 0.055 | 6008 | 8568 |
| sst2 | 200 | 0.940 | 0.940 | 0.040 | 0.099 | 0.195 | 0.020 | 3280 | 4605 |
| kev_noul | 66 | 0.818 | 0.810 | 0.115 | 0.238 | 0.373 | 0.015 | 7870 | 85698 |
| agnews | 200 | 0.875 | 0.873 | 0.118 | 0.230 | 0.770 | 0.100 | 5314 | 7060 |
| trec | 200 | 0.930 | 0.919 | 0.035 | 0.119 | 0.279 | 0.025 | 5972 | 6192 |
| mnli | 200 | 0.920 | 0.916 | 0.030 | 0.145 | 0.301 | 0.015 | 5010 | 6456 |
| banking77 | 200 | 0.865 | 0.858 | 0.126 | 0.246 | 0.885 | 0.095 | 11011 | 11568 |
| kev_choice | 235 | 0.587 | 0.640 | 0.098 | 0.542 | 1.063 | 0.017 | 7870 | 85698 |
| sst5 | 200 | 0.505 | 0.516 | 0.212 | 0.690 | 1.341 | 0.060 | 4166 | 5032 |
| amazon | 200 | 0.585 | 0.568 | 0.161 | 0.561 | 1.123 | 0.055 | 5231 | 8117 |
| kev_score | 12 | 0.417 | 0.428 | 0.310 | 0.749 | 1.866 | 0.000 | 7870 | 85698 |
| **mean** | | 0.754 | 0.756 | 0.120 | 0.348 | 0.777 | 0.042 | 5408 | 11306 |

#### apodex / sys-evidence / gtemp

Global temperature T = 1.000 (fitted on all calibration items)

| set | n | acc | macro-F1 | ECE | Brier | NLL | conf-err | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 200 | 0.855 | 0.854 | 0.076 | 0.209 | 0.354 | 0.055 | 6008 | 8568 |
| sst2 | 200 | 0.940 | 0.940 | 0.040 | 0.099 | 0.195 | 0.020 | 3280 | 4605 |
| kev_noul | 66 | 0.818 | 0.810 | 0.115 | 0.238 | 0.373 | 0.015 | 7870 | 85698 |
| agnews | 200 | 0.875 | 0.873 | 0.118 | 0.230 | 0.770 | 0.100 | 5314 | 7060 |
| trec | 200 | 0.930 | 0.919 | 0.035 | 0.119 | 0.279 | 0.025 | 5972 | 6192 |
| mnli | 200 | 0.920 | 0.916 | 0.030 | 0.145 | 0.301 | 0.015 | 5010 | 6456 |
| banking77 | 200 | 0.865 | 0.858 | 0.126 | 0.246 | 0.885 | 0.095 | 11011 | 11568 |
| kev_choice | 235 | 0.587 | 0.640 | 0.098 | 0.542 | 1.063 | 0.017 | 7870 | 85698 |
| sst5 | 200 | 0.505 | 0.516 | 0.212 | 0.690 | 1.341 | 0.060 | 4166 | 5032 |
| amazon | 200 | 0.585 | 0.568 | 0.161 | 0.561 | 1.123 | 0.055 | 5231 | 8117 |
| kev_score | 12 | 0.417 | 0.428 | 0.310 | 0.749 | 1.866 | 0.000 | 7870 | 85698 |
| **mean** | | 0.754 | 0.756 | 0.120 | 0.348 | 0.777 | 0.042 | 5408 | 11306 |
