# 2026-08-22 — the 15-second question: every local model, plus Ollama cloud

First full screen under the new bounded-pause mode (`--decision-deadline 15`,
PR #8): the game freezes once per piece, the pi subprocess is killed at
exactly 15 s, and a killed decision drops the piece untouched. Seed 1,
`features` harness, 30-piece cap, `medium` effort where the model supports
thinking, one model resident at a time, all arms watched live in the viewer.

The hypothesis under test: **tokens per second decide who can play.**

## Results (seed 1, 30-piece cap, +p15)

| arm | race | score | lines | pieces | timeouts | %≤10s | %≤15s | decisions | tok/decision |
|---|---|---|---|---|---|---|---|---|---|
| **gpt-oss:120b-cloud** | **430** | **280** | **2** | **30** | 0 | 100 | 100 | 30 | 593 |
| gpt-oss:20b-cloud | 135 | 0 | 0 | 27 | 7 | 74 | 74 | 20 | 554 |
| gpt-oss:20b (local) | 90 | 0 | 0 | 18 | 9 | 44 | 50 | 9 | 153 |
| nemotron-3.5-lightning-32k | 70 | 0 | 0 | 14 | 8 | 43 | 43 | 6 | 29 |
| deepseek-v4-flash:0731-cloud | 70 | 0 | 0 | 14 | 10 | 29 | 29 | 4 | 1911 |
| glm-4.7-flash-32k | 65 | 0 | 0 | 13 | 12 | 0 | 8 | 1 | 465 |
| gemma4:latest | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |
| qwen3.6-27b-32k | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |
| muse-glimmer-30b-32k | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |
| qwen3.8-27b-32k | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |
| qwen3.6-35b-32k | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |
| gemma4-31b-32k | 50 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | — |

Race 50 with 10 pieces is the flatline floor: every decision killed at 15 s,
the game plays itself. (08-20 baselines, same seed and cap: heuristic 490,
random 140, no-input 55.)

## The controlled A/B: same weights, different tok/s

`gpt-oss:20b` ran twice — on the Strix Halo iGPU and on Ollama cloud.
Identical weights, identical seed, identical prompts; only the serving speed
differs.

| | local iGPU | cloud |
|---|---|---|
| race | 90 | 135 |
| pieces survived | 18 | 27 |
| decisions in time | 50% | 74% |
| completed decisions | 9 | 20 |

**Speed alone bought +50% survival.** The hypothesis is confirmed as far as
it goes — and then it stops: the cloud copy *still scored zero*. Once
latency stops being the binding constraint, the 20B's skill ceiling is
exposed. `gpt-oss:120b-cloud` — bigger brain at the same fast serving —
placed all 30 pieces, 100% inside the clock, and was the **only arm in the
entire matrix to clear a line**. Capability and speed are two axes, and you
need both.

## Why the locals actually failed: prefill, not decode

Standalone probes (direct `/api/generate`, ~200-token completion, measured
after the matrix):

| model | decode tok/s | prompt tok/s | est. prefill of a ~2k-token board prompt |
|---|---|---|---|
| nemotron-3.5-lightning-32k | 80.3 | 158 | ~13 s |
| glm-4.7-flash-32k | 66.5 | 128 | ~16 s |
| qwen3.6-35b-32k | 64.7 | 155 | ~13 s |
| gemma4:latest | 59.4 | 451 | ~4 s |
| gpt-oss:20b | 52.5 | 246 | ~8 s |
| qwen3.8-27b-32k | 37.5 | 62 | ~32 s |
| muse-glimmer-30b-32k | 13.1 | 122 | ~16 s |
| qwen3.6-27b-32k | 12.9 | 69 | ~29 s |
| gemma4-31b-32k | 11.3 | 64 | ~31 s |

Three of the flatliners (qwen3.8-27b, qwen3.6-27b, gemma4-31b) lose the game
**during prompt processing**: at 62–69 tok/s of prefill, the ~2k-token board
prompt costs ~30 s before the first output token exists. Decode speed is
irrelevant for them — qwen3.8-27b decodes a healthy 37 tok/s and it never
mattered. The 08-20 note blamed decode bandwidth for the dense-27B wall;
this run corrects that: **on this box the wall is prefill rate**, with decode
second.

The other failure mode is **token budget**: deepseek-v4-flash-cloud averaged
1,911 tokens per decision — at datacenter speed, with fast prefill, it still
blew the clock on 10 of 14 pieces. glm-4.7-flash (465 tok/decision on a
~16 s prefill) completed exactly one decision all run. Fast serving cannot
save a model that won't stop talking. gemma4:latest is the anomaly — fast
prefill (451 tok/s), fine decode, zero completed calls — consistent with
unbounded verbosity, unverified.

## Cloud arms against the reference arms

Placing the three cloud rows on the reference scale (same seed, 30-piece cap;
baselines from the 08-20 note):

| arm | race | score | lines | pieces | %≤15s | tok/decision |
|---|---|---|---|---|---|---|
| heuristic (solver ceiling) | 490 | 340 | 8 | 30 | — | — |
| **gpt-oss:120b-cloud** | **430** | **280** | 2 | 30 | 100 | 593 |
| random (chance floor) | 140 | 0 | 0 | 28 | — | — |
| gpt-oss:20b-cloud | 135 | 0 | 0 | 27 | 74 | 554 |
| gpt-oss:20b (best local) | 90 | 0 | 0 | 18 | 50 | 153 |
| deepseek-v4-flash:0731-cloud | 70 | 0 | 0 | 14 | 29 | 1,911 |
| no-input (nobody plays) | 55 | 0 | 0 | 11 | — | — |

- **gpt-oss:120b-cloud lands within 12% of the heuristic solver** (430 vs
  490) — the solver exhaustively searches placements; the model just reads
  the same feature table. It is the only model arm ever to clear a line.
- **gpt-oss:20b-cloud ties the random floor (135 vs 140) for opposite
  reasons**: random survives 28 pieces on the luck of the legal action
  space; 20b-cloud survives 27 by answering in time and placing mediocrely.
  Same race score, different failure mode — the `lines`/`%≤15s` columns
  tell them apart.
- **deepseek-v4-flash-cloud scores below the best local arm** and barely
  above no-input: 1,911 tokens per decision fits no 15-second clock at any
  serving speed. Cloud hosting moves which term of the latency budget
  binds; it does not remove the budget.

Net: cloud takes the top two model slots, but the ranking follows the
latency budget, not the hosting — fast serving promoted the disciplined
models and did nothing for the verbose one.

## Verdict on the hypothesis

Tokens per second matter — the A/B proves it causally, not just
correlationally. But the full latency model has three terms, and any one of
them can be fatal on a 15-second clock:

```
prompt_tokens / prefill_rate  +  output_tokens / decode_rate  <  15 s
       (prefill wall)              (budget × decode wall)
```

- **Prefill-bound** (dense 27–31B locals): dead before the first token.
- **Budget-bound** (glm, deepseek-flash, likely gemma4): dead talking.
- **Capability-bound** (gpt-oss:20b at speed): alive, but can't clear lines.
- **All three solved** (gpt-oss:120b-cloud): the only score on the board.

**Roster implication:** a viable *local* arm on this box needs prefill
≳200 tok/s, decode ≳50 tok/s, and ≲300 tokens per decision. Of everything
pulled, only gpt-oss:20b clears all three bars — same conclusion as 08-20,
now with the mechanism measured. Shortening the prompt (a terser harness) is
the cheapest lever no model change can match: halving prompt tokens halves
the prefill term for every arm at once.

Results JSONs: `data/benchmarks/benchmark-20260822-*.json` (12 arms).
Probes: single-shot generate, model evicted between; not concurrent with any
arm.
