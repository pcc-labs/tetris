# 2026-09-03 — the local roster on the reasoning-level matrix: gemma4:26b reaches the cloud

The local counterpart of `2026-09-03-live-reasoning-matrix.md`: same screen (live gravity,
level 0, seed 1, 30-piece cap, `features`, effort pinned with `--fixed-effort`), the models
that fit the Strix Halo iGPU, host power sampled. One model resident at a time, evicted
between models; one `tetris-bench` invocation per (model, effort).

    uv run tetris-bench --models pi/<tag> --harnesses features --efforts <off|low|medium>
      --fixed-effort --seeds 1 --max-pieces 30 --no-control

Before this file no local arm had ever run with thinking actually off: the local entries in
`~/.pi/agent/models.json` lacked the `off` map, so pi dropped the flag — the same silent
default-thinking that hid the cloud tags' play until this morning. The entries now carry
`thinkingLevelMap.off = "none"` (backup `models.json.bak5`), and a pi probe confirmed the
flag lands: gemma4 answers a toy board in 10 tokens at `off`; gpt-oss:20b takes 137 tokens
and 3.5 s at `low` versus 3,024 tokens and 66 s at its default.

Roster: the fast MoE / small models at all three levels; `glm-4.7-flash-32k` at `off` only
(the local flash question — does it obey the flag, unlike glm-5.3-flash?); the dense
`gemma4-31b-32k` at `off` only (prefill-bound at any level). `gemma4:26b` is the 26B-A4B
MoE (19 GB, Q4_K_M), pulled today; it was the top local candidate after the 31B scored 450
in the cloud.

## The curves (race; bold ≥ 300)

| model | size | off | low | medium |
|---|---|---|---|---|
| **gemma4:26b** (26B-A4B) | 19 GB | **530** | 55 | 55 |
| gemma4 (e4b, `latest`) | 9.6 GB | **330** | 55 | 55 |
| laguna-xs-32k (33B-A3B) | 20 GB | **310** | 55 | 55 |
| glm-4.7-flash-32k | 19 GB | 230 | — | — |
| gpt-oss:20b (21B-A3.6B) | 13 GB | 55 (default) | 225 | 55 |
| nemotron-3.5-lightning-32k (30B-A3B) | 25 GB | 215 | 55 | 55 |
| gemma4-31b-32k (dense) | 19 GB | 55 | — | — |

Live-screen references: heuristic 490, random 140, no-input 55; race 55 at 11 pieces is
the live flatline. Cloud rows on the same screen, for the twins: gpt-oss:20b `low` 530,
gemma4 31B `off` 530, nemotron-3-super `off` 730.

## Every arm, ranked

| arm | race | score | lines | pieces | late | avg holes | tok/decision | s/decision | tok/s | Wh | mean W (idle) | peak W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **gemma4:26b / off** | **530** | **380** | **9** | 30 | 1 | **0.00** | 33 | 3.04 | 10.7 | 0.13* | 51.4 (48.1) | 99.1 |
| gemma4 / off | 330 | 180 | 4 | 30 | 0 | 10.20 | 34 | 2.18 | 15.8 | 0.88 | 46.1 (16.5) | 90.0 |
| laguna-xs-32k / off | 310 | 160 | 4 | 30 | 0 | 5.77 | 26 | 2.44 | 10.8 | 1.23 | 51.1 (12.0) | 96.0 |
| glm-4.7-flash-32k / off | 230 | 80 | 2 | 30 | 0 | 6.50 | 30 | 2.47 | 12.0 | 0.88 | 53.4 (25.5) | 106.0 |
| gpt-oss:20b / low | 225 | 80 | 2 | 29 | 2 | 0.48 | 107 | 3.62 | 29.6 | 1.40 | 63.0 (24.1) | 93.0 |
| nemotron-3.5-lightning-32k / off | 215 | 80 | 2 | 27 | 1 | 14.89 | 33 | 2.79 | 11.8 | 0.81 | 55.7 (27.0) | 101.0 |
| gpt-oss:20b / off (default) | 55 | 0 | 0 | 11 | 4 | 12.73 | 478 | 29.15 | 33.8 | 1.14 | 77.7 (28.6) | 106.1 |
| gpt-oss:20b / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 492 | 37.92 | 45.7 | 0.51 | 78.3 (56.6) | 94.0 |
| gemma4 / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 997 | 72.34 | 51.5 | 1.52 | 77.8 (13.0) | 93.0 |
| gemma4 / medium | 55 | 0 | 0 | 11 | 3 | 13.54 | 924 | 36.15 | 53.2 | 1.22 | 77.7 (25.6) | 90.0 |
| gemma4-31b-32k / off | 55 | 0 | 0 | 11 | 5 | 12.73 | 30 | 18.08 | 2.5 | 1.15 | 75.6 (27.1) | 113.1 |
| gemma4:26b / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 1,499 | 69.59 | 67.1 | 1.42 | 88.0 (27.5) | 107.0 |
| gemma4:26b / low; laguna-xs / low, medium; nemotron-lightning / low, medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 0 | ~69 | — | 1.1–1.7 | 74–88 | 87–106 |

Arm ids: `pi/<tag>/features/<effort>+live+fixed`. `Wh` is marginal over the idle baseline
sampled before each arm; `mean W` is the whole-run draw with that baseline in parentheses,
`peak W` the highest sample. *The gemma4:26b `off` baseline read 48 W — the previous
model's eviction was still settling — so its 0.13 Wh marginal is an artefact; read its mean
51 W against the ~12–28 W idle the other arms saw (≈1.5 Wh for the run). `tok/decision` 0
with ~69 s per decision is four killed calls with no usage reported. Runs:
`data/benchmarks/benchmark-20260903-145518.json` through `-152327.json`, one per arm.

## Read

**Local parity, on this screen: gemma4:26b with thinking off scores 530.** Nine lines, all 30
pieces, *zero* holes — the only zero-hole row on any screen — at 3.0 s per decision and about
50 W. 530 is the cloud cluster (gpt-oss:20b `low`, deepseek-v4-pro `off`, gemma4 31B `off`)
and the original 120b reference; it is not nemotron-3-super's 730. It came from the one
model pulled today, at 4B active parameters, at the setting the cloud matrix said to use.
Three days ago the best local row was 220.

**The curve shape is the cloud's.** Six of seven local models play at exactly one level,
`off`; gpt-oss:20b (no off) plays only at `low`. At `low` every other model wrote 900–1,500
tokens or was killed before writing any, at 36–72 s per decision, against a ~15 s fall. The
iGPU's decode rate makes the cutoff harsher than the cloud's: there, `low` cost kimi-k3 4.5 s;
here it costs gemma4 72 s.

**gpt-oss:20b at low is 225 locally against 530 for its cloud twin.** Latency is not the
difference — 3.6 s per decision here, 3.0 there, two late pieces. The play is: 0.48 holes per
piece (clean) but two lines in 29 pieces and a top-out, where the cloud copy cleared nine.
Same weights, same seed, same effort. Either the placements differ (a per-placement quality
measure would say — the harness has none yet) or the 0.6 s and two late pieces cascaded on
this seed. One seed cannot separate those.

**The small gemma4 is fast and sloppy.** 330 at 2.2 s per decision with 10 holes per piece:
it survives on speed. laguna-xs is the same story at 310 and 5.8 holes. glm-4.7-flash obeys
the flag (unlike glm-5.3-flash) and plays at 230; the 08-20 note that "pi's thinking knob
does not curb it" was the missing off map, not the model. The dense 31B flatlines at `off`
too: 18 s per decision for a 30-token answer is prefill, and no effort setting touches
prefill.

**Power.** Every playing arm drew 46–63 W mean, 90–106 W peak, about 1 Wh for a 30-piece
game at $0.39/kWh — a few ten-thousandths of a dollar. The flatlined `low`/`medium` arms
drew *more* (74–88 W mean) for nothing: the model was generating the whole time.

## What this changes

1. **The local reference arm is `pi/gemma4:26b/features/off+live+fixed` at 530.** The
   parity target on the live screen is nemotron-3-super's 730, and the local gap is now
   200 race at one seed, not 300+ with no local arm on the board.
2. **Next local rows:** `gemma4:26b` under `routed` (the harness that halves prefill) and
   across more seeds; nemotron-3-super's local tag (87 GB, needs the GTT bump) at `off`,
   since its cloud copy leads and it answers in 28 tokens.
3. **Per-placement quality is the next measurement** (see the README's open items): the
   gpt-oss:20b local-vs-cloud gap and gemma4-small's 10 holes are both questions the
   aggregates cannot answer, and a per-decision record is also the training signal for
   fine-tuning a local model past the cloud.

Single seed; ±100 is noise. 530 vs 220 is not.
