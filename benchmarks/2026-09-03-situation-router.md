# 2026-09-03 — the situation router: `routed` vs `legal` vs `features`, per model

First benchmark of the `routed` harness (docs/superpowers/specs/2026-09-02-situation-router-design.md).
Same screen as 08-22 so rows compare: seed 1, 30-piece cap, `--decision-deadline 15` (+p15),
`medium` effort where supported, one local model resident at a time. Baselines from 08-22 at
this seed and cap: heuristic 490, random 140, no-input 55 (race). Corpus validation:
`benchmarks/2026-09-03-situation-histogram.md`.

Hypothesis (spec section 08, refined in the plan): a named technique makes the model
deliberate less — output tokens per decision and latency fall — without `race`/`score`
regressing against `legal` and `features`.

## Session-to-session drift: the same `features` arms on 08-22 and today

Same seed, cap, deadline, harness and effort — and every arm moved between sessions:

| arm (`features`, +p15, seed 1) | 08-22 | 09-03 |
|---|---|---|
| gpt-oss:120b-cloud | race 430, 2 lines, 593 tok/decision | race 530, 9 lines, 560 tok/decision |
| gpt-oss:20b-cloud | race 135, 20 decisions | race 170, 18 decisions |
| gpt-oss:20b (local) | race 90, 9 decisions, 153 tok/decision | race 170, 16 decisions, 277 tok/decision |

That is why this sweep re-ran `features` in the same session instead of reusing the 08-22
rows: the within-session comparisons below are sound, but ±100 race at one seed is inside
the session-to-session swing and is not a result. In particular, the local arm's "175 vs
170" under `routed` vs `features` is a tie, not an improvement.

## Results (seed 1, 30-piece cap, +p15)

| arm | race | score | lines | pieces | timeouts | %≤15s | decisions | illegal | tok/decision | in tok | s/decision | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:120b-cloud/legal | 50 | 0 | 0 | 10 | 7 | 30.0 | 3 | 7 | 1,809 | 1,112 | 41.37 | $0.00 |
| **gpt-oss:120b-cloud/features** | **530** | **380** | **9** | **30** | 0 | 100.0 | 30 | 0 | 560 | 30,795 | 2.93 | $0.00 |
| gpt-oss:120b-cloud/routed | 190 | 40 | 1 | 30 | 2 | 93.3 | 28 | 2 | 253 | 18,821 | 2.99 | $0.00 |
| gpt-oss:20b-cloud/legal | 55 | 0 | 0 | 11 | 10 | 9.1 | 1 | 10 | 967 | 364 | 161.75 | $0.00 |
| **gpt-oss:20b-cloud/features** | **170** | **40** | **1** | **26** | 8 | 69.2 | 18 | 8 | 623 | 19,825 | 13.84 | $0.00 |
| gpt-oss:20b-cloud/routed | 80 | 0 | 0 | 16 | 8 | 50.0 | 8 | 8 | 268 | 5,211 | 18.83 | $0.00 |
| gpt-oss:20b (local)/legal | 65 | 0 | 0 | 13 | 4 | 69.2 | 9 | 4 | 342 | 4,204 | 15.21 | $0.00 |
| gpt-oss:20b (local)/features | 170 | 40 | 1 | 26 | 10 | 61.5 | 16 | 10 | 277 | 15,213 | 16.26 | $0.00 |
| **gpt-oss:20b (local)/routed** | **175** | **40** | **1** | **27** | 9 | 66.7 | 18 | 9 | 210 | 11,477 | 12.79 | $0.00 |

Arm ids as recorded in the JSON: `pi/gpt-oss:120b-cloud/<harness>+p15`,
`pi/gpt-oss:20b-cloud/<harness>+p15`, `pi/gpt-oss:20b/<harness>/medium+p15` (the local arm
is the only one that supports effort). The table shortens them for readability.

Cost is $0.00 across the board: the cloud models (`gpt-oss:120b-cloud`, `gpt-oss:20b-cloud`)
are unlisted in `pricing.py` so they ran effort-free with no `/medium` suffix in the arm
name (expected, per the sweep design); the local arm has no API bill by definition.

## Per-model read

**gpt-oss:120b-cloud**: `routed` cut tok/decision by 55% versus `features` (253 vs 560) but
latency was flat (~3.0s either way, suggesting serving speed, not token count, sets the floor
here). Race collapsed from 530 to 190 and score from 380 (9 lines) to 40 (1 line): `features`
ran clean (0 illegal, 0 timeouts, 100% ≤15s), `routed` picked up 2 illegal placements and 2
timeouts. `legal` is the outlier — 1,809 tok/decision and 41s/decision, flatlining at 10
pieces with 7 of 10 decisions timed out or illegal; giving the model the raw legal list with
no framing is by far the most expensive prompt for this model.

**gpt-oss:20b-cloud**: same token-count story — `routed` tok/decision fell 57% versus
`features` (268 vs 623) — but race fell too, 170 to 80, and score dropped to 0 (features held
1 line). Timeouts and illegal counts stayed flat at 8 across both, so the tokens saved did not
buy fewer failures; pieces survived actually dropped (26 → 16). `legal` again flatlines
hardest: 1 completed decision in 11 pieces, mean latency over 160s (dominated by the 10
timeouts eating the full 15s clock each).

**gpt-oss:20b (local)**: the one arm where the hypothesis holds as written. `routed` cut
tok/decision 24% versus `features` (210 vs 277) *and* latency fell 21% (12.8s vs 16.3s/decision),
while race held (175 vs 170, a tie), score held (40, 1 line), and pieces survived rose
(27 vs 26). `legal` sits between the other two harnesses on race (65) but is not directly
comparable — fewer decisions completed (9 vs 16–18) before the run ended.

## Which model plays best under routing

Ranked by race on the `routed` arm:

1. **gpt-oss:120b-cloud** — 190
2. **gpt-oss:20b (local)** — 175
3. **gpt-oss:20b-cloud** — 80

Under `features` the order was gpt-oss:120b-cloud (530) far ahead, then gpt-oss:20b-cloud and
gpt-oss:20b (local) tied at 170. Routing does not change who's on top — 120b-cloud stays
first — but it **flips the runner-up**: the local model, tied with the cloud 20b under
`features`, pulls ahead of it under `routed` (175 vs 80). Routing hurt the cloud 20b more than
the local 20b. They share weights but not serving backend, quantization or context handling,
and this is one seed, so which of those explains the gap is not something this run can say.

On the ≲300 tok/decision viability bar from 08-22: **all three arms cross it under `routed`**
(253, 268, 210) — including both cloud arms, which sat well above it under `features` (560,
623). The local arm was already under the bar in `features` (277); routing pushed it lower
still (210) without costing race.

## Verdict on the hypothesis

**Partially supported, and it plainly lost for two of three models.** Tok/decision fell under
`routed` versus `features` for every model (55%, 57%, 24% respectively) — the token-reduction
half of the hypothesis holds across the board. But the "without race/score regressing" half
only held for **gpt-oss:20b (local)**, where latency also fell and race/score held or was a
tie. For **gpt-oss:120b-cloud** and **gpt-oss:20b-cloud**, `routed` lost hard on race and score
against `features` (530→190 and 380→40 score for the 120b; 170→80 and a lost line for the
20b-cloud), and latency did not improve in step with the token cut (flat for 120b-cloud, worse
for 20b-cloud). The cloud arms also picked up illegal placements/timeouts under `routed` that
`features` didn't have (120b-cloud: 0→2 each). Which situations are implicated is not
measurable yet — `placement_decision` events don't carry the situation the model was told;
that is the first follow-up.

## Follow-ups

- **Annotation/technique coherence (candidate cause of the cloud regression).** Two of the
  five techniques ask for something the placement annotations do not show: MOUND (64.5 % of
  boards) says "without opening a hole" but `METRICS[MOUND]` shows only `bumpiness`;
  HOLE_RISK says "as low on the board as possible" but shows only `holes`; TETRIS_READY's
  non-I branch shows only `lines`, which is 0 on every row. The cloud arms picked up illegal
  placements and timeouts under `routed` that `features` did not have. Not changed here —
  it would orphan these rows — so the fix is a `routed-v2` metrics table and a re-run.
- **Record `situation` on `placement_decision` events** (both `agent.py` and
  `live_agent.py`) so a model run can be broken down per situation; without it the
  regression above cannot be attributed.
- **`tetris-bench` has no genome path.** `build_policy`/`run_arm` accept `genome_params`
  but `main` never passes one, so sweeps always classify with default thresholds and the
  spec's "evolve.py tunes them" cannot be exercised through the benchmark. One-liner plus a
  decision about whether the control arm should start reading `data/genome.json` too.

Runs: `data/benchmarks/benchmark-20260903-034956.json`, `data/benchmarks/benchmark-20260903-040108.json`.
Skipped: `claude-haiku-4-5` and `claude-sonnet-5` — no `ANTHROPIC_API_KEY` set in this session
(no key file found either); run later with:
`uv run tetris-bench --models claude-haiku-4-5 claude-sonnet-5 --harnesses legal features routed --efforts medium --seeds 1 --max-pieces 30 --decision-deadline 15 --max-usd 3.00 --no-control --no-power`.
`deepseek-v4-flash:0731-cloud` — flatlined on the 08-22 screen (10 timeouts of 14 pieces at
~1,900 tokens per decision); re-running it would cost ~20 minutes of dead time for no new
information.
