# Benchmarks

Dated results, one file per run day (`YYYY-MM-DD-<subject>.md`, UTC to match run
ids), **append-only**: never rewrite an old file, add a new dated one so progress
stays visible over time.

`tetris-bench` writes `data/benchmarks/*.json` automatically and `data/` is
gitignored, so those results never leave the machine that produced them. This
folder is the committed record. Every row should be reconstructible — from an
arm name plus the stated seed and harness, or from a run id in `runs/`.

## Two table shapes

**Matrix files** ([`2026-08-20-local-arms.md`](2026-08-20-local-arms.md),
[`2026-08-22-p15-deadline-screen.md`](2026-08-22-p15-deadline-screen.md))
compare arms against each other: one row per arm, a fixed piece cap so every arm
does equal work, and `race` as the ranking metric. Columns follow whatever the
run was testing — the 08-20 rows carry `api err` / `downshifts`, the 08-22 rows
carry `timeouts` / `%≤10s` / `%≤15s` for the bounded-pause deadline. Don't expect
column-for-column uniformity across files; expect the core to be there.

Core columns: `arm`, `race`, `score`, `lines`, `pieces`, `avg holes`,
`decisions`, `illegal`, `late`, `s/decision`, `tok/s`.

**Single-arm files** ([`2026-08-25-vm-demo-gemma3.md`](2026-08-25-vm-demo-gemma3.md))
profile one arm in depth: one row per run id, and metrics a matrix has no room
for (`topped out`, `max h`).

**Validation files** ([`2026-09-03-situation-histogram.md`](2026-09-03-situation-histogram.md))
have no arms at all. They record a check the harness depends on — there, the
class split of the `routed` harness's board classifier over a free
heuristic/random corpus, and the stopping rule (one class ≥ 90 % would have
meant there was nothing to route). They exist so a benchmark that leans on a
component can point at the evidence the component was built on.

## Baseline and goal

The reference screen since 2026-09-03 is **live gravity with the effort pinned**:
seed 1, 30-piece cap, `features`, `--efforts off low medium --fixed-effort`, arms
labeled `+live+fixed`. Its top row is `pi/nemotron-3-super:cloud/features/off` at race
**730**; gpt-oss:120b `low` is 570 and gpt-oss:20b `low` 530. The local reference on the
same screen is `pi/gemma4:26b/features/off` at **530**
([`2026-09-03-local-reasoning-matrix.md`](2026-09-03-local-reasoning-matrix.md)).
The standing goal is a **local** model that matches the cloud top; the Anthropic arms are
paused.

The earlier screen — `--decision-deadline 15` (+p15), the game frozen per piece — is
where the 08-22 and the first 09-03 files live, with `pi/gpt-oss:120b-cloud/features`
at 530 as its reference. It hides speed under 15 s, so its rows rank reasoning quality
only, and every cloud row on it ran at Ollama's default thinking (the flag was not
reaching the model). Compare rows within a screen, never across.

Effort matrices leave out `glm-5.3`, `glm-5.3-flash`, `minimax-m3` and `nemotron-3-ultra`:
as served on 2026-09-03 they ignore `reasoning_effort` / `think:false` and reason anyway
(glm in the visible output, minimax in a thinking block, nemotron-ultra never answers),
so they flatline at every level. Their `off` rows are in
[`2026-09-03-cloud-thinking.md`](2026-09-03-cloud-thinking.md) and, for
glm-5.3-flash, the addendum of the live matrix file. This is a serving/model
behaviour, not a capability verdict; re-probe before assuming it still holds.

Effort matrices carry two columns the harness matrices did not need: the tag's
published rate (`$/M in / out`, from `pricing.MODELS`) and `tok/s` (end-to-end, output
tokens over pi's wall clock — low for terse arms by construction). The same arm can
move ~100 race between sessions on identical settings, so treat ±100 at one seed as
noise.

## Harness comparisons

[`2026-09-03-situation-router.md`](2026-09-03-situation-router.md) is the first
matrix that varies the **harness** on a fixed model set rather than the model
on a fixed harness — `legal`, `features` and `routed` per model — and its
columns carry what that question needs:
`tok/decision` (output), `in tok`, `%≤15s`. Read `routed` rows against the same
model's `features` row, never across models.
[`2026-09-03-local-roster-routed.md`](2026-09-03-local-roster-routed.md)
extends it to the rest of the local roster with the single question "does the
shorter prompt get a flatlined model a decision?".

## Index

| file | shape | question |
|---|---|---|
| [`2026-08-20-local-arms.md`](2026-08-20-local-arms.md) | matrix | first local open-weight rows on the iGPU |
| [`2026-08-22-p15-deadline-screen.md`](2026-08-22-p15-deadline-screen.md) | matrix | every local model plus Ollama cloud under a 15 s deadline; the ≲300 tok/decision viability bar |
| [`2026-08-25-vm-demo-gemma3.md`](2026-08-25-vm-demo-gemma3.md) | single-arm | one arm profiled to game over inside the VM demo |
| [`2026-09-03-situation-histogram.md`](2026-09-03-situation-histogram.md) | validation | class split of the situation classifier over 690 free boards |
| [`2026-09-03-situation-router.md`](2026-09-03-situation-router.md) | matrix (harness) | `routed` vs `legal` vs `features` per model; the cloud baseline |
| [`2026-09-03-local-roster-routed.md`](2026-09-03-local-roster-routed.md) | matrix (harness) | gemma4, qwen 27B, gemma 31B, laguna-xs under `features` and `routed` |
| [`2026-09-03-cloud-roster.md`](2026-09-03-cloud-roster.md) | matrix | every Ollama cloud tag (14 arms) under the 15 s screen at Ollama's default thinking; gemma4 31B plays when served fast, ten frontier tags never answer |
| [`2026-09-03-cloud-thinking.md`](2026-09-03-cloud-thinking.md) | matrix (effort) | gpt-oss:120b at off/low/medium/high, then the roster with thinking off; `low` 570 and deepseek-v4-pro `off` 590 are the new tops, seven flatliners become players |
| [`2026-09-03-local-reasoning-matrix.md`](2026-09-03-local-reasoning-matrix.md) | matrix (effort × model, live, local) | seven local models × {off, low, medium} with power sampled; gemma4:26b `off` 530 is local parity with the cloud 530 cluster |
| [`2026-09-03-live-reasoning-matrix.md`](2026-09-03-live-reasoning-matrix.md) | matrix (effort × model, live) | thirteen cloud models × {off, low, medium} on live gravity with effort pinned; eleven play only at off, nemotron-3-super `off` 730 leads, gpt-oss:20b `low` 530, deepseek-v4-flash `off` 510 |

## Piece caps and what a score means

A matrix run uses a fixed cap deliberately — equal work per arm, bounded cost,
and it makes timeout behavior legible (an arm that flatlines shows its cap as
`pieces`). The cost is that **a capped run's total score is not a result**: stop
at 5 pieces and the score is 0 however well the model played.

So state the cap, and where the question is "how good is this arm", play to game
over and report `topped out`. `scripts/vm-demo.sh` defaults to a 300-piece safety
cap for that reason. Where the question is "which arm is better", keep the cap
and rank on `race`.

## Cost has two halves

`cost_usd` is the API bill, which is $0 for a local `pi/*` arm. That is not the
same as free: the watts move onto your machine, and `Wh` / `energy $` are where
they show up — marginal over an idle baseline, from `tetris_agent.power`.

An unmeasured run reports `n/a`, never `0.0` — unknown and free are different
claims. Local runs measure automatically when a backend is available
(`uv run python -m tetris_agent.power check` says which); on macOS that needs a
sudoers entry for `/usr/bin/powermetrics`.
