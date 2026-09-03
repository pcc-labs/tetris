# Benchmarks

Dated results, one file per run day (`YYYY-MM-DD-<subject>.md`, UTC to match run
ids), **append-only**: never rewrite an old file, add a new dated one so progress
stays visible over time.

`tetris-bench` writes `data/benchmarks/*.json` automatically and `data/` is
gitignored, so those results never leave the machine that produced them. This
folder is the committed record. Every row should be reconstructible — from an
arm name plus the stated seed and harness, or from a run id in `runs/`.

## Two table shapes

**Matrix files** (`2026-08-20-local-arms.md`, `2026-08-22-p15-deadline-screen.md`)
compare arms against each other: one row per arm, a fixed piece cap so every arm
does equal work, and `race` as the ranking metric. Columns follow whatever the
run was testing — the 08-20 rows carry `api err` / `downshifts`, the 08-22 rows
carry `timeouts` / `%≤10s` / `%≤15s` for the bounded-pause deadline. Don't expect
column-for-column uniformity across files; expect the core to be there.

Core columns: `arm`, `race`, `score`, `lines`, `pieces`, `avg holes`,
`decisions`, `illegal`, `late`, `s/decision`, `tok/s`.

**Single-arm files** (`2026-08-25-vm-demo-gemma3.md`) profile one arm in depth:
one row per run id, and metrics a matrix has no room for (`topped out`, `max h`).

**Validation files** (`2026-09-03-situation-histogram.md`) have no arms at all. They
record a check the harness depends on — there, the class split of the `routed`
harness's board classifier over a free heuristic/random corpus, and the stopping
rule (one class ≥ 90 % would have meant there was nothing to route). They exist so
a benchmark that leans on a component can point at the evidence the component
was built on.

## Baseline and goal

The reference arm is `pi/gpt-oss:120b-cloud` under `features`: race **530** at seed 1,
30-piece cap, `--decision-deadline 15` (+p15). The standing goal is a **local** model
that matches it; the Anthropic arms are paused. Every 09-03 file is on that screen
so its rows compare with each other and with `2026-08-22-p15-deadline-screen.md` —
with the caveat, measured in the router write-up, that the same arm can move ~100
race between sessions on identical settings, so treat ±100 at one seed as noise.

## Harness comparisons

`2026-09-03-situation-router.md` is the first matrix that varies the **harness** on
a fixed model set rather than the model on a fixed harness — `legal`, `features`
and `routed` per model — and its columns carry what that question needs:
`tok/decision` (output), `in tok`, `%≤15s`. Read `routed` rows against the same
model's `features` row, never across models. `2026-09-03-local-roster-routed.md`
extends it to the rest of the local roster with the single question "does the
shorter prompt get a flatlined model a decision?".

## Index

| file | shape | question |
|---|---|---|
| `2026-08-20-local-arms.md` | matrix | first local open-weight rows on the iGPU |
| `2026-08-22-p15-deadline-screen.md` | matrix | every local model plus Ollama cloud under a 15 s deadline; the ≲300 tok/decision viability bar |
| `2026-08-25-vm-demo-gemma3.md` | single-arm | one arm profiled to game over inside the VM demo |
| `2026-09-03-situation-histogram.md` | validation | class split of the situation classifier over 690 free boards |
| `2026-09-03-situation-router.md` | matrix (harness) | `routed` vs `legal` vs `features` per model; the cloud baseline |
| `2026-09-03-local-roster-routed.md` | matrix (harness) | gemma4, qwen 27B, gemma 31B, laguna-xs under `features` and `routed` |

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
