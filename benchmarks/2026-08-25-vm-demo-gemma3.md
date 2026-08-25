# 2026-08-25 — `pi/gemma3` in the Lima VM, live gravity

First complete game for the VM demo arm, plus four truncated smoke runs that
preceded it. Recorded because the complete game is the first row where total
score means anything, and because the gap between the two tables is itself the
finding.

**Setup.** Host Apple M4 Max, 36 GiB, macOS 26.5.1. Guest lima `tetris`, 4 vCPU,
12 GiB. `gemma3:latest` (4.3B, Q4_K_M, 3.3 GB) on the **host's** Ollama; the
agent plays in the guest and reaches inference through the tapes capture proxy on
:8092. `--harness chat --effort low --timer-div 0 --level 0 --no-self-heal`.

Run artifacts live **inside the VM** at `~/tetris/runs/<id>/`, not in the host's
`runs/` — the agent process is the guest's. Reach them with
`limactl shell tetris -- cat ~/tetris/runs/<id>/summary.json`.

## The complete game

Played to top-out, no piece cap in the way (`scripts/vm-demo.sh` now defaults to
a 300-piece safety cap, which this run came nowhere near).

| run id | pieces | **total score** | lines | topped out | avg holes | max h | decisions | s/decision | out tok/s | late | illegal / parse | Wh | energy $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260825-014443-a6e6b0` | 11 | **0** | 0 | yes | 18.09 | 17 | 9 | 2.269 | 12.9 | 1 | 0 / 0 | n/a | n/a |

**Total game score: 0 — and this time that is a result, not an artifact.** The
run ended because the stack hit the ceiling (max height 17 of 18) after 11
pieces, not because a counter ran out. gemma3 tops out before it ever clears a
line.

**It degenerates fast.** Average holes over the full game was 18.09, against
5.2–8.6 in the truncated runs below. The early game looks merely scrappy; the
collapse is in the pieces a 5-piece run never reaches.

**The first late decision appeared.** `late: 1`, and 11 pieces locked against
only 9 completed decisions — under a deteriorating board the model does not
always land a placement before gravity does. The truncated runs never showed
this because they never got there.

**Context grows with the game.** 14,922 input tokens for 11 pieces against ~6,100
for 5, since `--harness chat` carries history across pieces. Token counts are not
comparable between the two tables.

## The truncated runs (why they mislead)

Four earlier runs, `--max-pieces` 5 or 6, same seed and settings.

| run id | pieces | score | lines | avg holes | max h | s/decision | out tok/s | late | illegal / parse | Wh | energy $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260825-005559-231986` | 5 | 0 | 0 | 7.2 | 9 | 2.478 | 12.2 | 0 | 0 / 0 | n/a | n/a |
| `20260825-011924-b0f6ab` | 5 | 0 | 0 | 8.6 | 9 | 1.932 | 13.8 | 0 | 0 / 0 | n/a | n/a |
| `20260825-012547-4fb82e` | 5 | 0 | 0 | 5.2 | 10 | 2.085 | 14.2 | 0 | 0 / 0 | n/a | n/a |
| `20260825-012745-b799ab` | 6 | 0 | 0 | 6.5 | 12 | 1.709 | 17.7 | 0 | 0 / 0 | n/a | n/a |
| **total / aggregate** | **21** | **0** | **0** | **6.86** | **12** | **2.035** | **14.37** | **0** | **0 / 0** | **n/a** | **n/a** |

Their total score of 0 measured the piece cap, not the model: a run stopped at
five pieces scores 0 however well it played. (A cap is right for a *matrix* —
equal work per arm, as the 08-20 and 08-22 files use — but wrong when the
question is how good one arm is.) Read them only for the two things a
short run *can* establish — per-decision latency and harness conformance — and
read the complete game for everything else. Aggregates are piece-weighted, not
means of means (out tok/s is 613 output tokens over 42.7 s of generation).

## What holds across both tables

**It keeps up with gravity.** 1.7–2.5 s per decision against a ~15 s window at
level 0. That is the demo's premise: a 4B-class open model decides fast enough to
play live, where a thinking model (`pi/qwen3:8b`) blows the deadline outright.
The single `late` in the complete game is a board-state effect at the end, not a
throughput ceiling.

**The harness contract holds.** Zero illegal placements and zero parse failures
across all 30 decisions. Emitting a well-formed `(rotation, col)` under `chat` is
not one of gemma3's problems.

## What this still does not show

**n=1 for the complete game.** The four truncated runs used an identical piece
sequence (`--timer-div 0`) and still spread 5.2–8.6 on avg holes — 65% on model
nondeterminism alone. One complete game establishes that top-out-before-first-line
happens, not how often. Repeats per seed, then multiple seeds, before any
comparison against another arm.

Nothing here tests `--exemplars`, the lever the repo offers for exactly this
failure. A 0 that comes from topping out at 11 pieces is the baseline to beat.

## Energy: measured as n/a, on purpose

Blank because this host has no readable power sensor without root:

```
$ uv run python -m tetris_agent.power check
[power] powermetrics needs root — grant passwordless sudo to measure
```

`n/a` rather than `0.0` is the point of the change these runs accompanied. A
local arm bills nothing to an API, and reporting that as `$0.0000` beside a cloud
arm's real dollars said "free" when it meant "billed somewhere else". The measured
path (macOS `powermetrics`, Linux `nvidia-smi`/hwmon/RAPL, marginal over an idle
baseline) is built and unit-tested but has **not yet run against real hardware** —
the first row with a number in these columns also validates the sampler.

Grant it with a scoped sudoers line and the columns fill themselves:

```
bdougie ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```
