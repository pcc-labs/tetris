# 2026-09-03 — the reasoning-level matrix on live gravity: twelve cloud models × {off, low, medium}

The benchmark's axis is now the reasoning level. The bounded-pause screen (`+p15`) freezes the
game while the model thinks, so a 1.5 s answer and a 14 s answer play the same game and the rows
can only rank reasoning quality. This file runs on **live gravity** — no pause, level 0, the
piece falls while the model thinks (~15 s from spawn to floor), a decision that lands after the
piece locks is `late` — so thinking time costs board position exactly as it would for a person.

Same seed 1, 30-piece cap, `features` harness. Effort **pinned** (`--fixed-effort`, new today;
arms carry `+fixed`): the deadline controller may not step a `low` arm down to `off` mid-game,
so each row is the level it says it is. Twelve cloud tags that obey the thinking flag; the four
that ignore it (glm-5.3, glm-5.3-flash, minimax-m3, nemotron-3-ultra) are left out. One
`tetris-bench` invocation per (model, effort), 36 arms, $0.94 of credits at published rates.

    uv run tetris-bench --models pi/<tag> --harnesses features --efforts <off|low|medium>
      --fixed-effort --seeds 1 --max-pieces 30 --max-usd 10 --no-control --no-power

`off` is Ollama's `reasoning_effort: "none"`; gpt-oss has no off and reasons at its default
there. The per-call timeout on live gravity is twice the piece's remaining fall time, so a slow
level loses pieces to gravity rather than having decisions discarded.

## The curves (race score; bold ≥ 400)

| model | off | low | medium |
|---|---|---|---|
| nemotron-3-super (120B-A12B) | **730** | 55 | 55 |
| kimi-k3 | **510** | **590** | **550** |
| gpt-oss:120b | **510** (default) | **570** | 390 |
| kimi-k2.7-code | **550** | 55 | 55 |
| gpt-oss:20b | 55 (default) | **530** | 70 |
| deepseek-v4-pro | **530** | 55 | 50 |
| gemma4 (31B) | **530** | 60 | 75 |
| glm-5.2 | **490** | 55 | 55 |
| qwen3.5 (397B) | **450** | 55 | 80 |
| glm-5.1 | 390 | 55 | 55 |
| kimi-k2.6 | 200 | 55 | 55 |
| minimax-m2.7 | 55 | 55 | 55 |

Reference points from 08-20 on the same live screen: heuristic 490, random 140, no-input 55.
Race 55 with 11 pieces is the live flatline floor.

## Every arm, ranked

| arm | race | score | lines | pieces | late | avg holes | tok/decision | s/decision | tok/s | $/M in / out | run cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **nemotron-3-super / off** | **730** | 580 | 10 | 30 | 0 | 0.20 | 28 | 2.38 | 11.7 | $0.015 / $0.60 | $0.001 |
| kimi-k3 / low | 590 | 440 | 10 | 30 | 2 | 2.27 | 268 | 4.45 | 60.2 | $3.000 / $15.00 | $0.226 |
| gpt-oss:120b-cloud / low | 570 | 420 | 10 | 30 | 0 | 0.13 | 107 | 1.40 | 76.6 | $0.150 / $0.60 | $0.006 |
| kimi-k3 / medium | 550 | 400 | 10 | 30 | 0 | 0.10 | 182 | 3.43 | 53.0 | $3.000 / $15.00 | $0.180 |
| kimi-k2.7-code / off | 550 | 400 | 10 | 30 | 0 | 2.27 | 34 | 2.30 | 15.0 | $0.950 / $4.00 | $0.049 |
| gpt-oss:20b-cloud / low | 530 | 380 | 9 | 30 | 0 | 1.30 | 134 | 2.95 | 45.3 | $0.070 / $0.30 | $0.003 |
| deepseek-v4-pro / off | 530 | 380 | 9 | 30 | 0 | 0.23 | 28 | 1.39 | 20.4 | $1.320 / $3.96 | $0.067 |
| gemma4 / off | 530 | 380 | 9 | 30 | 0 | 1.57 | 30 | 1.31 | 22.6 | $0.140 / $0.40 | $0.007 |
| gpt-oss:120b-cloud / off | 510 | 360 | 8 | 30 | 4 | 2.63 | 607 | 5.22 | 160.6 | $0.150 / $0.60 | $0.014 |
| kimi-k3 / off | 510 | 360 | 9 | 30 | 0 | 1.60 | 36 | 1.90 | 19.2 | $3.000 / $15.00 | $0.111 |
| glm-5.2 / off | 490 | 340 | 8 | 30 | 0 | 2.07 | 34 | 2.16 | 15.6 | $1.400 / $4.40 | $0.041 |
| qwen3.5 / off | 450 | 300 | 7 | 30 | 0 | 2.93 | 32 | 2.11 | 15.1 | $0.600 / $3.60 | $0.033 |
| gpt-oss:120b-cloud / medium | 390 | 240 | 6 | 30 | 2 | 2.10 | 671 | 4.13 | 162.4 | $0.150 / $0.60 | $0.017 |
| glm-5.1 / off | 390 | 240 | 6 | 30 | 1 | 3.80 | 38 | 2.87 | 13.2 | $1.000 / $3.20 | $0.028 |
| kimi-k2.6 / off | 200 | 80 | 2 | 24 | 3 | 5.46 | 36 | 4.56 | 8.0 | $0.950 / $4.00 | $0.031 |
| qwen3.5 / medium | 80 | 0 | 0 | 16 | 3 | 6.44 | 563 | 13.58 | 70.8 | $0.600 / $3.60 | $0.025 |
| gemma4 / medium | 75 | 0 | 0 | 15 | 2 | 7.93 | 1998 | 8.79 | 227.3 | $0.140 / $0.40 | $0.009 |
| gpt-oss:20b-cloud / medium | 70 | 0 | 0 | 14 | 3 | 4.86 | 844 | 12.73 | 88.7 | $0.070 / $0.30 | $0.002 |
| gemma4 / low | 60 | 0 | 0 | 12 | 4 | 11.25 | 2142 | 10.04 | 213.4 | $0.140 / $0.40 | $0.008 |
| gpt-oss:20b-cloud / off | 55 | 0 | 0 | 11 | 3 | 12.82 | 565 | 28.89 | 69.4 | $0.070 / $0.30 | $0.001 |
| deepseek-v4-pro / low | 55 | 0 | 0 | 11 | 2 | 7.54 | 1400 | 15.18 | 148.3 | $1.320 / $3.96 | $0.023 |
| kimi-k2.6 / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 0 | 69.31 | 0.0 | $0.950 / $4.00 | $0.000 |
| kimi-k2.6 / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 0 | 69.31 | 0.0 | $0.950 / $4.00 | $0.000 |
| kimi-k2.7-code / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 0 | 69.31 | 0.0 | $0.950 / $4.00 | $0.000 |
| kimi-k2.7-code / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 521 | 72.06 | 82.9 | $0.950 / $4.00 | $0.003 |
| glm-5.2 / low | 55 | 0 | 0 | 11 | 4 | 9.00 | 306 | 32.91 | 55.1 | $1.400 / $4.40 | $0.005 |
| glm-5.2 / medium | 55 | 0 | 0 | 11 | 5 | 14.09 | 480 | 16.17 | 55.3 | $1.400 / $4.40 | $0.013 |
| glm-5.1 / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 545 | 72.05 | 25.6 | $1.000 / $3.20 | $0.003 |
| glm-5.1 / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 0 | 69.31 | 0.0 | $1.000 / $3.20 | $0.000 |
| minimax-m2.7 / off | 55 | 0 | 0 | 11 | 4 | 12.73 | 626 | 20.98 | 39.2 | $0.300 / $1.20 | $0.003 |
| minimax-m2.7 / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 460 | 19.69 | 36.0 | $0.300 / $1.20 | $0.003 |
| minimax-m2.7 / medium | 55 | 0 | 0 | 11 | 4 | 11.36 | 543 | 12.64 | 42.9 | $0.300 / $1.20 | $0.007 |
| qwen3.5 / low | 55 | 0 | 0 | 11 | 5 | 12.73 | 502 | 34.37 | 71.7 | $0.600 / $3.60 | $0.006 |
| nemotron-3-super / low | 55 | 0 | 0 | 11 | 4 | 7.82 | 464 | 23.20 | 61.0 | $0.015 / $0.60 | $0.001 |
| nemotron-3-super / medium | 55 | 0 | 0 | 11 | 4 | 12.73 | 1451 | 70.40 | 83.2 | $0.015 / $0.60 | $0.001 |
| deepseek-v4-pro / medium | 50 | 0 | 0 | 10 | 3 | 9.20 | 1399 | 30.27 | 146.8 | $1.320 / $3.96 | $0.015 |

Arm ids: `pi/<tag>/features/<effort>+live+fixed`. A `tok/decision` of 0 with ~69 s per
decision is four killed calls and no usage reported — pi returns nothing for a call it kills,
so those rows' costs are floors. Runs: `data/benchmarks/benchmark-20260903-131853.json`
through `-141957.json`, one per arm, in the order of the model list above with efforts
off/low/medium.

## Read

**Ten of twelve models play at exactly one level, and it is off.** Any thinking at all —
`low` included — puts the decision at 13 to 70 s and the piece is on the floor before the
answer arrives. On the bounded-pause screen the same `low` rows would have been discarded
decisions and a bad score; here they are a topped-out board by piece eleven. The two exceptions
are the two whose serving is fast enough to afford a few hundred tokens: kimi-k3 plays at all
three levels (510 / 590 / 550, one tier within noise) and gpt-oss:120b at its default and `low`
(510 / 570) but not `medium` (390, two late pieces).

**So: does some reasoning help?** For those two, yes, by a margin inside the ±100 single-seed
swing — kimi-k3 `low` 590 over `off` 510, gpt-oss `low` 570 over default 510 — and only because
`low` costs them under 5 s. For everyone else the answer is not "no" but "unaffordable": we do
not know whether deepseek-v4-pro's placements would improve with 1,400 tokens of thought,
because the piece did not wait to find out. The cutoff is sharp and it is a latency, not a
token count: every arm under ~3 s per decision placed all 30 pieces; every arm over ~9 s
topped out; the band between (kimi-k3 `low` at 4.5 s with two late, kimi-k2.6 `off` at 4.6 s
losing six pieces) is where the game is decided.

**nemotron-3-super at off is the best row on any screen: 730.** Ten lines, 580 points (there
are multi-line clears in there), 0.2 holes per piece, 2.4 s per decision, $0.001. This morning's
roster file called it "a no" on a default-thinking flatline and said not to spend the GTT bump
and the 87 GB local pull on it. That verdict was wrong and is withdrawn here: it is the same
120B-A12B whose local tag was ruled out for the wrong reason. Its 190 on the bounded-pause
screen a few hours earlier, same setting, is unexplained — that run had a parse failure and
6.5 holes; this one had neither — and is a reminder that one seed is one seed.

**Live gravity reorders the middle of the table.** Compared with the thinking-off rows on the
pause screen: gemma4 450 → 530, nemotron 190 → 730, deepseek 590 → 530, kimi-k3 350 → 510 —
noise or better for the fast ones. kimi-k2.6 510 → 200 and minimax-m2.7 430 → 55 — the two that
answered in 4.6 and 12.8 s were fine when the game waited and are not when it does not.
That is the point of running live.

**gpt-oss:20b at low is 530.** Same race as the standing 120b reference, from the 21B model
that runs on this iGPU. At its default it flatlined (29 s per decision); at `medium` it topped
out; at `low` it answered every piece in 3 s at 134 tokens. This is the local row that matters
most from this file.

**Cost.** $0.94 for 36 arms; kimi-k3 alone was $0.52 of it and did not buy the top score.
nemotron's 730 cost a tenth of a cent.

## What this changes for the parity goal

The target on the live screen is nemotron-3-super `off` at 730, with kimi-k3 `low`, gpt-oss:120b
`low`, and a cluster of `off` rows at 530–590 behind it. The local levers, in order:

1. **gpt-oss:20b local at `low`, fixed, live.** Cloud 530. Locally: ~134 output tokens at
   52 tok/s is ~2.6 s; prefill of the ~2 k-token `features` prompt at 246 tok/s is ~8 s;
   ~11 s against a ~15 s fall. Tight under `features`, comfortable under `routed` (half the
   prefill). Run both.
2. **gemma4 local (8B) at `off`.** Cloud 31B `off` is 530 at 30 tokens per decision. The 8B's
   prefill is 451 tok/s, so ~4.5 s per piece all-in. Needs `thinkingLevelMap.off = "none"`
   on the local `gemma4` entry in `~/.pi/agent/models.json` first — the cloud fix, applied
   locally — or the flag is dropped exactly as it was for the cloud tags.
3. **nemotron-3-super local (87 GB, GTT bump) at `off`.** 28 tokens per decision makes decode
   irrelevant; prefill at 12B active is the unknown. Probe before pulling.

Single seed throughout; ±100 is noise; 730 vs 55 is not.

---

## Addendum: the flash tags, queued behind the matrix

Same screen and command. `deepseek-v4-flash` had one row on record — 08-22, 1,911 tokens per
decision, 10 kills in 14 pieces — which we now know was Ollama's default thinking with the flag
dropped. `glm-5.3-flash` ignores the flag; one `off` row confirms that on live gravity.

| arm | race | score | lines | pieces | late | avg holes | tok/decision | s/decision | tok/s | $/M in / out | run cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v4-flash / off | 510 | 360 | 9 | 30 | 0 | 0.10 | 31 | 1.23 | 25.3 | $0.440 / $1.32 | $0.022 |
| deepseek-v4-flash / medium | 60 | 0 | 0 | 12 | 3 | 11.92 | 2104 | 22.87 | 173.4 | $0.440 / $1.32 | $0.010 |
| deepseek-v4-flash / low | 55 | 0 | 0 | 11 | 4 | 12.73 | 2079 | 29.16 | 218.5 | $0.440 / $1.32 | $0.007 |
| glm-5.3-flash / off | 55 | 0 | 0 | 11 | 3 | 9.82 | 653 | 60.45 | 114.3 | $0.150 / $0.50 | $0.000 |

Runs: `benchmark-20260903-142225.json` (off), `-142351` (low), `-142518` (medium),
`-142635` (glm-5.3-flash).

**deepseek-v4-flash at off plays like its pro sibling** — 510 vs 530, 0.10 holes per piece,
1.2 s per decision, at a third of the price. The 08-22 verdict ("1,911 tokens per decision
fits no 15-second clock at any serving speed") described the default thinking, not the model.
At `low` and `medium` it writes ~2,100 tokens per decision and tops out, the same curve as
every other off-only model. **glm-5.3-flash** reasons in its visible output whatever the flag
says (653 tokens, 60 s per decision) and flatlines; nothing to do for it on this harness.

Thirteen models now have a full off/low/medium curve on live gravity: eleven play only at
`off`, kimi-k3 and gpt-oss:120b also at `low`.
