# 2026-09-03 — thinking is the variable: the effort curve on gpt-oss:120b, and the cloud roster with thinking off

Follow-up to `2026-09-03-cloud-roster.md`, where ten of fourteen cloud tags lost to the 15 s
clock. Two questions: does more reasoning effort buy a better game, and do the flatliners play
once they stop thinking? Same screen throughout — seed 1, 30-piece cap, `--decision-deadline 15`
(+p15), `features` harness — one `tetris-bench` invocation per (model, effort) so every arm
writes its own JSON.

## First, a harness finding that reframes every cloud row before this file

**Every cloud arm recorded before this file ran at Ollama's default thinking, whatever the bench
asked for.** pi sends a reasoning level to Ollama's OpenAI-compatible endpoint only for models
listed in `~/.pi/agent/models.json` with `reasoning: true`, and sends "off" only when that entry
maps it (`thinkingLevelMap.off = "none"`). None of the cloud tags were listed, so `--thinking`
was silently dropped and each model reasoned as much as it liked. Verified with pi itself on a
toy board: before the fix, `--thinking off` on kimi-k3 ran past 40 s; after listing the tag,
the same call answered in 16 tokens in 1.3 s.

Consequences: the cloud tags are now listed (backup `models.json.bak4`); `pricing.MODELS` marks
them `supports_effort`, so the effort lands in the arm id (`/off`, `/low`, …) and the older
effort-free rows stay distinguishable; and `--efforts off` is a real setting with a one-rung
ladder (`PiPolicy._effort_ladder`), so the controller can never step a model *into* thinking.
Two models ignore the knob regardless: glm-5.3 (and -flash) and minimax-m3 reason in the visible
output until the cap; nemotron-3-ultra never answers at all.

Probes (raw `/api/generate`, toy board: I piece over a 2-deep well) that motivated the runs:
gpt-oss:120b `low` 119 tokens and a wrong answer; `medium` ~1,000 tokens, correct; `high` hit
the 2,000 cap with no answer. kimi-k3, deepseek-v4-pro, qwen3.5 with `think:false`: 10–16
tokens each, one of the three wrong.

## A. The effort curve: gpt-oss:120b-cloud at four settings

| effort | race | score | lines | pieces | timeouts | %≤15s | downshifts | avg holes | tok/decision | s/decision | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off (= Ollama default; gpt-oss ignores "none") | 410 | 260 | 6 | 30 | 4 | 86.7 | 0 | 3.53 | 579 | 6.36 | $0.013 |
| **low** | **570** | **420** | **10** | **30** | **0** | **100.0** | 0 | **0.10** | 101 | **1.52** | $0.006 |
| medium | 490 | 340 | 8 | 30 | 1 | 96.7 | 0 | 2.53 | 301 | 3.36 | $0.009 |
| high | 95 | 0 | 0 | 19 | 9 | 52.6 | 2 | 13.32 | 815 | 19.85 | $0.006 |

For scale: the effort-free 530 row from the router file, at 560 tokens per decision, is the
same setting as `off` here (both are Ollama's default) — 410 vs 530 is the session swing on
identical settings, measured once more. The heuristic control is 490.

**More effort is worse under a clock, and the peak is at the bottom.** `low` placed all 30
pieces in 1.5 s each, cleared 10 lines, left one hole in thirty pieces, and beat the heuristic
solver — at a hundred tokens per decision and half the bill of the default. `medium` was fine.
`high` topped out at 19 pieces with nine kills; the controller downshifted twice but the EMA
climbs one rung per call and the game was lost first. The toy-prompt result ("low is wrong")
did not survive contact with the real board: with the `features` table in front of it the
placement is a lookup, and reasoning past that only spends the clock.

## B. The roster with thinking off

| arm | race | score | lines | pieces | timeouts | %≤15s | avg holes | tok/decision | s/decision | cost | default-thinking row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **deepseek-v4-pro:cloud** | **590** | **440** | **10** | **30** | 0 | 100.0 | **0.07** | 28 | 1.48 | $0.068 | 75 |
| kimi-k2.6:cloud | 510 | 360 | 9 | 30 | 0 | 100.0 | 0.70 | 36 | 4.26 | $0.048 | 50 |
| gemma4:cloud (31B) | 450 | 300 | 7 | 30 | 0 | 100.0 | 0.87 | 30 | 0.96 | $0.007 | 450 |
| minimax-m2.7:cloud | 430 | 280 | 7 | 30 | 5 | 83.3 | 3.77 | 463 | 12.81 | $0.026 | 80 |
| kimi-k3:cloud | 350 | 200 | 5 | 30 | 0 | 100.0 | 5.00 | 40 | 2.30 | $0.104 | 390 |
| qwen3.5:cloud (397B) | 350 | 200 | 5 | 30 | 0 | 100.0 | 5.13 | 33 | 2.27 | $0.032 | 310 |
| glm-5.2:cloud | 350 | 200 | 5 | 30 | 0 | 100.0 | 4.20 | 32 | 5.16 | $0.042 | 85 |
| kimi-k2.7-code:cloud | 310 | 160 | 4 | 30 | 0 | 100.0 | 4.07 | 37 | 2.01 | $0.048 | 65 |
| glm-5.1:cloud | 230 | 80 | 2 | 30 | 0 | 100.0 | 7.37 | 39 | 2.73 | $0.030 | 50 |
| nemotron-3-super:cloud | 190 | 40 | 1 | 30 | 0 | 100.0 | 6.50 | 29 | 2.10 | $0.001 | 60 |
| glm-5.3-flash:cloud | 60 | 0 | 0 | 12 | 10 | 16.7 | 11.42 | 980 | 82.36 | $0.001 | 55 |
| glm-5.3:cloud | 55 | 0 | 0 | 11 | 10 | 9.1 | 12.64 | 1,004 | 157.63 | $0.005 | 50 |
| minimax-m3:cloud | 55 | 0 | 0 | 11 | 10 | 9.1 | 12.64 | 485 | 161.19 | $0.002 | 50 |
| nemotron-3-ultra:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 8.50 | — | — | $0.000 | 50 |

Arm ids: `pi/<tag>/features/off+p15`. The last column is the same tag's row from the roster
file (Ollama default thinking, no flag). `s/decision` on flatlined arms is the timeout artefact.
Runs: `data/benchmarks/benchmark-20260903-075451.json` through `-082827.json`, one per arm.

**Thinking off turned seven flatliners into players.** deepseek-v4-pro went from 75 to 590 —
the best score on any screen so far, 10 lines and two holes in thirty pieces at 28 tokens per
decision. kimi-k2.6 went 50 → 510. glm-5.2 85 → 350, kimi-k2.7-code 65 → 310, glm-5.1 50 → 230,
nemotron-3-super 60 → 190, minimax-m2.7 80 → 430. Ten of fourteen arms placed all 30 pieces
with zero kills. These models were never bad at Tetris; they were slow at *stopping*.

**Thinking off cost nothing for the two that already played.** gemma4 31B scored 450 both ways —
but at 30 tokens per decision instead of 1,582 and 0.96 s instead of 7.9. kimi-k3 350 vs 390,
noise. qwen3.5 350 vs 310, noise, but with zero kills instead of eight.

**The four that ignore the knob are the same four as before.** glm-5.3, glm-5.3-flash,
minimax-m3, nemotron-3-ultra. Nothing on this screen can be done for them short of a prompt
that stops them talking, and the roster file already showed that fast serving does not.

**Cost, now that the flag reaches the model.** With thinking off the whole fourteen-arm pass
billed about $0.42 at published rates; kimi-k3 is a quarter of that by itself for a mid-table
score. deepseek-v4-pro's 590 cost $0.07; gpt-oss:120b's 570 cost less than a cent.

## What this changes

1. **The reference arm should move.** gpt-oss:120b-cloud under `features` at `low` (570, 1.5 s,
   zero kills, $0.006) is the same weights as the standing reference, cheaper, faster, and
   better. deepseek-v4-pro at `off` (590) is a hair above it at ten times the price. Either way
   the parity target is now ~570–590, not 530, and it is cleaner: zero timeouts, so it is a pure
   capability number.
2. **Thinking off is the first thing to try locally.** Every local row on record ran either
   effort-free (Ollama default thinking) or under the `medium` ladder. gemma4 31B just went from
   1,582 tokens per decision to 30 with no loss of score; at the iGPU's 11 tok/s decode that
   is 140 s → 3 s of output per piece. The local `models.json` entries carry `reasoning: true`
   but only nemotron-3.5-lightning-32k maps `off`, so the same silent drop applies locally:
   add `thinkingLevelMap.off = "none"` to the local entries, then re-run gemma4 (8B and 31B),
   gpt-oss:20b, and the incoming `gemma4:26b` at `--efforts off`. Prefill is still the wall for
   the dense 31B under `features`; under `routed` it becomes borderline instead of hopeless.
3. **Effort is a per-model knob with the optimum at the floor for this task.** Not "equal
   thinking" — the vendors' levels are not commensurable and four models ignore them — but
   "least thinking that still reads the table". For gpt-oss that is `low`, not `off`, only
   because gpt-oss has no off.

Single seed. The ±100 swing applies to every row, so deepseek 590 vs gpt-oss 570 vs kimi-k2.6
510 is one tier, not a ranking; 590 vs 75 is not noise.
