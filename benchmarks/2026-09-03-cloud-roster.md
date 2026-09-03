# 2026-09-03 — the Ollama cloud roster: is 530 the open-weight ceiling, and can gemma's 31B play?

Two batches, same screen. First three arms chosen to answer local questions; then, once the
cloud tier proved open, the other eleven tags the account can run.

Same screen as every 09-03 file — seed 1, 30-piece cap, `--decision-deadline 15` (+p15),
`features` harness, no effort flag (unlisted pi models never get `--thinking`, same as the
gpt-oss cloud rows) — against the reference arm `pi/gpt-oss:120b-cloud/features` at race 530.

Why these three, when the goal on record is *local* parity: Ollama's cloud tier now runs every
tag this account tried (on 08-22 the frontier tags rejected the pull), and cloud copies answer
two local questions cheaply. `gemma4:cloud` is the same 31B weights as `gemma4-31b-32k`, which
has never produced a decision on the iGPU — does the 31B play at all when serving is fast?
`nemotron-3-super:cloud` is 120B-A12B with an 87 GB local tag — worth a GTT bump and the pull?
`kimi-k3:cloud` has no local counterpart; it is here to ask whether 530 is the ceiling.

    uv run tetris-bench --models pi/kimi-k3:cloud pi/nemotron-3-super:cloud pi/gemma4:cloud
      --harnesses features --efforts medium --seeds 1 --max-pieces 30 --decision-deadline 15
      --no-control --no-power

`gemma4:cloud` resolves to the 31B (32.7B params, BF16). The run's JSON records `cost_usd` 0.00
because, at the time, every `pi/` arm was priced as free. That was wrong for cloud tags — they
bill the account's Ollama credits — and `pricing.MODELS` now carries each tag's published
"Cost /1M tokens" rate (an unlisted cloud tag is refused rather than priced at $0). The `cost`
column below is recomputed from the recorded token counts at those rates.

## Results (seed 1, 30-piece cap, +p15, `features`)

| arm | race | score | lines | pieces | timeouts | %≤15s | decisions | illegal | avg holes | tok/decision | in tok | s/decision | tok/s | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:120b-cloud (09-03 reference) | 530 | 380 | 9 | 30 | 0 | 100.0 | 30 | 0 | 0.30 | 560 | 30,795 | 2.93 | 191.3 | $0.015 |
| heuristic (08-20 control) | 490 | 340 | 8 | 30 | — | — | — | — | — | — | — | — | — | — |
| **gemma4:cloud (31B)** | **450** | **300** | **7** | **30** | 1 | 96.7 | 29 | 1 | 0.17 | 1,582 | 47,668 | 7.87 | 215.2 | $0.025 |
| kimi-k3:cloud | 390 | 240 | 6 | 30 | 2 | 93.3 | 28 | 2 | 2.37 | 233 | 26,693 | 4.86 | 61.4 | $0.178 |
| random (08-20 floor) | 140 | 0 | 0 | 28 | — | — | — | — | — | — | — | — | — | — |
| nemotron-3-super:cloud (120B-A12B) | 60 | 0 | 0 | 12 | 10 | 16.7 | 2 | 10 | 10.50 | 620 | 4,310 | 86.73 | 53.1 | $0.001 |

Arm ids as recorded: `pi/<tag>/features+p15`. `illegal` counts the deterministic fallback after
a timed-out call, not a wrong answer. `tok/s` is the harness's end-to-end rate (output tokens
over wall time), not the provider's decode rate. `s/decision` on nemotron is ten 15 s timeouts
divided by two decisions, an artefact of the flatline. pi reported zero reasoning tokens for
all three, so gemma4's 1,582 tokens per decision are visible output, not a hidden thinking
block.

`cost` is at each tag's published rate (kimi-k3 $3/$15 per M in/out; gemma4 $0.14/$0.40;
nemotron-3-super $0.015/$0.60; gpt-oss:120b $0.15/$0.60); no cached-input tokens were reported.
kimi-k3 is 7× the reference arm's bill for a worse score, which is the only cost figure on this
screen that changes a decision.

Per-decision latencies (s): gemma4 ranged 3.0–15.0 with one kill; kimi 1.7–15.0 with two
kills and a median near 3 s; nemotron 10.7, 12.7 and ten kills.

## Read

**The gemma4 31B can play. Its local copy cannot, and the reason is now measured.** Same
weights that flatlined twice on the iGPU (08-22 under `features`, 09-03 under `routed`) posted
race 450 with fast serving: 7 lines, 0.17 holes per piece (cleaner than the reference arm's
0.30), 29 of 30 decisions inside the clock. So the 31B is not capability-bound; it is
speed-bound, and by a wide margin — it wrote 1,582 tokens per decision. At the iGPU's measured
11 tok/s decode and 64 tok/s prefill for this model (08-22 probes), one decision is over two
minutes. No prompt trim closes that; `routed` halves the prefill term and leaves the output
term untouched.

**The family lever is the 26B MoE, not the 31B.** The gemma4 line ships a `26b-a4b` tag: 26B
total, 4B active, 19 GB. That is the gpt-oss shape (few active parameters, which is what sets
decode speed on this box) inside the family whose 31B just scored 450 and whose 8B (`gemma4:
latest`, e4b) is the current local leader at 220 under `routed`. Untested here; it is the next
local pull. `gemma4:12b` (7.6 GB, dense) is the cheaper intermediate.

**kimi-k3 is the disciplined one.** 233 tokens per decision — under the ≲300 bar, less than
half of gpt-oss:120b's 560 — and 28 of 30 on time, yet race 390 with 2.4 holes per piece. Fast
and terse, and it plays worse than the two verbose arms above it. Token budget bounds who can
answer; it does not decide who answers well.

**nemotron-3-super is a no.** Two decisions in twelve pieces, ten kills, topped out. 620
tokens per decision at 53 tok/s end-to-end is ~12 s before prefill, so nearly every call hit
the 15 s wall. That settles the local question it was pulled to answer: do not spend the GTT
bump and the 87 GB download on it. (`nemotron-3-ultra:cloud` took 26 s on a one-word probe
and was not run.)

**Is 530 the ceiling?** Not settled. kimi-k3 landed under it and under the heuristic; gemma4
31B landed between them. Nothing on this screen beat gpt-oss:120b-cloud, which stays the
reference arm. glm-5.3 and deepseek-v4-pro answered one-word probes quickly and are the
remaining candidates if the ceiling question is worth more credits; on the parity goal it is
not, and the credits are better spent on nothing — the next row is local.

## What this changes

Ranking the parity levers after this file, in order: (1) `gemma4:26b` (the 4B-active MoE)
under `features` and `routed` on the iGPU, with the standalone prefill/decode probe first;
(2) `gemma4:12b`; (3) the `routed-v2` prompt work for whichever of those plays. The dense 31B
is off the local list for good — 450 is what it would score if it could answer, and it cannot.

Run: `data/benchmarks/benchmark-20260903-060918.json`. Single seed; ±100 race at one seed is
within the session-to-session swing measured in the router write-up, so 450 vs 390 is not a
ranking between gemma4 and kimi, but both vs nemotron's 60 is.

---

## Second batch, same day: the other eleven cloud tags

Same screen (seed 1, 30-piece cap, +p15, `features`, no `--thinking` flag). Run one
`tetris-bench --models pi/<tag>` per model rather than one matrix: a first attempt as a single
eleven-arm matrix was killed at nine arms by a 10-minute tool ceiling and, because the bench
writes its JSON only when the whole matrix ends, all nine rows were lost. One invocation per
model gives one JSON per arm as it finishes. (The harness note: a matrix should checkpoint per
arm.) The arms run in the order listed in the scratch script; results below are sorted by race.

    for m in deepseek-v4-pro:cloud glm-5.3:cloud glm-5.3-flash:cloud qwen3.5:cloud minimax-m3:cloud
             kimi-k2.7-code:cloud kimi-k2.6:cloud glm-5.2:cloud glm-5.1:cloud minimax-m2.7:cloud
             nemotron-3-ultra:cloud; do
      uv run tetris-bench --models pi/$m --harnesses features --efforts medium --seeds 1
        --max-pieces 30 --decision-deadline 15 --max-usd 10 --no-control --no-power
    done

| arm | race | score | lines | pieces | timeouts | %≤15s | decisions | illegal | avg holes | tok/decision | in tok | s/decision | tok/s | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **qwen3.5:cloud (397B)** | **310** | **160** | **4** | **30** | 8 | 73.3 | 22 | 8 | 8.80 | 486 | 35,574 | 12.66 | 67.6 | $0.060 |
| glm-5.2:cloud | 85 | 0 | 0 | 17 | 10 | 41.2 | 7 | 10 | 12.06 | 298 | 6,602 | 27.05 | 53.1 | $0.018 |
| minimax-m2.7:cloud | 80 | 0 | 0 | 16 | 6 | 62.5 | 10 | 6 | 6.19 | 484 | 15,660 | 20.32 | 42.8 | $0.011 |
| deepseek-v4-pro:cloud | 75 | 0 | 0 | 15 | 12 | 20.0 | 3 | 12 | 9.07 | 1,218 | 4,730 | 67.72 | 158.6 | $0.021 |
| kimi-k2.7-code:cloud | 65 | 0 | 0 | 13 | 12 | 7.7 | 1 | 12 | 8.62 | 1,408 | 1,376 | 193.12 | 108.4 | $0.007 |
| glm-5.3-flash:cloud | 55 | 0 | 0 | 11 | 10 | 9.1 | 1 | 10 | 12.64 | 408 | 756 | 154.01 | 104.7 | $0.000 |
| glm-5.3:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 0 | 10 | 8.50 | — | 0 | — | — | $0.000 |
| minimax-m3:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 0 | 10 | 8.50 | — | 0 | — | — | $0.000 |
| kimi-k2.6:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 0 | 10 | 8.50 | — | 0 | — | — | $0.000 |
| glm-5.1:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 0 | 10 | 8.50 | — | 0 | — | — | $0.000 |
| nemotron-3-ultra:cloud | 50 | 0 | 0 | 10 | 10 | 0.0 | 0 | 10 | 8.50 | — | 0 | — | — | $0.000 |

Race 50 with 10 pieces is the flatline floor. `cost` is at the published rates now in
`pricing.MODELS`, from the tokens pi reported — and pi reports no usage for a call killed at
15 s, so the five zero-decision arms show $0.000 for runs that did bill credits (ten
15-second generations each). Treat every cost in this table as a floor. Runs:
`data/benchmarks/benchmark-20260903-065817.json` through `-072800.json`, one per arm.

### Read

**One of eleven played.** qwen3.5 at 397B placed all 30 pieces, cleared 4 lines, and posted
race 310 at 486 tokens per decision — third among cloud arms behind gemma4 31B (450) and kimi-k3
(390), and 8 of its 30 calls still hit the wall. Everything else on this batch finished below
the random floor (140).

**Ten frontier tags lost to the clock, not to the board.** Five of them (glm-5.3, glm-5.1,
minimax-m3, kimi-k2.6, nemotron-3-ultra) never produced a single token pi could read in 15 s:
every call killed, `output_tokens` 0. Three more (deepseek-v4-pro, kimi-k2.7-code,
glm-5.3-flash) got one to three answers out, at 400–1,400 tokens each. These models default to
thinking, and no arm on this screen is sent a `--thinking` flag; the reasoning runs past the
deadline and the JSON never arrives. That is the deepseek-v4-flash story from 08-22 repeated ten
times: fast serving cannot save a model that will not stop talking. It is also why the 08-22
viability bar (≲300 tokens per decision) was written.

**The ceiling question, answered for this screen.** Fourteen cloud tags have now run at seed 1,
30 pieces, `features`, 15 s. Ranked: gpt-oss:120b 530, gemma4 31B 450, kimi-k3 390, qwen3.5
397B 310, gpt-oss:20b 170, then everything else at or under 85. Nothing beat the reference arm,
and the order is not a capability ranking: it is a ranking of who answers inside 15 seconds.
The two models above 400 both answered every piece; the frontier models with the strongest
reputations answered none.

**What would change the frontier rows is not more credits.** It is `--thinking off` (or `low`)
for these tags — `PiPolicy` already has the ladder, gated on `supports_effort`, which is `False`
for every cloud entry so the recorded rows stay reproducible. Flipping it for one flatliner and
re-running is a cheap, separate experiment. Until then these rows say what a default-configured
frontier model does under a 15 s clock, which is nothing.

**For the parity goal, nothing here moves the target.** 530 stands. The local levers from the
first section are unchanged: `gemma4:26b` (26B-A4B), then `gemma4:12b`, then `routed-v2`.

Recorded spend for the batch: $0.12 at published rates, plus the undercounted kills, plus a
similar amount for the lost first attempt.
