# 2026-09-03 — the local roster under `routed`: does a shorter prompt get a local model under the clock?

Follow-up to `2026-09-03-situation-router.md`, against the same baseline (`pi/gpt-oss:120b-cloud`
under `features`: race 530). Goal on record: **local plays as well as cloud.** Same screen —
seed 1, 30-piece cap, `--decision-deadline 15` (+p15), `medium` effort where the model accepts
a thinking setting, one model resident on the iGPU at a time, power sampled.

The question: on 08-22 every dense 27–31B model and every gemma flatlined under `features`
(zero decisions inside 15 s; prefill on the ~2k-token prompt was the wall). `routed` sends
about half the prompt. Is that enough to get any of them a decision?

    scripts: scratch local-sweep.sh / laguna-sweep.sh — one `tetris-bench --models pi/<m>
    --harnesses features routed --efforts medium --seeds 1 --max-pieces 30
    --decision-deadline 15 --no-control` per model, evicting between models.

`laguna-xs-32k` is `laguna-xs-2.1` (Poolside Laguna XS 2.1, 33B-A3B MoE, Q4_K_M, 20 GB) with
`PARAMETER num_ctx 32768`, made the same way as the other `-32k` variants.

## Results (seed 1, 30-piece cap, +p15)

| arm | race | score | lines | pieces | decisions | timeouts | %≤15s | illegal | tok/decision | in tok | s/decision | tok/s | Wh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4 (9.6 GB) / features | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 2.99 |
| **gemma4 (9.6 GB) / routed** | **220** | **80** | **2** | **28** | **23** | 5 | 82.1 | 5 | 253 | 28,249 | 8.80 | 45.7 | 3.64 |
| qwen3.6-27b-32k / features | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 3.35 |
| qwen3.6-27b-32k / routed | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 3.63 |
| gemma4-31b-32k / features | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 3.09 |
| gemma4-31b-32k / routed | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 2.54 |
| laguna-xs-32k / features | 50 | 0 | 0 | 10 | 0 | 10 | 0.0 | 10 | — | 0 | — | — | 2.78 |
| laguna-xs-32k / routed | 70 | 0 | 0 | 14 | 3 | 11 | 21.4 | 11 | 784 | 2,374 | 65.5 | 74.7 | 2.88 |

Arm ids as recorded: `pi/gemma4:latest/<harness>+p15`, `pi/qwen3.6-27b-32k/<harness>/medium+p15`,
`pi/gemma4-31b-32k/<harness>+p15`, `pi/laguna-xs-32k/<harness>+p15`. Race 50 with 10 pieces is
the flatline floor: no decision ever arrived, gravity placed every piece. `illegal` counts the
deterministic fallback after a timed-out call, not a wrong answer. `s/decision` on a flatlined
arm is the 150 s of ten timeouts divided by one, an artefact of zero decisions.

For comparison, earlier today at the same settings: `pi/gpt-oss:20b` (13 GB, MoE) 170 under
`features`, 175 under `routed`; the 08-22 baselines heuristic 490, random 140, no-input 55.

## Read

**gemma4 is the only model routing rescued, and it is now the local leader.** Under `features`
it never answered inside 15 s. Under `routed` — a prompt about half the size — it answered 23
of 28 pieces, 82 % of them on time, averaged 8.8 s per decision at 253 output tokens, cleared
two lines, and posted race 220: above random (140), above the local gpt-oss:20b (175), 42 % of
the cloud baseline. Zero decisions to twenty-three is not seed noise.

**The dense 27–31B models did not move.** qwen3.6-27b-32k and gemma4-31b-32k produced no
decision under either harness. Halving the prompt is not enough for them on this box; prefill
remains the wall, as the 08-22 screen said.

**laguna-xs is not the answer here either.** Its MoE shape (3B active) gave 316 tok/s on the
RTX 5090 in the pokemon-kafka roster; on the iGPU it decoded at 75 tok/s and wrote 784 tokens
per answer, so only 3 of 14 decisions landed inside 15 s and it topped out at 14 pieces with
12.8 holes on average. Race 70, below random. A shorter output budget (no thinking is already
off; it is verbose in the answer itself) might change that, but that is a different experiment.

**What this says about the parity goal.** The two local models that play are the two smallest.
The gap to 530 is not going to close by adding larger local models under a 15 s deadline; it
will close, if it does, by changing what gemma4 sees — the `routed-v2` metrics fix (the MOUND
technique says "without opening a hole" but shows no holes number, and gemma4 still took 5
illegal placements) and a shorter board rendering — and by measuring gemma4 across more than
one seed before believing 220.

Runs: `data/benchmarks/benchmark-20260903-044548.json` (gemma4), `-045108.json` (qwen),
`-045625.json` (gemma4-31b), `-050312.json` (laguna). Single seed; ±100 race is within the
session-to-session swing noted in the router write-up.
