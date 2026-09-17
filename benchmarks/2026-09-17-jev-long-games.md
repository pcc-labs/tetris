# 2026-09-17 — Jev, and the 150-piece screen: what 30 pieces was hiding

Two questions in one day. First: how does Jev (TypeSafe's System One model, a typed-choice
API with no generation — see `jev_policy.py`) play against the local and cloud rosters?
Second, once the first answer looked too good: is the 30-piece cap still measuring anything?

It is not. The default cap is now **150** (`DEFAULT_MAX_PIECES`), and the rows below are the
first on that screen: live gravity from level 0, `features`, effort pinned with
`--fixed-effort`, **seeds 1 2 3** (means over three seeds), no pausing anywhere — the game
runs on one thread while the model thinks on another, and a decision that arrives after its
piece locked is discarded and counted `late`.

    uv run [--env-file .env] tetris-bench --models <model> --harnesses features \
      --efforts <off|low> --fixed-effort --seeds 1 2 3 --no-control [--jev-shortlist 5]

One invocation per model; local models one at a time on the iGPU. Jev arms need
`TYPESAFE_API_KEY` (nothing in the project reads `.env` by itself, hence `--env-file`).

## Why the cap moved: 30 pieces is saturated

Same seeds, 30-piece cap:

| arm | race | lines | avg holes | regret | top-1 |
|---|---|---|---|---|---|
| `jev-latest/features` | 570 | 9.7 | 0.66 | 0.114 | 0.56 |
| two-ply lookahead oracle | 557 | 9.7 | 0.21 | 0 | 1.00 |
| heuristic | 497 | 8.3 | 0.78 | 0.068 | 0.66 |
| `pi/gemma4:26b` `off` | 430 | 6.7 | 3.0 | 0.148 | 0.47 |
| `jev-latest/board` | 73 | 0 | 13.9 | 0.247 | 0.05 |

Thirty pieces is 120 cells — at most 12 lines — and everything competent clears ~10 of
them. Jev "beats" the oracle by 13 points of noise. At 150 pieces the oracle/heuristic gap
alone is over 1,000 race points, so placement quality separates. Two more things the short
screen hid: `gemma4:26b` `off` replayed seed 1 at 390 against the 530 recorded on 09-03
(same latency, same `late` — the model answers differently on the same seed, so **single-seed
rows on this benchmark are not rankable**), and `jev-latest/board` stacked every piece in
column 0 (top-1 agreement at chance): Jev does not read the board, it ranks the per-option
consequences the harness computes.

## The 150-piece screen, without Jev in the loop

| arm | race | score | lines | pieces | avg holes | regret | top-1 | top-3 | illegal | late | s/decision | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two-ply oracle (reference) | 8,030 | 7,280 | 54.0 | 150 | 0.31 | 0 | 1.00 | 1.00 | — | — | 0 | free |
| **`pi/gpt-oss:120b-cloud` `low`** | **7,898** | 7,273 | 43.0 | 125 | 2.24 | 0.115 | 0.63 | 0.89 | 0 | 1 | 1.42 | $0.081 |
| heuristic (reference) | 6,903 | 6,153 | 49.7 | 150 | 2.36 | 0.065 | 0.66 | 0.92 | — | — | 0 | free |
| `jev-latest/features` | 2,453 | 1,953 | 24.0 | 100 | 8.33 | 0.153 | 0.50 | 0.77 | 0 | 0 | 0.32 | $0.023 |
| `pi/gpt-oss:20b` `low` (local) | 1,927 | 1,587 | 18.0 | 68 | 3.52 | 0.136 | 0.51 | 0.83 | 0 | 6 | 3.51 | 6.8 Wh |
| `pi/gemma4:26b` `off` (local) | 1,465 | 1,160 | 13.7 | 61 | 7.78 | 0.150 | 0.46 | 0.66 | 0 | 3 | 2.45 | 7.3 Wh |
| `pi/nemotron-3-super:cloud` `off` | 433 | 240 | 6.0 | 39 | 6.49 | 0.153 | 0.43 | 0.70 | 8 | 7 | 2.33 | $0.005 |

Per seed (score @ pieces; anything under 150 pieces topped out):

| arm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| `gpt-oss:120b-cloud` `low` | 12,060 @ 150 | 980 @ 75 | 8,780 @ 150 |
| `jev-latest/features` | 4,080 @ 146 | 980 @ 77 | 800 @ 77 |
| `gpt-oss:20b` `low` | 1,000 @ 68 | 3,520 @ 98 | 240 @ 38 |
| `gemma4:26b` `off` | 140 @ 30 | 3,100 @ 110 | 240 @ 43 |
| `nemotron-3-super:cloud` `off` | 120 @ 33 | 320 @ 44 | 280 @ 39 |

Readings:

- **The 30-piece ranking does not survive.** `nemotron-3-super` `off`, the 730 top row of the
  09-03 live matrix, tops out near piece 39 on every seed (8 illegal, 7 late in ~116
  decisions: one piece in fifteen was the fallback placement). `gemma4:26b` `off`, "local
  parity at 530", is 5× behind `gpt-oss:120b` here.
- **Low reasoning holds up — in the cloud.** `gpt-oss:120b` `low` answers in 1.4 s, was late
  once in 375 decisions up to level 5, and plays at the oracle's race. The local arms at the
  same or lower reasoning do not: their gap is placement quality first (regret 0.136–0.150
  against 0.115, 3.5–7.8 holes against 2.2) and latency second (3–6 late).
- **Jev alone is fast, never illegal, and a weak long-game player.** 0.32 s a decision, zero
  late, and it topped out in all three games — about a third of what the free heuristic scores
  from the same features. Its quality decays as the board gets messy (holes 0.66 → 8.3,
  regret 0.114 → 0.153 from the 30- to the 150-piece screen). Speed was not the constraint
  for anyone but nemotron, so speed bought nothing at a level-0 start.
- **Three seeds is still thin.** Every model arm, the winner included, has one early
  collapse; Jev's seed 1 scored 5,380 and 4,080 on two runs of the same command.

## With Jev and without: `--jev-shortlist 5`

Jev as a pre-filter instead of a player: it rates every legal placement (~0.3 s), only its top
five go into the model's prompt, and the model chooses among those. A failed Jev call hands
the model the full list. Jev's time is inside the arm's latency and its bill inside the arm's
cost. Arms are labeled `+jev5`.

| `pi/gpt-oss:120b-cloud` `low` | race | score | pieces | avg holes | regret | top-1 | top-3 | late | s/decision | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| without Jev | 7,898 | 7,273 | 125 | 2.24 | 0.115 | 0.63 | 0.89 | 1 | 1.42 | $0.081 |
| with Jev (`+jev5`) | 8,593 | 7,987 | 121 | 2.29 | 0.097 | 0.62 | 0.91 | 4 | 2.26 | $0.077 |

With Jev, per seed: 660 @ 64, 8,420 @ 150, 14,880 @ 150 (the best single game of the day).
Each arm has exactly one early top-out, on different seeds, and that accounts for most of
the 9 % race difference — inside the noise. The regret drop (0.115 → 0.097, averaged over
360+ decisions) is the steadier signal that removing weak options helps a little. It is not a
speed-up here: latency rose 0.85 s, more than Jev's own call, and `late` went 1 → 4. (The
iGPU was serving a local arm at the time; some of that may be host contention.)

The local pairs — `gemma4:26b` `off` and `gpt-oss:20b` `low`, each `+jev5` — were still
running when this file was written; they go in a follow-up file. They are the better test:
the local arms are the ones that are slow, prefill-bound and losing pieces to gravity.

## Verdict so far

Do not embed Jev as the player, and do not embed the shortlist yet. Jev is a ranker over
features the harness already computes, and it ranks them worse than the heuristic does
(regret 0.153 against 0.065), so the control that has to run before any decision is a
**heuristic-built top-5 shortlist** — free, instant, no network call — against Jev's. A paid
cloud call inside every local decision also cuts against the standing goal of local parity.
Where Jev fits regardless is the viewer's LABEL deck: a cheap, fast second opinion when
promoting exemplars.

## Also in this change

- `DEFAULT_MAX_PIECES` 30 → 150.
- `--max-seconds`: optional wall-clock cap on a live game, whichever of time or pieces comes
  first (off by default; baselines and `--paused` arms ignore it). Not used for the rows
  above — on an uncapped live screen the clock already shows up as `late`, holes and an
  early top-out.
- `--jev-shortlist K` and `JevShortlist` (`jev_policy.py`).

The nemotron row ran with `--max-seconds 300` set; every game topped out well inside it
(`time_limit_hit` false on all three), so it is the same row the uncapped command produces.

Results: `data/benchmarks/benchmark-20260917-{163526,170524,170826,170850,165646,180712,181655}.json`;
every game is under `runs/20260917-*` and opens in the LABEL deck.
