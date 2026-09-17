# 2026-09-17 — every model alone vs. with Jev: local reaches cloud

Follow-up to [`2026-09-17-jev-long-games.md`](2026-09-17-jev-long-games.md). Same screen:
150-piece cap, live gravity from level 0 (the game never pauses), `features`, effort pinned,
seeds 1 2 3, means over three games. Two arms per model, one invocation each:

    uv run tetris-bench --models <model> --harnesses features --efforts <off|low> \
      --fixed-effort --seeds 1 2 3 --no-control                           # alone
    uv run --env-file .env tetris-bench ... --jev-shortlist 5 --jev-gate 0.7   # with Jev

Local models ran one at a time on the iGPU; cloud arms in a parallel queue.

## "With Jev" means

Jev (TypeSafe System One) is a helper in front of the model, never the player. Per piece:

1. Jev rates every legal placement in one call (~0.2 s) and keeps its top 5 (`+jev5`).
2. If its top pick has probability ≥ 0.7, that move is played; the model is not asked (`g70`).
3. If the piece will land before the model can answer — time to the top of the stack, at the
   game's current gravity, against the median of the model's last 7 answers — Jev's top pick
   is played whatever its probability. A fast decent move beats a late one.
4. Otherwise the model chooses among Jev's five.

Why the gate is trusted: on 640 graded Jev decisions from earlier in the day, p ≥ 0.7 covered a
quarter of moves at regret 0.047 (better than any model arm), p < 0.5 half of them at 0.19.

## The pairs (race score; pieces survived of 150)

| model | where | alone | with Jev | pieces alone → with Jev | model's share of moves with Jev |
|---|---|---|---|---|---|
| gemma4:31b `off` | cloud | 595 | **6,222** | 47 → 136 | 67 % |
| qwen3.5 `off` | cloud | 1,795 | **6,137** | 78 → 139 | — |
| deepseek-v4-pro `off` | cloud | 693 | **6,060** | 51 → 141 | 72 % |
| laguna-xs-32k `off` | **local** | 293 | **5,952** | 32 → 141 | 67 % |
| gpt-oss:20b `low` | **local** | 1,927 | **5,810** | 68 → 138 | 62 % |
| gemma4:26b `off` | **local** | 1,465 | **5,768** | 61 → 138 | 61 % |
| nemotron-3-super `off` | cloud | 433 | 4,072 | 39 → 118 | — |
| glm-5.2 `off` | cloud | 1,005 | 2,825 | 62 → 101 | — |
| kimi-k3 `low` | cloud | 605 | 928 | 41 → 60 | 49 % |
| glm-4.7-flash-32k `off` | **local** | 193 | 752 | 28 → 60 | 61 % |
| nemotron-3.5-lightning-32k `off` | **local** | 88 | 583 | 18 → 51 | 57 % |
| gemma4 e4b (`latest`) `off` | **local** | 147 | 432 | 24 → 45 | 67 % |
| gpt-oss:120b `low` | cloud | 7,898 | rerun pending | 125 → ? | |
| gpt-oss:20b `low` | cloud | 3,328 | rerun pending | 72 → ? | |

References on the same seeds: two-ply oracle 8,030 (never loses), heuristic 6,903 (never loses).

Readings:

- **Every pair improved — 12 of 12, by 1.5× to 20×.** No model got worse with Jev.
- **Local reaches cloud.** The three best local arms with Jev (5,768–5,952) sit with the three
  best cloud arms with Jev (6,060–6,222). Alone, local was 4–5× behind cloud. Two of three
  games reach piece 150 for each of them; alone none did.
- **The model still plays.** In the strong pairs the model makes ~⅔ of the moves, Jev about a
  quarter on confidence, and the out-of-time rescue fires on under 10 %. In the weak pairs the
  rescue fires on 20–35 % — the stack is tall because the model's own picks are poor — and
  those are Jev's unsure moves, so the arm improves but still loses.
- **Late pieces mostly vanish.** 6–11 per arm alone; 0–4 with Jev. Seconds per move roughly
  halve (gemma4:26b 2.5 → 1.2, gpt-oss:20b 3.5 → 2.4).
- **A ceiling near 6,000.** The with-Jev scores cluster just under the heuristic's 6,903, and
  Jev ranks the same per-option features the heuristic scores. Whether the ceiling is Jev's or
  the model's is the open question: the gpt-oss:120b rerun (7,898 alone) and a
  heuristic-built shortlist are the two checks that answer it.
- **Three seeds.** Per-game spread is still wide (qwen3.5 with Jev: 2,560 / 7,000 / 6,760).
  The direction is consistent across 12 models; the sizes are not precise.

## Two rows that were thrown out, and why

The first version of rule 3 estimated the model's speed from a running average that included
failed and slow calls, and once Jev took a move the estimate never refreshed. One slow network
call latched Jev on: gpt-oss:120b-cloud with Jev played 61 % of moves by Jev (race 5,283, holes
7.6), gpt-oss:20b-cloud 99 % (2,100) — Jev alone, which loses. Fixed the same evening (median
of the last 7 *successful* answers, and no rescue until three are in); both pairs rerun with
the fix, results to be appended here.

## Spend

Cloud arms ~$4 for the day, Jev ~$0.40. Jev's price is not a consideration in this project.

Results: `data/benchmarks/benchmark-20260917-2*.json`; games under `runs/20260917-*`.
