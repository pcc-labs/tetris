# 2026-09-03 — situation histogram over a free corpus

Validation for the situation router (docs/superpowers/specs/2026-09-02-situation-router-design.md,
section 06). No per-decision traces existed on disk, so the corpus is generated:
`heuristic` and `random` runs, seeds 0–3, 150-piece cap, paused/headless, replayed into
verified boards by `traces.mine_run` and classified with default thresholds
(`topping_out_height=12`, `mound_rise=3`, `well_depth=4`).

    uv run tetris-situations generate --seeds 0 1 2 3 --max-pieces 150
    uv run tetris-situations histogram

Heuristic play yields well-formed boards; random play yields messy ones. Neither is a model
arm's distribution — this measures the classifier's coverage and priority order, not its
effect on play.

Note: the run day rolled past midnight UTC between task setup and execution — the brief was
dated 2026-09-02, but `date -u +%F` read 2026-09-03 at run time, so this file and its title use
the actual run date per the brief's own instruction.

## Results

| kind | heuristic | random | total | share |
|---|---|---|---|---|
| TOPPING_OUT | 0 | 10 | 10 | 9.7% |
| TETRIS_READY | 0 | 0 | 0 | 0.0% |
| HOLE_RISK | 1 | 3 | 4 | 3.9% |
| MOUND | 24 | 31 | 55 | 53.4% |
| FLAT | 25 | 9 | 34 | 33.0% |
| total | 50 | 53 | 103 | 100.0% |

Boards classified: 103. Pieces per run: heuristic 150 each (seeds 0–3, all four ran to the
150-piece cap without topping out); random 15, 27, 26, 22 (seeds 0–3 respectively, each run
topping out before the cap).

## Verdict

Dominant class: none. Stopping rule (≥ 90 % in one class): not triggered — proceed to the
routed benchmark.

Observations: TETRIS_READY never fires for either policy across all 103 boards — neither free
policy builds the well-and-clear shape it requires, so its priority-order slot (checked before
MOUND/FLAT) is untested by this corpus and will need a routed or hand-built board to exercise.
TOPPING_OUT is exclusively a random-play phenomenon (0 heuristic vs. 10 random, matching that
all four heuristic runs finished with `topped_out: false` and all four random runs with
`topped_out: true`), confirming its top spot in the priority order correctly gates the one
policy that actually collapses. MOUND is the largest single class (53.4%) and — together with
FLAT (33.0%) — accounts for essentially all heuristic boards (24 vs. 25, nearly an even split),
which is consistent with a competent policy spending most of its life in gently uneven or flat
terrain rather than in the risk classes; HOLE_RISK stays rare in both policies (1 heuristic, 3
random; 3.9% overall), suggesting the well-depth=4 threshold is a tight gate that mostly waits
for genuinely bad boards rather than firing on ordinary bumpiness.
