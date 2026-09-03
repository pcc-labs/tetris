# 2026-09-03 — situation histogram over a free corpus

Validation for the situation router (docs/superpowers/specs/2026-09-02-situation-router-design.md,
section 06). No per-decision traces existed on disk, so the corpus is generated:
`heuristic` and `random` runs, seeds 0–3, 150-piece cap, paused/headless, classified with default
thresholds (`topping_out_height=12`, `mound_rise=3`, `well_depth=4`). Boards are captured directly
at decision time into each run's `boards.jsonl` (not reconstructed by replaying the recorded plan
afterward), so every piece placed contributes a classified board.

    uv run tetris-situations generate --seeds 0 1 2 3 --max-pieces 150
    uv run tetris-situations histogram

Heuristic play yields well-formed boards; random play yields messy ones. Neither is a model
arm's distribution — this measures the classifier's coverage and priority order, not its
effect on play.

Note: the run day rolled past midnight UTC between task setup and execution — the brief was
dated 2026-09-02, but `date -u +%F` read 2026-09-03 at run time, so this file and its title use
the actual run date per the brief's own instruction.

Method note: an earlier run of this task used `traces.mine_run` to reconstruct boards by
replaying each run's recorded plan, which stops at the first controller replan and so kept only
103 of ~590 boards played; capturing boards at decision time (commit 2081bf0) replaced that
replay path and now yields one board per piece placed, with none lost.

## Results

| kind | heuristic | random | total | share |
|---|---|---|---|---|
| TOPPING_OUT | 0 | 38 | 38 | 5.5% |
| TETRIS_READY | 0 | 0 | 0 | 0.0% |
| HOLE_RISK | 10 | 5 | 15 | 2.2% |
| MOUND | 407 | 38 | 445 | 64.5% |
| FLAT | 183 | 9 | 192 | 27.8% |
| total | 600 | 90 | 690 | 100.0% |

Boards classified: 690. Pieces per run: heuristic 150 each (seeds 0–3, all four ran to the
150-piece cap without topping out, 150 boards captured each); random 15, 27, 26, 22 (seeds 0–3
respectively, each run topping out before the cap, boards captured equal to pieces placed in
every case).

## Verdict

Dominant class: none. Stopping rule (≥ 90 % in one class): not triggered — proceed to the
routed benchmark.

Observations: TETRIS_READY still never fires for either policy across all 690 boards — neither
free policy builds the well-and-clear shape it requires, so its priority-order slot (checked
before MOUND/FLAT) remains untested by this corpus and will need a routed or hand-built board to
exercise. TOPPING_OUT is exclusively a random-play phenomenon (0 heuristic vs. 38 random),
matching that every random run topped out before the piece cap while every heuristic run did
not — this confirms its top spot in the priority order correctly isolates the one policy that
actually collapses, and with the full corpus captured it now shows up across a meaningfully
larger share of random boards (38/90, 42%) than the truncated first pass suggested. MOUND
dominates heuristic play at full scale (407/600, 68%) with FLAT a distant second (183/600, 30%)
now that every piece is counted, indicating a competent policy spends most of its life in
gently uneven terrain rather than dead flat or at risk; HOLE_RISK stays rare in both policies
(10 heuristic, 5 random; 2.2% overall), which continues to suggest the well-depth=4 threshold is
a tight, well-calibrated gate rather than one that fires on ordinary bumpiness.
