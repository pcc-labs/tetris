# Placement quality: grading every decision against a lookahead oracle

**Status:** approved 2026-09-03, not yet implemented.

## Problem

The benchmark measures outcomes and one board statistic. A row carries score, lines,
pieces survived, average holes, and decision hygiene (illegal, late, timeouts). Nothing
grades a *placement*. So the tables cannot answer questions the 09-03 runs raised:

- `pi/gpt-oss:20b/features/low` scored 225 locally and 530 in the cloud, same weights,
  same seed, same effort, near-identical latency. Better placements, or one seed's luck?
- `gemma4` (e4b) scored 330 with 10.2 holes per piece; `gpt-oss:20b` scored 225 with 0.48.
  One is fast and sloppy, the other clean and stacking badly. Two numbers cannot separate
  "chose well" from "survived".

The longer goal is fine-tuning a local model to play better than cloud. That needs a
per-decision label, which the harness does not produce.

## Goals

1. Grade every model decision against a lookahead oracle: how much worse than the best
   available placement, and where the choice ranked.
2. Aggregate those grades into results columns, so a matrix row says whether reasoning
   level changed *placement quality* or only latency.
3. Persist the per-decision record in the format the exemplar miner already reads, so it
   can become eval and training data without a new parser.
4. Change nothing about how existing benchmarks run or what they score.

## Non-goals

Dataset export, a broker, a training pipeline, prompt or harness changes, and any change
to existing metrics. Each is a separate decision, deliberately deferred (see *Future*).

## The oracle

A new module, `src/tetris_agent/quality.py`, ranks every legal placement for the current
piece and grades the chosen one.

**Value function.** Identical in form to the control arm's evaluation in `policy.py`:

```
value(board, lines) = w_lines·lines
                    + w_agg_height·agg_height(board)
                    + w_holes·holes(board)
                    + w_bumpiness·bumpiness(board)
```

with weights from the same `Genome` the heuristic arm plays with.

**Two-ply search.** For each legal placement of the current piece, place it to get
`(b1, l1)`, then take the maximum over every legal placement of the *known next piece* on
`b1` of `value(b2, l1 + l2)`.

**When the next piece has no legal placement, the game is over.** That placement must
therefore rank BELOW every placement that survives, using a finite constant chosen to sit
under any reachable board value.

> **Corrected 2026-09-03, after the whole-branch review.** This rule originally said to fall
> back to the one-ply `value(b1, l1)` "so a forced top-out does not score as unboundedly
> bad". That was wrong, and dangerously so. One-ply and two-ply values are on different
> scales: a searched two-ply value already carries the next piece's height and hole penalty,
> so it is systematically lower. A one-ply fallback therefore scores the lethal move *above*
> every survivable one. Measured over 19,816 random high-stack boards, 1,205 had both dead
> and live options and in 537 of them — 45 % — the oracle's top choice was the move that
> leaves the next piece nowhere to go. The ceiling arm suicides on exactly the boards that
> end games, and every model that correctly avoids the killer is labelled with positive
> regret. Since these rankings are the training labels, a systematically inverted label on
> near-death boards is the worst failure this design can have.

The constant must be finite, not `-inf` or `NaN`, so arithmetic downstream stays well
behaved. It must also be encoded in the *value*, not only in the sort order: `grade`
computes regret as `best_value - chosen_value`, so a dead placement carrying a high value
with a low sort position would produce negative regret.

**Depth is a parameter.** `ply=1` and `ply=2` share one code path. At `ply=1` the oracle
must reproduce `policy.plan_placement` exactly, including its tie-break of
`(score, landing row, centermost column)`. That equivalence is a test, and it keeps a
cheaper grader available if depth ever costs too much.

**Ranking.** Sort all legal placements by that same key, descending. Rank is the 1-based
position of the chosen placement.

**Measured cost** on a half-full board, per decision:

| depth | cost |
|---|---|
| one-ply | 0.3 - 1.0 ms |
| two-ply | 4 - 33 ms (worst case: T piece, 34 first-ply placements) |

## The grade

```
chosen        (rotation, col)
best          (rotation, col) the oracle would have played
chosen_value  float
best_value    float
worst_value   float
rank          1-based, 1 = the oracle's own choice
legal_count   how many placements were legal
regret        best_value - chosen_value, always >= 0
regret_norm   regret / (best_value - worst_value), in [0, 1]; 0.0 when the span is 0
ply           search depth used
genome        the four weights used, inline, so a label is reproducible
```

`regret_norm` exists because raw regret is in arbitrary units whose scale grows with board
height, so raw values are not comparable across boards. Recording the genome matters
because the self-healing loop mutates those weights: grades from two sessions are only
comparable if they were graded with the same weights.

Grading returns nothing, and the decision is counted as ungraded, when the chosen
placement is not in the legal set. That should not happen, since the policy validates
against the same enumeration, so it is a guard rather than a path.

## Where it runs

Both loops (`agent.py` for paused and bounded-pause modes, `live_agent.py` for live
gravity) keep the pre-decision board, and grade **after the piece has locked**.

The live loop already copies the board when it submits a decision to its worker thread, so
that copy is where the board is kept. It has two lock points: the normal one after
execution, and a lock with no decision at all. Only the first is graded.

Grading after the lock, rather than before execution, is what makes the feature free: the
cost lands in the gap between pieces instead of inside the piece's fall. Grading is
therefore never charged to the model's latency, and never costs gravity distance.

**Decisions that are not graded**, each counted so the exclusion is visible:

- A late or timed-out decision. Gravity placed the piece; the model's choice never ran.
- A fallback placement. When every attempt fails, the policy returns the first legal
  placement and increments `illegal_count`; that is not a choice and must not pollute the
  model's regret. The loop cannot currently tell, so the policy gains a
  `last_fallback: bool` alongside the `last_reason` and `last_output_tokens` it already
  exposes.

## The record

A new event type, `placement_graded`, emitted after the lock, carrying the turn and the
grade.

A new event rather than a field on `placement_decision` for two reasons. First, the
decision event is published *before* execution because the viewer renders the think-freeze
between spawn and decision, and the grade does not exist yet at that point; reordering
would change what watchers see. Second, additive-only: every existing event keeps its
exact shape, so the viewer and `traces.py` are unaffected until they opt in.

Every benchmark arm gets a `RunRecorder`, which today only `tetris-agent` and
`tetris-play` do, so a matrix leaves one trace directory per arm under `runs/`. Events
only, frames off, because frames are the expensive part. `runs/` stays gitignored.

`traces.py` must keep mining runs recorded before this change, and must not break on the
new event.

## The columns

`FitnessTracker` accumulates grades. Fitness gains:

```
graded_decisions   int
mean_regret        mean regret_norm over graded decisions
top1_rate          fraction where rank == 1
top3_rate          fraction where rank <= 3
```

The summary table gains `regret`, `top1` and `top3` beside `avg holes`. Existing fields
and `race_score` are untouched, so every published row stays comparable.

## The ceiling arm

`lookahead`: a policy that plays the oracle's own recommendation. Opt-in via
`--lookahead-control`, and deliberately **not** added to `BASELINE_POLICIES`, so a command
run yesterday produces the same rows tomorrow.

Its purpose is evidence, not decoration. If the oracle's rankings are to be trusted as
training labels, the oracle must be shown to play well. The one-ply control scores 490 on
the live screen while `nemotron-3-super` at thinking-off scored 730, so a model beating the
oracle is already possible. This arm puts the oracle's own score in the same table as the
regret it assigns.

## What must not change

The isolation guarantee, stated so it can be tested:

- Same seeds, same prompts, same placements, same score, lines, pieces, holes and
  `race_score` as before the change.
- Same rows for the same command. The oracle arm is opt-in.
- No added latency inside a piece's fall.
- Existing event types keep their exact fields.
- A `--no-quality` flag disables the whole feature for a run: no grading, no graded
  events, and no trace directory. The `regret` / `top1` / `top3` columns still render, as
  `n/a` — matching how the energy columns already report an unmeasured run, and keeping the
  table's shape the same from run to run. (Corrected 2026-09-03: this line originally said
  "no new columns", which contradicted the repo's own `n/a` convention.)

The one intended addition to how a benchmark runs: with quality on, each arm writes a
trace directory under `runs/`. Events only, no frames. That is new disk I/O, and it is the
point of the feature.

## Risks and caveats

**The oracle is a strong heuristic, not ground truth.** Regret measures disagreement with a
two-ply weighted evaluator. High regret is not proof of bad play: a model scoring above the
oracle's own arm would show high regret while playing better. Every write-up using these
columns must say what the labels are relative to, and the ceiling arm's score belongs in
the same table.

**Regret is per-move and path-relative.** Each decision is graded against the board the
model actually built, not against the board the oracle would have built playing from the
start, and the column is a mean rather than a running total. That is what the training
goal wants: the label answers "from this position, what was the best move", so one bad
move does not poison every label after it. It also means a model can ruin its board, play
the ruin flawlessly, and post near-zero mean regret while losing. Accumulated damage lives
in the outcome columns instead: lines, average holes, pieces survived, topped out. The two
halves of a row measure different things and neither substitutes for the other.

Summing regret instead of averaging would grow with every move, conflating bad play with
long survival, which is the trap `race_score` already navigates. A genuinely cumulative
measure is available later at no new cost: the piece sequence is deterministic, so the
`lookahead` arm on the same seed yields a reference curve of board value per piece to
compare a model's curve against. That is a reporting layer over records this design
already produces, and is deliberately out of scope here.

**Genome drift** makes labels session-dependent. Mitigated by recording the weights with
each grade.

**Two-ply is not deep enough to be an authority** on well-building strategy, which spans
more pieces. It is a floor on placement quality, not a ceiling.

## Future, and why files are enough now

The stated direction is many runs in parallel, on the RTX 5090 or in the cloud, feeding
datasets.

Files do not block that. Each run writes its own directory, so parallel runs never
contend, on one machine or several. The concurrency ceiling today is elsewhere anyway: one
model resident on the iGPU at a time, and three concurrent requests on the Ollama cloud
plan.

A broker is not the tool for the dataset goal. `pokemon-kafka` puts Kafka behind Flink for
real-time anomaly detection, streaming game activity so alerts feed back into the agent's
memory mid-run; its own README records that the JSONL files are the universal interchange
format regardless of who wrote them, and that all four of its published runs were local
files plus DuckDB with no broker. This repo already committed to the same seam
deliberately: `publisher.py` states that the agent never talks to a broker, that JSONL on
disk is the integration point, and that a bridge can tail those files into Kafka without
the agent knowing. That seam still holds after this change, and needs no work now.

When the file count becomes the annoyance, the upgrade is a query layer over the same
records: DuckDB reads a glob of JSONL directly, and the tapes Postgres already runs on
this box if a real table is wanted. Both get easier once the records exist.

## Testing

Test-driven, each watched failing first.

**Oracle**
1. At `ply=1` the oracle reproduces `plan_placement` exactly, tie-breaks included.
2. On a board built so the greedy move ruins the next piece, two-ply chooses differently
   from one-ply.
3. A next piece with no legal placement ranks LAST, below every surviving placement, on a
   board that offers several legal placements of which only one is lethal. Asserting
   finiteness alone is not enough: the original test had a single legal placement and so
   had nothing to compare a rank against, which is how the inverted valuation shipped.
4. Rank and `legal_count` agree with the length of the ranked list.

**Grade**
5. Grading the oracle's own choice gives regret 0.0 and rank 1.
6. Grading the worst legal choice gives `rank == legal_count` and `regret_norm == 1.0`.
7. A board where every placement scores alike gives `regret_norm == 0.0`, not a division
   by zero.
8. A chosen placement outside the legal set returns nothing and counts as ungraded.

**Wiring**
9. The recorded `latency_ms` is identical with and without grading, under a fake clock.
10. Late decisions and fallback placements are excluded, and `graded_decisions` reflects it.
11. Fitness averages over graded decisions only.
12. The `lookahead` arm reports regret 0.0 and `top1_rate` 1.0 through the real fitness
    path.
13. `--no-quality` produces a run with no grading, no `placement_graded` events, no new
    columns, and no trace directory.
14. `expand_arms` output for an existing command is unchanged; the oracle arm appears only
    with its flag.

**Compatibility**
15. `traces.mine_run` reads a graded events file and an ungraded one, with the same
    exemplars from the shared events.
