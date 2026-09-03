# Placement Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grade every model placement against a two-ply lookahead oracle, so a benchmark row says whether a change improved play or only latency, and so each decision leaves a reusable training label.

**Architecture:** A pure module ranks every legal placement by the same weighted evaluation the heuristic control arm uses, searched two pieces deep, and turns the model's choice into a regret and a rank. Both game loops grade after the piece has locked and emit a new event, so existing events keep their shape and grading never costs gravity. Aggregates become three table columns; the per-decision records land in `runs/` in the format the exemplar miner already reads.

**Tech Stack:** Python 3.13, numpy, pytest, ruff, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-03-placement-quality-design.md`

## Global Constraints

- **Isolation.** Same seeds, prompts, placements, `score`, `lines`, `pieces_placed`, `avg_holes` and `race_score` as before this change. Existing event types keep their exact fields. No added latency inside a piece's fall.
- **Same rows for the same command.** The `lookahead` arm is opt-in and must not join `BASELINE_POLICIES`.
- **Unmeasured is not zero.** An ungraded run reports `n/a`, never `0.0`, matching the existing energy-column convention.
- **Grade after the lock**, never between the decision and its execution.
- **TDD.** Every step writes the test first and watches it fail.
- **Style.** `uv run ruff check src tests` and `uv run ruff format` must pass. One pre-existing failure is expected in the suite: `tests/test_power.py::test_scoped_sudoers_entry_counts_as_available` is macOS-only and fails on Linux. One pre-existing ruff error exists in `tests/test_live_agent.py`. Neither is yours to fix.
- **Run the suite with** `uv run pytest -q`.
- **Existing tie-break**, used everywhere placements are ranked: `(value, landing row, -abs(col - 4))`, maximum wins. It comes from `policy.plan_placement` and must not be reinvented.

---

### Task 1: The oracle — ranking every legal placement

**Files:**
- Create: `src/tetris_agent/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: `tetris_agent.board.drop/place/features/COLS`, `tetris_agent.pieces.SHAPES/distinct_rotations/shape_width`, `tetris_agent.policy.Genome`.
- Produces: `DEFAULT_PLY = 2`; `rank_placements(board: np.ndarray, piece: str, next_piece: str | None = None, genome: Genome | None = None, ply: int = DEFAULT_PLY) -> list[tuple[tuple[int, int], float]]`, best first, each entry `((rotation, col), value)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quality.py`:

```python
"""The lookahead oracle and the grade it produces."""

import numpy as np

from tetris_agent import quality
from tetris_agent.policy import Genome, plan_placement

PIECES = ("I", "J", "L", "O", "S", "T", "Z")


def empty_board():
    return np.zeros((18, 10), dtype=bool)


def rough_board(seed=1, fill=0.55, top_row=12):
    """A half-full board with an uneven surface, so placements actually differ."""
    board = empty_board()
    rng = np.random.default_rng(seed)
    for r in range(top_row, 18):
        for c in range(10):
            board[r, c] = rng.random() < fill
    return board


def test_one_ply_reproduces_the_heuristic_solver():
    """ply=1 must be the control arm's own choice, tie-breaks included."""
    genome = Genome()
    for seed in (1, 2, 3):
        board = rough_board(seed)
        for piece in PIECES:
            ranked = quality.rank_placements(board, piece, "T", genome, ply=1)
            best = plan_placement(board, piece, genome)
            assert ranked, f"{piece} seed {seed}: no legal placements"
            assert ranked[0][0] == (best.rotation, best.col), f"{piece} seed {seed}"


def test_ranking_is_ordered_best_first():
    ranked = quality.rank_placements(rough_board(), "T", "I")
    values = [v for _, v in ranked]
    assert values == sorted(values, reverse=True)


def test_every_legal_placement_appears_exactly_once():
    from tetris_agent.policy import legal_moves

    board = rough_board()
    for piece in PIECES:
        ranked = quality.rank_placements(board, piece, "O")
        assert sorted(rc for rc, _ in ranked) == sorted(legal_moves(board, piece))


def test_two_ply_differs_from_one_ply_when_the_greedy_move_traps_the_next_piece():
    """A flat floor with one deep single-width well.

    One-ply happily fills the well with a vertical I. Two-ply, knowing another
    I is next, should value the alternatives differently — the point is only
    that looking ahead changes the ranking at all.
    """
    board = empty_board()
    board[14:18, :] = True
    board[14:18, 3] = False  # a 4-deep well in column 3
    one = quality.rank_placements(board, "I", "I", ply=1)
    two = quality.rank_placements(board, "I", "I", ply=2)
    assert [rc for rc, _ in one] != [rc for rc, _ in two]


def test_a_dead_next_piece_falls_back_to_the_one_ply_value():
    """No legal reply must not crash, and must not produce a non-finite value."""
    board = empty_board()
    board[1:18, :] = True  # only row 0 is free; nothing can follow
    board[0, 0] = False
    ranked = quality.rank_placements(board, "O", "O", ply=2)
    for _, value in ranked:
        assert np.isfinite(value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_quality.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tetris_agent.quality'`.

- [ ] **Step 3: Write the module**

Create `src/tetris_agent/quality.py`:

```python
"""Grade a placement against a lookahead oracle.

The benchmark scores outcomes; this scores decisions. `rank_placements` ranks
every legal placement for the current piece by the same weighted evaluation the
heuristic control arm plays with, searched `ply` pieces deep. `grade` turns a
model's choice into a regret and a rank against that ranking.

The oracle is a strong heuristic, not ground truth: regret is disagreement with
a two-ply evaluator, so a model that outplays the oracle would show high regret.
The `lookahead` arm exists to put the oracle's own score in the same table.
"""

import numpy as np

from tetris_agent.board import COLS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.policy import Genome

DEFAULT_PLY = 2


def _value(board: np.ndarray, lines: int, genome: Genome) -> float:
    """The control arm's evaluation, so the oracle and the solver agree at ply=1."""
    f = features(board)
    return (
        genome.w_lines * lines
        + genome.w_agg_height * f.agg_height
        + genome.w_holes * f.holes
        + genome.w_bumpiness * f.bumpiness
    )


def _drops(board: np.ndarray, piece: str):
    """Every legal (rotation, col, landing row, shape) for `piece` on `board`."""
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is not None:
                yield rot, col, row, shape


def rank_placements(
    board: np.ndarray,
    piece: str,
    next_piece: str | None = None,
    genome: Genome | None = None,
    ply: int = DEFAULT_PLY,
) -> list[tuple[tuple[int, int], float]]:
    """[((rotation, col), value)], best first.

    At ply=1 this is `policy.plan_placement`'s ranking, tie-break included. At
    ply=2 each placement is worth the best reply the known next piece has to it;
    a placement that leaves the next piece nowhere to go keeps its one-ply value
    rather than scoring as unboundedly bad.
    """
    genome = genome or Genome()
    scored = []
    for rot, col, row, shape in _drops(board, piece):
        after, lines = place(board, shape, col, row)
        value = _value(after, lines, genome)
        if ply > 1 and next_piece is not None:
            best_reply = None
            for _, col2, row2, shape2 in _drops(after, next_piece):
                board2, lines2 = place(after, shape2, col2, row2)
                reply = _value(board2, lines + lines2, genome)
                best_reply = reply if best_reply is None else max(best_reply, reply)
            if best_reply is not None:
                value = best_reply
        # Tie-break from policy.plan_placement: deeper landing, then centermost.
        scored.append(((rot, col), value, row, -abs(col - 4)))
    scored.sort(key=lambda e: (e[1], e[2], e[3]), reverse=True)
    return [(placement, value) for placement, value, _, _ in scored]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_quality.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/tetris_agent/quality.py tests/test_quality.py
uv run ruff check src/tetris_agent/quality.py tests/test_quality.py
git add src/tetris_agent/quality.py tests/test_quality.py
git commit -m "feat: rank every legal placement with a lookahead oracle

At ply=1 it reproduces the heuristic control arm's ranking exactly, so the
solver and the grader cannot drift apart; at ply=2 each placement is worth the
best reply the known next piece has to it."
```

---

### Task 2: The grade — regret and rank for one decision

**Files:**
- Modify: `src/tetris_agent/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: `rank_placements` from Task 1; `tetris_agent.policy.Placement`.
- Produces: frozen dataclass `Grade` with fields `chosen, best, chosen_value, best_value, worst_value, rank, legal_count, regret, regret_norm, ply, genome` and a `to_dict() -> dict`; `grade(board, piece, next_piece, chosen: Placement, genome: Genome | None = None, ply: int = DEFAULT_PLY) -> Grade | None`. Argument order is `(board, piece, next_piece, chosen)` so call sites read like the game loop.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_quality.py`:

```python
from tetris_agent.policy import Placement


def placement_of(entry):
    (rotation, col), value = entry
    return Placement(rotation=rotation, col=col, score=value)


def test_grading_the_oracles_own_choice_is_zero_regret_rank_one():
    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    g = quality.grade(board, "T", "I", placement_of(ranked[0]))
    assert g.rank == 1
    assert g.regret == 0.0
    assert g.regret_norm == 0.0
    assert g.best == g.chosen
    assert g.legal_count == len(ranked)


def test_grading_the_worst_legal_choice_is_last_rank_and_full_regret():
    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    g = quality.grade(board, "T", "I", placement_of(ranked[-1]))
    assert g.rank == g.legal_count == len(ranked)
    assert g.regret_norm == 1.0
    assert g.regret > 0


def test_an_all_equal_board_is_zero_regret_not_a_division_by_zero():
    """An empty board with an O piece: every column is worth the same."""
    g = quality.grade(empty_board(), "O", "O", Placement(rotation=0, col=7, score=0.0), ply=1)
    assert g.regret_norm == 0.0
    assert g.best_value == g.worst_value


def test_a_placement_outside_the_legal_set_is_not_graded():
    board = np.ones((18, 10), dtype=bool)
    assert quality.grade(board, "T", "I", Placement(rotation=0, col=0, score=0.0)) is None


def test_the_grade_records_the_weights_it_used():
    genome = Genome(w_holes=-0.9)
    g = quality.grade(rough_board(), "T", "I", Placement(rotation=0, col=0, score=0.0), genome=genome)
    if g is not None:
        assert g.genome["w_holes"] == -0.9
        assert g.ply == 2


def test_to_dict_is_json_safe_and_carries_the_ranking():
    import json

    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    payload = quality.grade(board, "T", "I", placement_of(ranked[1])).to_dict()
    assert json.loads(json.dumps(payload))["rank"] == 2
    assert payload["best"] == list(ranked[0][0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_quality.py -q`
Expected: FAIL, `AttributeError: module 'tetris_agent.quality' has no attribute 'grade'`.

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `src/tetris_agent/quality.py`:

```python
from dataclasses import dataclass

from tetris_agent.policy import Genome, Placement
```

Append to `src/tetris_agent/quality.py`:

```python
@dataclass(frozen=True)
class Grade:
    """One decision, scored against the oracle's ranking of the same board.

    `regret` is in the evaluation's arbitrary units, whose scale grows with
    board height, so `regret_norm` is the comparable one. Both are per-move and
    path-relative: the board graded is the one the model built, not the one the
    oracle would have built, so a ruined board played well scores near zero.
    Accumulated damage lives in the outcome metrics instead.
    """

    chosen: tuple[int, int]
    best: tuple[int, int]
    chosen_value: float
    best_value: float
    worst_value: float
    rank: int
    legal_count: int
    regret: float
    regret_norm: float
    ply: int
    genome: dict

    def to_dict(self) -> dict:
        return {
            "chosen": list(self.chosen),
            "best": list(self.best),
            "chosen_value": round(self.chosen_value, 6),
            "best_value": round(self.best_value, 6),
            "worst_value": round(self.worst_value, 6),
            "rank": self.rank,
            "legal_count": self.legal_count,
            "regret": round(self.regret, 6),
            "regret_norm": round(self.regret_norm, 6),
            "ply": self.ply,
            "genome": self.genome,
        }


def grade(
    board: np.ndarray,
    piece: str,
    next_piece: str | None,
    chosen: Placement,
    genome: Genome | None = None,
    ply: int = DEFAULT_PLY,
) -> Grade | None:
    """Score `chosen` against every alternative, or None if it cannot be graded.

    None means the choice was not in the legal set — a guard, not a path, since
    the policy validates against the same enumeration.
    """
    genome = genome or Genome()
    ranked = rank_placements(board, piece, next_piece, genome, ply)
    if not ranked:
        return None
    key = (chosen.rotation, chosen.col)
    index = next((i for i, (placement, _) in enumerate(ranked) if placement == key), None)
    if index is None:
        return None
    best_value, worst_value = ranked[0][1], ranked[-1][1]
    chosen_value = ranked[index][1]
    regret = best_value - chosen_value
    span = best_value - worst_value
    return Grade(
        chosen=key,
        best=ranked[0][0],
        chosen_value=chosen_value,
        best_value=best_value,
        worst_value=worst_value,
        rank=index + 1,
        legal_count=len(ranked),
        regret=regret,
        regret_norm=(regret / span) if span > 0 else 0.0,
        ply=ply,
        genome={
            "w_lines": genome.w_lines,
            "w_agg_height": genome.w_agg_height,
            "w_holes": genome.w_holes,
            "w_bumpiness": genome.w_bumpiness,
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_quality.py -q`
Expected: 11 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/tetris_agent/quality.py tests/test_quality.py
uv run ruff check src/tetris_agent/quality.py tests/test_quality.py
git add src/tetris_agent/quality.py tests/test_quality.py
git commit -m "feat: grade a placement into a regret and a rank

Normalized regret because raw regret's scale grows with board height. The
weights are recorded with the grade, since the healer mutates them and two
sessions' labels are only comparable if they were graded the same way."
```

---

### Task 3: The ceiling arm — the oracle plays

**Files:**
- Modify: `src/tetris_agent/policy.py` (add `LookaheadPolicy` after `HeuristicPolicy`)
- Modify: `src/tetris_agent/benchmark.py` (`build_policy`, `--lookahead-control`)
- Test: `tests/test_policy.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `quality.rank_placements` from Task 1.
- Produces: `LookaheadPolicy(genome: Genome | None = None, ply: int = 2)` with `.name == "lookahead"`, `.plan(board, piece, next_piece, turn) -> Placement | None`, `.stats() -> dict`. `Arm(policy="lookahead")` builds it. CLI flag `--lookahead-control`.

Why this arm exists: the oracle's rankings become training labels, so the oracle has to be shown to play well. The one-ply control scores 490 on the live screen while a model already scored 730, so a teacher that plays badly is a real possibility and must be visible in the same table.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy.py`:

```python
def test_lookahead_policy_plays_the_oracles_own_choice():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.policy import Genome, LookaheadPolicy

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :7] = True
    policy = LookaheadPolicy()
    placement = policy.plan(board, "T", "I", turn=1)
    best = quality.rank_placements(board, "T", "I", Genome(), ply=2)[0][0]
    assert (placement.rotation, placement.col) == best
    assert policy.name == "lookahead"
    assert policy.stats()["cost_usd"] == 0.0


def test_lookahead_policy_returns_none_when_nothing_fits():
    import numpy as np

    from tetris_agent.policy import LookaheadPolicy

    assert LookaheadPolicy().plan(np.ones((18, 10), dtype=bool), "T", "I", turn=1) is None
```

Append to `tests/test_benchmark.py`:

```python
def test_lookahead_arm_is_opt_in_and_absent_by_default():
    """A command run yesterday must produce the same rows tomorrow."""
    default = [a.name for a in expand_arms(["claude-opus-5"], ["board"], ["low"])]
    assert "lookahead" not in default

    opted = [a.name for a in expand_arms(["claude-opus-5"], ["board"], ["low"], lookahead_control=True)]
    assert "lookahead" in opted
    assert opted[: len(default)] == default  # appended, nothing reordered


def test_build_policy_dispatches_the_lookahead_arm():
    from tetris_agent.policy import LookaheadPolicy

    assert isinstance(build_policy(Arm("lookahead")), LookaheadPolicy)


def test_lookahead_arm_costs_nothing_in_the_estimate():
    arms = expand_arms([], [], [], lookahead_control=True)
    assert estimate_cost(arms, [0], max_pieces=50) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_policy.py tests/test_benchmark.py -q`
Expected: FAIL, `ImportError: cannot import name 'LookaheadPolicy'` and `TypeError: expand_arms() got an unexpected keyword argument 'lookahead_control'`.

- [ ] **Step 3: Write the implementation**

In `src/tetris_agent/policy.py`, directly after the `HeuristicPolicy` class:

```python
class LookaheadPolicy:
    """Two-ply ceiling arm: plays the oracle that grades every other arm.

    Opt-in, never a default baseline, so adding it cannot change the rows an
    existing command produces. Its score is the evidence that the grader's
    rankings are worth trusting as labels — a teacher that plays badly is a bad
    teacher, and `quality`'s regret column means nothing without this row.
    """

    def __init__(self, genome: Genome | None = None, ply: int = 2):
        self.genome = genome or Genome()
        self.ply = ply
        self.name = "lookahead"

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        # Imported here, not at module scope: quality imports Genome from this
        # module, so a top-level import would be circular.
        from tetris_agent.quality import rank_placements

        ranked = rank_placements(board, piece, next_piece, self.genome, self.ply)
        if not ranked:
            return None
        (rotation, col), value = ranked[0]
        return Placement(rotation=rotation, col=col, score=value)

    def stats(self) -> dict:
        return {"policy": self.name, "decisions": 0, "cost_usd": 0.0}
```

In `src/tetris_agent/benchmark.py`, add the parameter to `expand_arms` (after `fixed_effort`):

```python
    lookahead_control: bool = False,
```

and inside `expand_arms`, immediately after the `arms = [...]` baseline line. Append rather
than insert, so every existing arm keeps its position in the list:

```python
    if lookahead_control:
        arms.append(Arm(policy="lookahead"))
```

In `build_policy`, alongside the other baseline branches:

```python
    if arm.policy == "lookahead":
        from tetris_agent.policy import LookaheadPolicy

        return LookaheadPolicy(genome)
```

Add the import of `LookaheadPolicy` to the existing `from tetris_agent.policy import ...` line at the top of `build_policy` instead if that reads cleaner; either is fine as long as it is one import.

Add the CLI flag next to `--no-control`:

```python
    parser.add_argument(
        "--lookahead-control",
        action="store_true",
        help="add the two-ply oracle as a ceiling arm (the arm that placement quality is graded against)",
    )
```

and thread it into the `expand_arms(...)` call in `main`:

```python
        lookahead_control=args.lookahead_control,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_policy.py tests/test_benchmark.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src/tetris_agent/policy.py src/tetris_agent/benchmark.py tests/test_policy.py tests/test_benchmark.py
git add src/tetris_agent/policy.py src/tetris_agent/benchmark.py tests/test_policy.py tests/test_benchmark.py
git commit -m "feat: opt-in lookahead ceiling arm

The oracle's rankings are about to become labels, so the oracle needs its own
score in the same table. Opt-in via --lookahead-control and deliberately not a
default baseline, so existing commands keep producing identical rows."
```

---

### Task 4: The event — recording a grade

**Files:**
- Modify: `src/tetris_agent/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `Grade` from Task 2.
- Produces: `build_graded_event(turn: int, grade: Grade) -> dict` with `event_type == "placement_graded"`; `EventCollector.graded(grade: Grade, turn: int | None = None) -> None`.

A new event rather than a field on `placement_decision`, because the decision event is published before execution so the viewer can render the think-freeze, and the grade does not exist yet at that point. Additive-only: every existing event keeps its exact shape.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_events.py`:

```python
def test_graded_event_carries_the_grade_and_leaves_the_decision_event_alone():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import EventCollector, build_decision_event, build_graded_event
    from tetris_agent.policy import Placement

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    chosen = Placement(rotation=0, col=7, score=0.0)
    g = quality.grade(board, "O", "I", chosen)

    event = build_graded_event(turn=4, grade=g)
    assert event["event_type"] == "placement_graded"
    assert event["turn"] == 4
    assert event["data"]["rank"] == g.rank
    assert event["data"]["regret_norm"] == round(g.regret_norm, 6)

    # The decision event must be byte-identical to before this feature.
    decision = build_decision_event(turn=4, placement=chosen)
    assert set(decision["data"]) == {"rotation", "col", "score"}


def test_collector_publishes_a_graded_event():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Placement

    published = []

    class Sink:
        def publish(self, event):
            published.append(event)

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    collector = EventCollector(Sink())
    collector.graded(quality.grade(board, "O", "I", Placement(rotation=0, col=7, score=0.0)))
    assert [e["event_type"] for e in published] == ["placement_graded"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events.py -q`
Expected: FAIL, `ImportError: cannot import name 'build_graded_event'`.

- [ ] **Step 3: Write the implementation**

In `src/tetris_agent/events.py`, after `build_locked_event`:

```python
def build_graded_event(turn: int, grade) -> dict:
    """A decision scored against the lookahead oracle (see quality.py).

    Separate from `placement_decision` because that event is published before
    execution, so the viewer can render the think-freeze, and the grade does not
    exist until the piece has locked.
    """
    return _envelope("placement_graded", turn, grade.to_dict())
```

and on `EventCollector`, mirroring the existing `decision` method's optional turn:

```python
    def graded(self, grade, turn: int | None = None) -> None:
        self.publisher.publish(build_graded_event(self.turn if turn is None else turn, grade))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_events.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/tetris_agent/events.py tests/test_events.py
uv run ruff check src/tetris_agent/events.py tests/test_events.py
git add src/tetris_agent/events.py tests/test_events.py
git commit -m "feat: placement_graded event

Additive: existing events keep their exact fields, so the viewer and the
exemplar miner are unaffected until they opt in."
```

---

### Task 5: The columns — aggregating grades into a row

**Files:**
- Modify: `src/tetris_agent/fitness.py`
- Modify: `src/tetris_agent/benchmark.py` (`summarize`, `render_table`)
- Test: `tests/test_fitness.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `Grade` from Task 2.
- Produces: `FitnessTracker.on_grade(grade: Grade) -> None`; `compute()` gains `graded_decisions: int`, `mean_regret: float | None`, `top1_rate: float | None`, `top3_rate: float | None`. `benchmark._quality_cells(runs) -> dict` producing `regret`, `top1`, `top3`, `graded`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fitness.py`:

```python
def test_grades_average_over_graded_decisions_only():
    from types import SimpleNamespace

    from tetris_agent.fitness import FitnessTracker

    tracker = FitnessTracker()
    for holes in (0, 1, 2):
        tracker.on_lock(SimpleNamespace(holes=holes, agg_height=0, bumpiness=0, max_height=1), misexec=0)
    tracker.on_grade(SimpleNamespace(regret_norm=0.0, rank=1))
    tracker.on_grade(SimpleNamespace(regret_norm=0.5, rank=4))
    # The third piece was late: never graded.

    out = tracker.compute(score=0, lines=0, level=0)
    assert out["pieces_placed"] == 3
    assert out["graded_decisions"] == 2
    assert out["mean_regret"] == 0.25
    assert out["top1_rate"] == 0.5
    assert out["top3_rate"] == 0.5


def test_an_ungraded_run_reports_none_not_zero():
    """Unknown and perfect are different claims, as with energy."""
    from tetris_agent.fitness import FitnessTracker

    out = FitnessTracker().compute(score=0, lines=0, level=0)
    assert out["graded_decisions"] == 0
    assert out["mean_regret"] is None
    assert out["top1_rate"] is None


def test_the_oracle_arm_grades_itself_as_perfect():
    """The grader's self-test: the arm that plays the oracle must post regret 0.

    If this ever fails, the ranking the arm plays and the ranking the grader
    scores against have drifted apart, and every regret column is suspect.
    """
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.fitness import FitnessTracker
    from tetris_agent.policy import LookaheadPolicy

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :7] = True
    policy = LookaheadPolicy()
    tracker = FitnessTracker()
    for piece, next_piece in (("T", "I"), ("L", "O"), ("S", "Z")):
        chosen = policy.plan(board, piece, next_piece, turn=1)
        tracker.on_grade(quality.grade(board, piece, next_piece, chosen))

    out = tracker.compute(score=0, lines=0, level=0)
    assert out["graded_decisions"] == 3
    assert out["mean_regret"] == 0.0
    assert out["top1_rate"] == 1.0
```

Append to `tests/test_benchmark.py`:

```python
def test_quality_columns_are_na_when_nothing_was_graded():
    rows = summarize([ArmResult("a", 0, fitness(), {})])
    assert rows[0]["regret"] == "n/a"
    assert rows[0]["top1"] == "n/a"
    assert rows[0]["graded"] == 0


def test_quality_columns_weight_seeds_by_their_graded_decisions():
    graded_a = {**fitness(), "graded_decisions": 10, "mean_regret": 0.10, "top1_rate": 1.0, "top3_rate": 1.0}
    graded_b = {**fitness(), "graded_decisions": 30, "mean_regret": 0.50, "top1_rate": 0.0, "top3_rate": 0.0}
    rows = summarize([ArmResult("a", 0, graded_a, {}), ArmResult("a", 1, graded_b, {})])
    # (0.10*10 + 0.50*30) / 40 = 0.40 — a decision mean, not a mean of means.
    assert rows[0]["regret"] == 0.4
    assert rows[0]["top1"] == 0.25
    assert rows[0]["graded"] == 40


def test_quality_columns_appear_in_the_table():
    rows = summarize([ArmResult("a", 0, {**fitness(), "graded_decisions": 1, "mean_regret": 0.0,
                                         "top1_rate": 1.0, "top3_rate": 1.0}, {})])
    table = render_table(rows)
    assert "regret" in table and "top1" in table and "top3" in table
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fitness.py tests/test_benchmark.py -q`
Expected: FAIL, `AttributeError: 'FitnessTracker' object has no attribute 'on_grade'` and `KeyError: 'regret'`.

- [ ] **Step 3: Write the implementation**

In `src/tetris_agent/fitness.py`, add to `FitnessTracker.__init__`:

```python
        self._regrets: list[float] = []
        self._ranks: list[int] = []
```

add the method:

```python
    def on_grade(self, grade) -> None:
        """One decision scored against the oracle. Late and fallback placements
        never reach here, so the averages describe choices the model actually made."""
        self._regrets.append(grade.regret_norm)
        self._ranks.append(grade.rank)
```

and in `compute`, bind the count just above the existing `return {`, then add four keys to
that dict after its last existing entry, `"topped_out"`. Every existing key keeps its
current expression untouched:

```python
        graded = len(self._regrets)
```

```python
            "topped_out": self._topped_out,
            "graded_decisions": graded,
            "mean_regret": round(sum(self._regrets) / graded, 4) if graded else None,
            "top1_rate": round(sum(1 for r in self._ranks if r == 1) / graded, 3) if graded else None,
            "top3_rate": round(sum(1 for r in self._ranks if r <= 3) / graded, 3) if graded else None,
```

In `src/tetris_agent/benchmark.py`, add beside `_energy_cells`:

```python
def _quality_cells(runs: list[ArmResult]) -> dict:
    """Placement-quality columns for one arm's rows.

    Weighted by each seed's graded-decision count, so the figure is a mean over
    decisions rather than a mean of means. `n/a` rather than 0.0 when nothing was
    graded: quality off and every-decision-late are unknown, not perfect.
    """
    scored = [r for r in runs if r.fitness]
    total = sum(r.fitness.get("graded_decisions") or 0 for r in scored)
    if not total:
        return {"regret": "n/a", "top1": "n/a", "top3": "n/a", "graded": 0}

    def weighted(key: str) -> float:
        return sum((r.fitness.get(key) or 0) * (r.fitness.get("graded_decisions") or 0) for r in scored) / total

    return {
        "regret": round(weighted("mean_regret"), 4),
        "top1": round(weighted("top1_rate"), 3),
        "top3": round(weighted("top3_rate"), 3),
        "graded": total,
    }
```

and in `summarize`'s row dict, immediately after the `**_energy_cells(runs),` line:

```python
                **_quality_cells(runs),
```

In `render_table`, insert into `cols` immediately after `"avg_holes"`:

```python
        "regret",
        "top1",
        "top3",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fitness.py tests/test_benchmark.py -q`
Expected: all pass.

- [ ] **Step 5: Run the whole suite, lint, commit**

```bash
uv run pytest -q   # expect only the known macOS-only power failure
uv run ruff format src tests
uv run ruff check src/tetris_agent/fitness.py src/tetris_agent/benchmark.py
git add src/tetris_agent/fitness.py src/tetris_agent/benchmark.py tests/test_fitness.py tests/test_benchmark.py
git commit -m "feat: regret, top1 and top3 columns

Weighted by graded decisions so the figure is a mean over decisions, and n/a
rather than 0.0 when nothing was graded."
```

---

### Task 6: Wire the paused loop

**Files:**
- Modify: `src/tetris_agent/agent.py`
- Modify: `src/tetris_agent/model_policy.py` (the `last_fallback` flag)
- Test: `tests/test_agent.py`, `tests/test_pi_policy.py`

**Interfaces:**
- Consumes: `quality.grade` (Task 2), `EventCollector.graded` (Task 4), `FitnessTracker.on_grade` (Task 5).
- Produces: `TetrisAgent(..., grader=None)` where `grader` is `Callable[[np.ndarray, str, str, Placement], Grade | None]`; `None` disables grading. `LLMPlacementPolicy.last_fallback: bool`.

Grading runs after the locked event, so its cost lands between pieces rather than inside the piece's fall.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pi_policy.py`, which already has the helpers this needs (`policy`,
`completed`, `ok_events`, `empty_board`):

```python
def test_last_fallback_marks_a_placement_the_model_did_not_choose():
    """A fallback is not a decision, so the grader must skip it."""
    failed = policy([completed("", returncode=1)])
    failed.plan(empty_board(), "O", "I", turn=1)
    assert failed.last_fallback is True
    assert failed.stats()["illegal_count"] == 1

    chose = policy([ok_events()])
    chose.plan(empty_board(), "O", "I", turn=1)
    assert chose.last_fallback is False
```

Append to `tests/test_agent.py`. These build on the file's existing
`_paused_agent_fixtures(monkeypatch, think_s=..., deadline_s=...)` helper, which already
supplies a fake emulator, a fake controller, a fake clock and a capturing publisher.
**First, add a `grader=None` parameter to that helper's signature and pass `grader=grader`
into the `TetrisAgent(...)` it constructs.** Then:

```python
from types import SimpleNamespace

from tetris_agent.policy import Placement


def _fake_grader(calls, regret=0.25, rank=2):
    """A grader that records what it was asked and answers instantly."""

    def grade(board, piece, next_piece, placement):
        calls.append((piece, placement.rotation, placement.col))
        return SimpleNamespace(
            regret_norm=regret,
            rank=rank,
            to_dict=lambda: {"rank": rank, "regret_norm": regret},
        )

    return grade


class FallbackPolicy:
    """A policy whose every answer is the deterministic fallback, not a choice."""

    name = "fallback"
    last_reason = ""
    last_fallback = True

    def plan(self, board, piece, next_piece, turn):
        return Placement(rotation=0, col=0, score=0.0)

    def stats(self):
        return {"policy": self.name, "decisions": 1, "cost_usd": 0.0}


def test_grading_is_not_charged_to_the_model(monkeypatch):
    """Recorded latency must be the think time and nothing else."""
    calls = []
    agent, _, publisher = _paused_agent_fixtures(
        monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader(calls)
    )
    agent.run(timer_div=0)
    decision = next(e for e in publisher.events if e["event_type"] == "placement_decision")
    assert decision["data"]["latency_ms"] == 3000.0
    assert calls, "the grader never ran, so this proves nothing"


def test_the_graded_event_follows_the_lock(monkeypatch):
    agent, _, publisher = _paused_agent_fixtures(
        monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader([])
    )
    agent.run(timer_div=0)
    kinds = [e["event_type"] for e in publisher.events]
    assert kinds.index("piece_locked") < kinds.index("placement_graded")


def test_a_late_decision_is_not_graded(monkeypatch):
    """Gravity placed the piece; the model's choice never ran."""
    calls = []
    agent, _, publisher = _paused_agent_fixtures(
        monkeypatch, think_s=20, deadline_s=15.0, grader=_fake_grader(calls)
    )
    fitness = agent.run(timer_div=0)
    assert calls == []
    assert "placement_graded" not in [e["event_type"] for e in publisher.events]
    assert fitness["graded_decisions"] == 0
    assert fitness["mean_regret"] is None


def test_a_fallback_placement_is_not_graded(monkeypatch):
    calls = []
    agent, _, _ = _paused_agent_fixtures(
        monkeypatch, think_s=3, deadline_s=15.0, policy=FallbackPolicy(), grader=_fake_grader(calls)
    )
    fitness = agent.run(timer_div=0)
    assert calls == []
    assert fitness["graded_decisions"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent.py tests/test_model_policy.py -q`
Expected: FAIL on the missing `last_fallback` attribute and the missing `grader` parameter.

- [ ] **Step 3: Write the implementation**

In `src/tetris_agent/model_policy.py`, add to `__init__` beside `self.last_reason`:

```python
        # True when `plan` fell back to the first legal placement because every
        # attempt failed. Not a choice, so the grader skips it.
        self.last_fallback = False
```

and in `plan`, replace the fallback block with:

```python
        choice = self._decide(prompt, legal)
        self.last_fallback = choice is None
        if choice is None:
            # Every attempt failed; keep the run alive with a deterministic
            # fallback so the arm is still comparable, and count it as illegal.
            self.usage["illegal_count"] += 1
            choice = legal[0]
```

In `src/tetris_agent/agent.py`, add the constructor parameter beside `meter`:

```python
        grader=None,
```

store it as `self.grader = grader`, and in the play loop, immediately after the existing `self.collector.locked(...)` call:

```python
            # After the lock, so grading costs the gap between pieces and never
            # the piece's own fall. Late decisions and fallbacks are not choices.
            if self.grader is not None and not timed_out and not getattr(self.policy, "last_fallback", False):
                g = self.grader(state.board, state.falling.name, state.next_piece, placement)
                if g is not None:
                    tracker.on_grade(g)
                    self.collector.graded(g)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent.py tests/test_model_policy.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src/tetris_agent/agent.py src/tetris_agent/model_policy.py
git add src/tetris_agent/agent.py src/tetris_agent/model_policy.py tests/test_agent.py tests/test_model_policy.py
git commit -m "feat: grade decisions in the paused loop

After the locked event, so the cost lands between pieces and never inside a
piece's fall. Late decisions and fallback placements are excluded: neither is
a choice the model made."
```

---

### Task 7: Wire the live loop

**Files:**
- Modify: `src/tetris_agent/live_agent.py`
- Test: `tests/test_live_agent.py`

**Interfaces:**
- Consumes: the `grader` parameter from Task 6 (inherited, `LiveTetrisAgent` extends `TetrisAgent`).
- Produces: `_Pending.board`, `_Pending.next_piece`; `LiveTetrisAgent._last_decision: tuple | None`.

`_submit` already copies the board for its worker thread, so that copy is where the pre-decision board is kept. There are two lock points; only the executed one is graded.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_agent.py`. It already has `LiveFakeEmulator`, `ManualThreads`,
`ScriptedPolicy`, `install_live`, `install_controller` and `make_agent`. **First, add a
`grader=None` parameter to `make_agent` and pass `grader=grader` into the
`LiveTetrisAgent(...)` it constructs.** Then:

```python
from types import SimpleNamespace


def _fake_grader(calls):
    def grade(board, piece, next_piece, placement):
        calls.append((piece, next_piece, placement.rotation, placement.col))
        return SimpleNamespace(
            regret_norm=0.25, rank=2, to_dict=lambda: {"rank": 2, "regret_norm": 0.25}
        )

    return grade


def test_live_grades_an_executed_decision_after_the_lock(monkeypatch):
    """Order must read spawn -> decision -> locked -> graded."""
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    calls = []
    agent = make_agent(
        emu,
        ScriptedPolicy([Placement(0, 0, 0.0)]),
        pub,
        ManualThreads(immediate=True),
        max_pieces=1,
        grader=_fake_grader(calls),
    )
    agent.run(timer_div=0)

    kinds = [e["event_type"] for e in pub.events]
    assert "placement_graded" in kinds
    assert kinds.index("piece_locked") < kinds.index("placement_graded")
    # The board and next piece handed to the grader are the ones the decision saw.
    assert calls == [("J", "I", 0, 0)]


def test_a_piece_that_locks_without_a_decision_is_not_graded(monkeypatch):
    """The worker never answered, so gravity placed the piece and there is
    nothing to grade."""
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    calls = []
    agent = make_agent(
        emu,
        ScriptedPolicy([Placement(0, 0, 0.0)]),
        pub,
        ManualThreads(immediate=False),  # the decision is never delivered
        max_pieces=1,
        grader=_fake_grader(calls),
    )
    agent.run(timer_div=0)

    assert calls == []
    assert "placement_graded" not in [e["event_type"] for e in pub.events]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_live_agent.py -q`
Expected: FAIL, no `placement_graded` event is ever published.

- [ ] **Step 3: Write the implementation**

In `_Pending.__init__`, add the two fields the grader needs:

```python
    def __init__(self, turn: int, piece: str, board=None, next_piece: str | None = None):
        self.turn = turn
        self.piece = piece
        self.board = board
        self.next_piece = next_piece
```

In `_submit`, pass them through when constructing the pending:

```python
        board = state.board.copy()
        piece, next_piece, turn = state.falling.name, state.next_piece, self.collector.turn
        pending = _Pending(turn=turn, piece=piece, board=board, next_piece=next_piece)
```

In `LiveTetrisAgent.__init__`, initialise the hand-off slot:

```python
        self._last_decision = None
```

In `_ride_piece`, on the executed path, record what was decided before returning:

```python
                self._capture_frame()
                self._last_decision = (pending.board, pending.piece, pending.next_piece, placement)
                return None, "executed", controller.execute(placement)
```

In `_play`, inside the `if outcome == "executed":` branch, immediately after the existing `self.collector.locked(...)` call:

```python
                decided, self._last_decision = self._last_decision, None
                if self.grader is not None and decided is not None and not getattr(self.policy, "last_fallback", False):
                    board, piece, next_piece, placement = decided
                    g = self.grader(board, piece, next_piece, placement)
                    if g is not None:
                        tracker.on_grade(g)
                        self.collector.graded(g)
```

Late decisions never reach the executed path, so they need no extra guard here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_live_agent.py -q`
Expected: pass, except the file's pre-existing ruff error which is unrelated.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/tetris_agent/live_agent.py tests/test_live_agent.py
uv run ruff check src/tetris_agent/live_agent.py
git add src/tetris_agent/live_agent.py tests/test_live_agent.py
git commit -m "feat: grade decisions in the live loop

The board copy _submit already makes for its worker thread is the pre-decision
board the grader needs. Graded after the locked event, so the cost lands in the
gap between pieces rather than in the piece's fall."
```

---

### Task 8: Turn it on for benchmarks, and keep old traces readable

**Files:**
- Modify: `src/tetris_agent/benchmark.py` (`run_arm`, `--no-quality`)
- Modify: `src/tetris_agent/cli.py` (`--no-quality` for single runs)
- Test: `tests/test_benchmark.py`, `tests/test_traces.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_arm(..., grade_quality: bool = True)`; CLI flag `--no-quality` on both entry points.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark.py`. Playing a real game needs the ROM and is marked
`@pytest.mark.rom`, so the flag is checked where it is plumbed; step 5 verifies the
behaviour by hand:

```python
def test_grading_is_on_by_default(monkeypatch):
    import inspect

    from tetris_agent.benchmark import run_arm

    assert inspect.signature(run_arm).parameters["grade_quality"].default is True


def test_no_quality_is_an_accepted_flag():
    assert main(["--models", "claude-opus-5", "--estimate", "--max-pieces", "5", "--no-quality"]) == 0
```

Append to `tests/test_traces.py`, reusing its `write_run` helper and `_envelope`:

```python
def test_mine_run_ignores_graded_events(tmp_path):
    """A graded run and the same run without grades mine to identical exemplars."""
    pieces = [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, THEN_O_AT_2)]
    plain = write_run(tmp_path, "20260903-000001-aaaaaa", "human", pieces)
    graded = write_run(tmp_path, "20260903-000002-bbbbbb", "human", pieces)

    # Interleave a grade after every lock, exactly as the game loops now do.
    events = [json.loads(line) for line in (graded / "events.jsonl").read_text().splitlines()]
    with_grades = []
    for event in events:
        with_grades.append(event)
        if event["event_type"] == "piece_locked":
            with_grades.append(
                _envelope("placement_graded", event["turn"], {"rank": 1, "regret": 0.0, "regret_norm": 0.0})
            )
    (graded / "events.jsonl").write_text("\n".join(json.dumps(e) for e in with_grades) + "\n")

    def summary(run_dir):
        # Compared field by field: an Exemplar holds a numpy board, so == on the
        # dataclass raises rather than answering.
        return [(e.piece, e.next_piece, e.rotation, e.col, e.lines_delta, e.holes) for e in mine_run(run_dir)]

    assert summary(plain), "the fixture produced no exemplars, so this proves nothing"
    assert summary(graded) == summary(plain)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py tests/test_traces.py -q`
Expected: FAIL, `run_arm() got an unexpected keyword argument 'grade_quality'`.

- [ ] **Step 3: Write the implementation**

In `src/tetris_agent/benchmark.py`, add the `run_arm` parameter beside `measure_power`:

```python
    grade_quality: bool = True,
```

build the grader and the per-arm recorder inside `run_arm`, next to where the meter is built:

```python
    # Placement grading, and the trace it writes. Events only: frames are the
    # expensive part of a recording and nothing here needs them.
    grader = recorder = None
    if grade_quality:
        import functools

        from tetris_agent.quality import DEFAULT_PLY, grade

        grader = functools.partial(grade, genome=Genome.from_params(genome_params or {}), ply=DEFAULT_PLY)
        recorder = RunRecorder(RUNS_DIR, label=arm.name)
```

pass `grader=grader` and `recorder=recorder` into the agent construction alongside `meter=meter`, and add the argument-parser flag:

```python
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="skip placement grading (no regret columns, no run traces)",
    )
```

threading `grade_quality=not args.no_quality` into the `run_arm` call in `run_matrix`, following how `measure_power` is already threaded.

Add the same `--no-quality` flag to `src/tetris_agent/cli.py` for single runs, passing `grader=None` when set.

`benchmark.py` has no runs-directory constant yet. Add one beside the existing
`RESULTS_DIR`, matching the `runs` default the CLI already uses, and import the recorder:

```python
from tetris_agent.recorder import RunRecorder

RUNS_DIR = Path("runs")
```

`runs/` is already gitignored, so the traces stay on the machine that produced them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: everything passes except the known macOS-only power test.

- [ ] **Step 5: Verify the isolation guarantee by hand**

Run a two-arm matrix twice, once with grading and once without, and confirm the outcome columns match:

```bash
uv run tetris-bench --models pi/gemma4:26b --harnesses features --efforts off \
  --fixed-effort --seeds 1 --max-pieces 30 --no-control --no-power --no-quality
uv run tetris-bench --models pi/gemma4:26b --harnesses features --efforts off \
  --fixed-effort --seeds 1 --max-pieces 30 --no-control --no-power
```

Expected: identical `race_score`, `score`, `lines`, `pieces`, `avg_holes`. The second run additionally reports `regret`, `top1`, `top3` and writes one directory under `runs/`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/tetris_agent/benchmark.py src/tetris_agent/cli.py tests/test_benchmark.py tests/test_traces.py
git commit -m "feat: grade benchmark arms and record their traces

Every arm now writes an events-only trace under runs/ and reports regret,
top1 and top3. --no-quality turns the whole feature off. Verified by hand that
a graded and an ungraded run produce identical outcome columns."
```

---

### Task 9: Document it

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Write the README section**

Add to `README.md`, after the "The clock" section:

```markdown
### Placement quality

Every decision is graded against a two-ply oracle: the same weighted
evaluation the control arm plays with, searched one piece deeper. A row
reports `regret` (0 is the oracle's own choice, 1 is the worst legal
placement), `top1` and `top3`.

The oracle is a strong heuristic, not ground truth. Regret is disagreement with
a two-ply evaluator, so a model that outplays the oracle shows high regret; the
`--lookahead-control` arm puts the oracle's own score in the same table so the
comparison is visible. Regret is also per-move and path-relative: each decision
is graded against the board the model built, so a ruined board played well
scores near zero. Accumulated damage shows up in lines, holes and pieces
survived instead.

Grades are written to `runs/<id>/events.jsonl` as `placement_graded` events,
alongside the spawn, decision and lock events the exemplar miner already reads.
`--no-quality` turns all of it off.
```

- [ ] **Step 2: Add the convention to the benchmarks index**

Add to `benchmarks/README.md`, in the section describing columns:

```markdown
Rows carrying `regret` / `top1` / `top3` are graded against the two-ply oracle
in `quality.py`, with the genome recorded per decision. State that in the
write-up, and put a `--lookahead-control` row in the same table: regret is
disagreement with the oracle, not distance from optimal, and a model can score
above the oracle's own arm.
```

- [ ] **Step 3: Commit**

```bash
git add README.md benchmarks/README.md
git commit -m "docs: placement quality columns and what they mean

Regret is disagreement with a two-ply heuristic, not distance from optimal,
and it is per-move rather than cumulative. Both are easy to misread, so both
are stated where the columns are introduced."
```

---

## Definition of done

- `uv run pytest -q` passes, except the known macOS-only power test.
- `uv run ruff check src tests` reports only the pre-existing `tests/test_live_agent.py` error.
- A graded and an ungraded run of the same command produce identical `race_score`, `score`, `lines`, `pieces` and `avg_holes`.
- `expand_arms` output for an existing command is unchanged; `lookahead` appears only with its flag.
- A `--lookahead-control` arm reports `regret` 0.0 and `top1` 1.0.
- `runs/<id>/events.jsonl` from a benchmark arm mines to the same exemplars as an ungraded run of the same game.
