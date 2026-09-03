# Situation Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic board classifier (five situations, explicit priority) feeding a new `routed` harness arm, validated on a free corpus, then benchmarked across models.

**Architecture:** `situation.py` is a pure module over the settled board (like `board.py`) that names the situation and renders the one technique it calls for; thresholds live in `Genome`. `prompts.py` gains a `routed` harness that shows the legal placements with only the numbers that situation turns on, under a system prompt whose general play guidance is replaced by "follow the named technique". `LLMPlacementPolicy` (base of both the Anthropic and pi arms) computes the situation per piece. A corpus command records free `heuristic`/`random` runs and histograms the classes over their replayed boards; a benchmark sweep compares `routed` against `legal` and `features`.

**Tech Stack:** Python 3.11+, NumPy, pytest, `uv`, PyBoy (integration tests and corpus only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-02-situation-router-design.md`

## Global Constraints

- `SYSTEM_PROMPT` in `src/tetris_agent/prompts.py` must remain **byte-identical** — every published `features`/`legal`/`board`/`chat` row was produced with it, and `tests/test_model_policy.py:160` asserts it is sent unchanged.
- `HARNESSES` becomes `("board", "legal", "features", "chat", "routed")` — append only.
- New `Genome` fields are ints ≥ 1 (`evolve.sample_variants` jitters ints by ±1 with a floor of 1).
- Line length 120 (`ruff`, `pyproject.toml`). Run `uv run ruff check src tests` before every commit.
- Full suite command: `uv run pytest -q`. **One failure is pre-existing and unrelated**: `tests/test_power.py::test_scoped_sudoers_entry_counts_as_available` (a macOS `powermetrics` sudo probe; fails identically on `main`). Expected result after each task is "1 failed, N passed" with N growing. Do not fix that test.
- Work happens in this worktree on branch `worktree-situation-router`. Commit after each task with the message given; end every commit message with the two trailer lines shown in Task 1.
- `data/` and `runs/` are gitignored. Results that must persist go under `benchmarks/` as dated markdown (append-only; see `benchmarks/README.md`).
- Today is 2026-09-02 (UTC). Benchmark files are dated by their run day in UTC.

## Deviations from the spec (decided while planning, with reasons)

1. **`TETRIS_READY` does not require an I in the queue.** The spec's condition included "I current or next". A ready well is a board state; the *technique* branches on the queue (I now → drop it in; I next → don't cover the well; neither → don't cover the well). Without this, a ready board with no I in view fell through to `MOUND` and was told to flatten — i.e. fill its own well.
2. **`MOUND` uses height range (highest column − lowest column ≥ `mound_rise`, default 3), not the bumpiness sum.** Neubauer's rule is "never more than two high"; range is that rule directly. No side bias (left vs centre) in v1 — it is a benchmark-phase A/B, not a classifier property. The metric that triggers MOUND (height range) is not the metric the routed prompt shows for it (`bumpiness`), a known asymmetry recorded in the 09-03 router write-up's follow-ups.
3. **`HOLE_RISK` means "every legal placement of this O/S/Z opens at least one hole".** That is the purpose of Neubauer's two-wide-gap rule, computed exactly and cheaply instead of approximated by surface shape.
4. **`routed` builds on `legal`, not `features`.** Each placement is annotated with only the situation's metrics, so the prompt sits between `legal` and `features` in size. The 08-22 screen found local models' wall is *prefill* on the ~2k-token board prompt, so shorter matters for local arms.
5. **Success metric is refined.** `tokens_per_decision` (`model_policy.py:111`) is *output* tokens per decision, so the hypothesis under test is "a named technique makes the model deliberate less": output tokens and `latency_ms_mean` fall, `race`/`score` do not regress, versus both `legal` and `features` on identical seeds.

---

### Task 1: `situation.py` — kinds, thresholds in the genome, and `classify`

**Files:**
- Create: `src/tetris_agent/situation.py`
- Modify: `src/tetris_agent/policy.py:17-33` (the `Genome` dataclass)
- Test: `tests/test_situation.py`

**Interfaces:**
- Consumes: `board.COLS`, `board.ROWS`, `board.features`, `board.drop`, `board.place`; `pieces.SHAPES`, `pieces.distinct_rotations`, `pieces.shape_width`; `policy.Genome`.
- Produces (later tasks rely on these exact names):
  - `class Kind(str, Enum)` with members `TOPPING_OUT, TETRIS_READY, HOLE_RISK, MOUND, FLAT` (values equal their names).
  - `@dataclass(frozen=True) class Situation: kind: Kind; well_col: int | None = None`.
  - `METRICS: dict[Kind, tuple[str, ...]]` — per kind, the `LegalPlacement` numbers the routed prompt shows (`"lines"` is on the placement; `"holes"`, `"bumpiness"`, `"max_height"` are on `.features`).
  - `classify(board: np.ndarray, piece: str, next_piece: str, genome: Genome | None = None) -> Situation`.
  - `situation_block(situation: Situation, board: np.ndarray, piece: str, next_piece: str) -> str` — the "Situation: …" paragraph.
  - Helpers `column_heights(board) -> np.ndarray`, `tetris_well(board, heights, depth) -> int | None`, `min_new_holes(board, piece) -> int | None`.
  - `Genome` gains `topping_out_height: int = 12`, `mound_rise: int = 3`, `well_depth: int = 4`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_situation.py`:

```python
"""The deterministic left-hand side of the routing table: five situations, explicit priority."""

import numpy as np

from tetris_agent.policy import Genome
from tetris_agent.situation import (
    METRICS,
    Kind,
    Situation,
    classify,
    column_heights,
    min_new_holes,
    situation_block,
    tetris_well,
)


def board_from(*rows: str) -> np.ndarray:
    """An 18x10 board from its bottom rows, written top-first the way prompts render them."""
    b = np.zeros((18, 10), dtype=bool)
    for i, row in enumerate(reversed(rows)):
        assert len(row) == 10
        b[17 - i] = [ch == "#" for ch in row]
    return b


TETRIS_READY = board_from(*(["#########."] * 4))
SAW = board_from("#.#.#.#.#.")
MOUND = board_from(*(["###......."] * 3))


def test_column_heights_counts_from_the_floor():
    b = board_from("#.........", "#.#.......")
    assert column_heights(b).tolist() == [2, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def test_empty_board_is_flat():
    assert classify(board_from(), "T", "O") == Situation(Kind.FLAT)


def test_tetris_ready_names_the_well_regardless_of_the_queue():
    assert classify(TETRIS_READY, "T", "O") == Situation(Kind.TETRIS_READY, well_col=9)
    assert tetris_well(TETRIS_READY, column_heights(TETRIS_READY), 4) == 9


def test_tetris_ready_beats_mound():
    # The same board has a height range of 4 (>= mound_rise); the well wins the priority.
    assert classify(TETRIS_READY, "T", "O").kind is Kind.TETRIS_READY


def test_three_deep_well_is_not_ready():
    assert classify(board_from(*(["#########."] * 3)), "I", "O").kind is Kind.MOUND


def test_topping_out_beats_tetris_ready():
    tall = board_from(*(["#########."] * 12))
    assert classify(tall, "I", "O") == Situation(Kind.TOPPING_OUT)


def test_hole_risk_when_no_placement_of_o_avoids_a_hole():
    assert min_new_holes(SAW, "O") == 1
    assert classify(SAW, "O", "T") == Situation(Kind.HOLE_RISK)


def test_hole_risk_is_only_for_o_s_z():
    assert classify(SAW, "T", "O") == Situation(Kind.FLAT)  # range 1 < mound_rise 3


def test_o_with_a_flat_pair_is_not_at_risk():
    assert min_new_holes(board_from("##........"), "O") == 0
    assert classify(board_from("##........"), "O", "T") == Situation(Kind.FLAT)


def test_mound_from_height_range():
    assert classify(MOUND, "T", "O") == Situation(Kind.MOUND)


def test_thresholds_come_from_the_genome():
    assert classify(MOUND, "T", "O", Genome(mound_rise=5)).kind is Kind.FLAT
    assert classify(MOUND, "T", "O", Genome(topping_out_height=3)).kind is Kind.TOPPING_OUT
    assert classify(board_from(*(["#########."] * 3)), "T", "O", Genome(well_depth=3)).kind is Kind.TETRIS_READY


def test_genome_carries_the_thresholds():
    params = Genome().to_params()
    assert (params["topping_out_height"], params["mound_rise"], params["well_depth"]) == (12, 3, 4)
    assert Genome.from_params({**params, "well_depth": 5}).well_depth == 5


def test_every_kind_has_metrics():
    for kind in Kind:
        assert METRICS[kind]
    assert METRICS[Kind.HOLE_RISK] == ("holes",)


def test_situation_block_states_the_technique_for_the_queue():
    ready = Situation(Kind.TETRIS_READY, well_col=9)
    assert "drop it vertically" in situation_block(ready, TETRIS_READY, "I", "O")
    assert "next piece is an I" in situation_block(ready, TETRIS_READY, "T", "I")
    assert "without covering column 9" in situation_block(ready, TETRIS_READY, "T", "O")
    assert "this O" in situation_block(Situation(Kind.HOLE_RISK), SAW, "O", "T")
    assert "spans 3 rows" in situation_block(Situation(Kind.MOUND), MOUND, "T", "O")
    assert "12 rows high" in situation_block(Situation(Kind.TOPPING_OUT), board_from(*(["#########."] * 12)), "T", "O")
    assert situation_block(Situation(Kind.FLAT), board_from(), "T", "O").startswith("Situation: FLAT")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_situation.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tetris_agent.situation'`.

- [ ] **Step 3: Add the thresholds to `Genome`**

In `src/tetris_agent/policy.py`, inside `class Genome`, after `ticks_per_press: int = 4`:

```python
    # Situation thresholds (situation.py) for the routed harness; tuned like any other knob.
    topping_out_height: int = 12  # max column height that flips play into survival mode
    mound_rise: int = 3  # highest column minus lowest that counts as a mound
    well_depth: int = 4  # rows a one-column well must span to be Tetris-ready
```

- [ ] **Step 4: Write `situation.py`**

Create `src/tetris_agent/situation.py`:

```python
"""Board situations — the deterministic left-hand side of the routing table.

Neubauer's guidance is a small set of board states, each with a prescribed move.
`classify` names the state; `situation_block` renders the move for the `routed`
harness. Pure over the settled board, like board.py: no emulator, no model, no
I/O. Thresholds come from the Genome so evolve.py can tune them.

Priority is explicit and top-down — first match wins (see docs/superpowers/specs/
2026-09-02-situation-router-design.md, section 04). Real boards are mixtures, so
the order is a decision under test, not a derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tetris_agent.board import COLS, ROWS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.policy import Genome

HOLE_PRONE = ("O", "S", "Z")


class Kind(str, Enum):
    TOPPING_OUT = "TOPPING_OUT"
    TETRIS_READY = "TETRIS_READY"
    HOLE_RISK = "HOLE_RISK"
    MOUND = "MOUND"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Situation:
    kind: Kind
    well_col: int | None = None  # TETRIS_READY only: the column the I goes in


# The consequence numbers the routed harness shows per placement. Names are
# LegalPlacement fields: `lines` on the placement itself, the rest on `.features`.
METRICS: dict[Kind, tuple[str, ...]] = {
    Kind.TOPPING_OUT: ("lines", "max_height"),
    Kind.TETRIS_READY: ("lines",),
    Kind.HOLE_RISK: ("holes",),
    Kind.MOUND: ("bumpiness",),
    Kind.FLAT: ("holes", "bumpiness"),
}


def column_heights(board: np.ndarray) -> np.ndarray:
    """Cells from the floor up to the topmost settled cell, per column (0 when empty)."""
    heights = np.zeros(COLS, dtype=int)
    for c in range(COLS):
        filled = np.flatnonzero(board[:, c])
        if filled.size:
            heights[c] = ROWS - filled[0]
    return heights


def tetris_well(board: np.ndarray, heights: np.ndarray, depth: int) -> int | None:
    """A column whose `depth` lowest empty rows are full in every other column, or None."""
    for w in range(COLS):
        floor = ROWS - 1 - int(heights[w])  # lowest empty row in this column
        top = floor - depth + 1
        if top < 0:
            continue
        rows = board[top : floor + 1]
        if np.delete(rows, w, axis=1).all() and not rows[:, w].any():
            return w
    return None


def min_new_holes(board: np.ndarray, piece: str) -> int | None:
    """Fewest holes any legal placement of `piece` opens; None if nothing fits."""
    base = features(board).holes
    best: int | None = None
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is None:
                continue
            new_board, _ = place(board, shape, col, row)
            opened = max(0, features(new_board).holes - base)
            if best is None or opened < best:
                best = opened
            if best == 0:
                return 0
    return best


def classify(board: np.ndarray, piece: str, next_piece: str, genome: Genome | None = None) -> Situation:
    """Name the board's situation. First match in priority order wins."""
    g = genome or Genome()
    heights = column_heights(board)
    if int(heights.max()) >= g.topping_out_height:
        return Situation(Kind.TOPPING_OUT)
    well = tetris_well(board, heights, g.well_depth)
    if well is not None:
        return Situation(Kind.TETRIS_READY, well_col=well)
    if piece in HOLE_PRONE and (min_new_holes(board, piece) or 0) >= 1:
        return Situation(Kind.HOLE_RISK)
    if int(heights.max() - heights.min()) >= g.mound_rise:
        return Situation(Kind.MOUND)
    return Situation(Kind.FLAT)


def situation_block(situation: Situation, board: np.ndarray, piece: str, next_piece: str) -> str:
    """The one technique this turn calls for, as the routed user prompt states it."""
    kind = situation.kind
    heights = column_heights(board)
    if kind is Kind.TOPPING_OUT:
        return (
            f"Situation: TOPPING_OUT — the stack is {int(heights.max())} rows high on an {ROWS}-row board. "
            "Survive: take a placement that clears lines if one exists; otherwise the one leaving the "
            "lowest max height. Do not set anything up."
        )
    if kind is Kind.TETRIS_READY:
        w = situation.well_col
        lead = f"Situation: TETRIS_READY — the well at column {w} is ready: the rows beside it are full. "
        if piece == "I":
            return lead + f"This piece is an I: drop it vertically (the width-1 rotation) in column {w}."
        if next_piece == "I":
            return lead + f"The next piece is an I: place this piece without covering column {w}."
        return lead + f"No I is in view yet: place this piece without covering column {w}, keeping the rest flat."
    if kind is Kind.HOLE_RISK:
        return (
            f"Situation: HOLE_RISK — this {piece} cannot land anywhere without opening a hole. "
            "Choose the placement that opens the fewest, as low on the board as possible."
        )
    if kind is Kind.MOUND:
        rise = int(heights.max() - heights.min())
        return (
            f"Situation: MOUND — the surface spans {rise} rows between its lowest and highest columns. "
            "Flatten: choose the placement that lowers bumpiness most without opening a hole."
        )
    return (
        "Situation: FLAT — the surface is even. Keep it that way: no column more than two above its "
        "neighbours, no gap deeper than two; prefer the least bumpiness and no new holes."
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_situation.py -q`
Expected: `14 passed`.

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests`
Expected: `1 failed, 275 passed` (the pre-existing `test_power` failure only) and `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add src/tetris_agent/situation.py src/tetris_agent/policy.py tests/test_situation.py
git commit -m "$(cat <<'EOF'
feat: situation classifier — five board situations with thresholds in the genome

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

---

### Task 2: the `routed` harness in `prompts.py`

**Files:**
- Modify: `src/tetris_agent/prompts.py` (`HARNESSES` at line 16; `SYSTEM_PROMPT` at lines 28-72; `build_user_prompt` at lines 123-171)
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `situation.METRICS`, `situation.Situation`, `situation.classify`, `situation.situation_block` (Task 1).
- Produces:
  - `HARNESSES == ("board", "legal", "features", "chat", "routed")`.
  - `ROUTED_SYSTEM_PROMPT: str` and `system_prompt_for(harness: str) -> str` (returns `SYSTEM_PROMPT` for every harness except `routed`).
  - `build_user_prompt(harness, board, piece, next_piece, placements, turn, deadline_s=None, situation=None)` — new keyword `situation: Situation | None`; for `routed`, `None` means "classify with default thresholds".

- [ ] **Step 1: Snapshot `SYSTEM_PROMPT` so byte-identity can be checked after the edit**

Run:
```bash
uv run python -c "from tetris_agent.prompts import SYSTEM_PROMPT; open('/tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/system_prompt_before.txt','w').write(SYSTEM_PROMPT)"
```
Expected: no output, file written.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
def test_routed_prompt_names_the_situation_and_scopes_the_numbers():
    board = empty_board()
    legal = legal_placements(board, "T")
    prompt = build_user_prompt("routed", board, "T", "O", legal, 1)
    assert "Situation: FLAT" in prompt
    assert "-> holes 0, bumpiness" in prompt  # FLAT's metrics, nothing else
    assert "aggregate height" not in prompt
    assert "Choose one of them." in prompt


def test_routed_prompt_accepts_a_precomputed_situation():
    from tetris_agent.situation import Kind, Situation

    board = empty_board()
    legal = legal_placements(board, "O")
    prompt = build_user_prompt("routed", board, "O", "I", legal, 1, situation=Situation(Kind.HOLE_RISK))
    assert "Situation: HOLE_RISK" in prompt
    listing = prompt.split("Legal placements")[1]
    assert "-> holes" in listing and "bumpiness" not in listing


def test_routed_sits_between_legal_and_features_in_length():
    board = empty_board()
    legal = legal_placements(board, "T")
    sizes = {h: len(build_user_prompt(h, board, "T", "I", legal, 1)) for h in ("legal", "routed", "features")}
    assert sizes["legal"] < sizes["routed"] < sizes["features"]


def test_system_prompt_for_only_swaps_for_routed():
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT, SYSTEM_PROMPT, system_prompt_for

    assert system_prompt_for("features") is SYSTEM_PROMPT
    assert system_prompt_for("chat") is SYSTEM_PROMPT
    assert system_prompt_for("routed") is ROUTED_SYSTEM_PROMPT
    assert "# What good play looks like" in SYSTEM_PROMPT
    assert "# What good play looks like" not in ROUTED_SYSTEM_PROMPT
    assert "# How to play" in ROUTED_SYSTEM_PROMPT
    assert ROUTED_SYSTEM_PROMPT.startswith("You are playing Game Boy Tetris")
    assert ROUTED_SYSTEM_PROMPT.endswith("one short sentence of reasoning.")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: 4 failures — `ValueError: unknown harness 'routed'` for the first three, `ImportError` (no `ROUTED_SYSTEM_PROMPT`) for the last.

- [ ] **Step 4: Split `SYSTEM_PROMPT` into three constants and add the routed variant**

In `src/tetris_agent/prompts.py`, replace the single `SYSTEM_PROMPT = """\ … """` assignment with three constants concatenated. The text of `_PREAMBLE` is everything from `You are playing Game Boy Tetris.` through `is illegal and will be rejected.` **plus the blank line after it**; `_GOOD_PLAY` is the `# What good play looks like` heading through `Survival beats greed.` **plus the blank line after it**; `_CLOSING` is the final sentence with no trailing newline.

```python
# Static across every decision in a run, so it caches. Keep the volatile board
# out of here — it goes in the user turn, after the cache breakpoint.
_PREAMBLE = """\
You are playing Game Boy Tetris. You choose where each falling piece lands; a
controller executes your choice exactly. Your goal is to survive as long as
possible and clear as many lines as you can.

# The board

The playfield is 18 rows by 10 columns. Row 0 is the top, row 17 is the floor.
Column 0 is the far left, column 9 is the far right. Boards are drawn with `.`
for an empty cell and `#` for a settled block, one line per row, top row first.

# Pieces and rotation

The seven tetrominoes are L, J, I, O, Z, S, T. Each has up to four rotation
states, numbered 0 to 3. Rotation 0 is the spawn orientation. Rotation values
count counter-clockwise: pressing the A button once takes rotation 0 to
rotation 3, then 3 to 2, and so on. Pieces with symmetry have fewer distinct
silhouettes — O has one, and I, S, and Z have two, so rotations 2 and 3 there
duplicate 0 and 1.

# Your decision

For each piece you pick two numbers:

- `rotation`: which rotation state to lock the piece in (0 to 3).
- `col`: the leftmost column the rotated piece occupies (0 to 9).

The piece then drops straight down from that column and locks where it lands.
`col` refers to the left edge of the piece's bounding box, so a piece four
cells wide can only use columns 0 through 6. A choice whose piece would hang
off the right edge, or that cannot fit at all, is illegal and will be rejected.

"""

_GOOD_PLAY = """\
# What good play looks like

- Clearing lines is the only way to score, and the only way to reduce height.
- A hole is an empty cell with a settled block somewhere above it in the same
  column. Holes are expensive: they cannot be filled without first clearing
  every row above them. Avoid creating them.
- Keep the stack low and the surface flat. Deep single-column wells are only
  worth it when you are deliberately setting up an I piece.
- You are told the next piece as well as the current one. Use it — leaving a
  spot the next piece fits is usually better than a locally optimal placement
  that strands it.
- Losing happens when the stack reaches the top. Survival beats greed.

"""

# The routed harness names the situation and the technique each turn, so the
# general guidance above is replaced by "follow it" — the whole point is that
# the model spends its output on the placement, not on re-deriving strategy.
_ROUTED_PLAY = """\
# How to play

Each turn names the board's situation and the one technique it calls for, then
lists the legal placements with only the numbers that technique cares about.
Follow the technique rather than re-deriving strategy from the board. Losing
happens when the stack reaches the top; survival beats greed.

"""

_CLOSING = "Respond with your chosen placement and one short sentence of reasoning."

SYSTEM_PROMPT = _PREAMBLE + _GOOD_PLAY + _CLOSING
ROUTED_SYSTEM_PROMPT = _PREAMBLE + _ROUTED_PLAY + _CLOSING


def system_prompt_for(harness: str) -> str:
    """The cached system prompt for a harness; only `routed` swaps the play guidance."""
    return ROUTED_SYSTEM_PROMPT if harness == "routed" else SYSTEM_PROMPT
```

Then update the imports and `HARNESSES`:

```python
from tetris_agent.board import COLS, Features, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.situation import METRICS, Situation, classify, situation_block

HARNESSES = ("board", "legal", "features", "chat", "routed")
```

- [ ] **Step 5: Verify `SYSTEM_PROMPT` is byte-identical**

Run:
```bash
uv run python -c "from tetris_agent.prompts import SYSTEM_PROMPT; open('/tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/system_prompt_after.txt','w').write(SYSTEM_PROMPT)" && cmp /tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/system_prompt_before.txt /tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/system_prompt_after.txt && echo IDENTICAL
```
Expected: `IDENTICAL`. If `cmp` reports a difference, fix the split (most likely a missing or extra blank line at a boundary) before going on.

- [ ] **Step 6: Add the routed branch to `build_user_prompt`**

Change the signature and add a branch before the final `else` (the `features` branch):

```python
def _metric(p: LegalPlacement, name: str) -> int:
    return p.lines if name == "lines" else getattr(p.features, name)


def build_user_prompt(
    harness: str,
    board: np.ndarray,
    piece: str,
    next_piece: str,
    placements: list[LegalPlacement],
    turn: int,
    deadline_s: float | None = None,
    situation: Situation | None = None,
) -> str:
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r}; known: {HARNESSES}")

    f = features(board)
    header = (
        f"Piece {turn}. Current piece: {piece}. Next piece: {next_piece}.\n\n"
        f"Board (row 0 at top):\n{render_board(board)}\n\n"
        f"Stack: max height {f.max_height}, {f.holes} holes, bumpiness {f.bumpiness}.\n\n"
        f"Shapes for {piece}:\n{_shapes_block(piece)}"
    )

    if harness in ("board", "chat"):
        body = header + "\n\nChoose a rotation and column."
    elif harness == "legal":
        lines = [f"  rotation={p.rotation} col={p.col}" for p in placements]
        body = (
            header
            + "\n\nLegal placements:\n"
            + "\n".join(lines)
            + "\n\nChoose one of them."
        )
    elif harness == "routed":
        situation = situation or classify(board, piece, next_piece)
        scoped = [
            f"  rotation={p.rotation} col={p.col} -> "
            + ", ".join(f"{m.replace('_', ' ')} {_metric(p, m)}" for m in METRICS[situation.kind])
            for p in placements
        ]
        body = (
            header
            + "\n\n"
            + situation_block(situation, board, piece, next_piece)
            + "\n\nLegal placements, with the numbers this situation turns on:\n"
            + "\n".join(scoped)
            + "\n\nChoose one of them."
        )
    else:
        detailed = [
            f"  rotation={p.rotation} col={p.col} -> clears {p.lines} line(s), "
            f"holes {p.features.holes}, aggregate height {p.features.agg_height}, "
            f"bumpiness {p.features.bumpiness}, max height {p.features.max_height}"
            for p in placements
        ]
        body = (
            header
            + "\n\nLegal placements, each with the resulting board after it locks:\n"
            + "\n".join(detailed)
            + "\n\nChoose one of them. The numbers describe consequences, not a ranking — "
            "weigh them yourself."
        )
    if deadline_s is not None:
        # Live mode: the game does not pause. This stays in the (uncached)
        # user turn — the system prompt must remain byte-identical.
        body += f"\n\nYou have about {deadline_s:.0f} seconds before this piece locks. Answer immediately."
    return body
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py tests/test_situation.py -q`
Expected: all pass (the parametrized `test_every_harness_includes_board_and_pieces` now runs 5 cases).

- [ ] **Step 8: Full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests`
Expected: `1 failed, 280 passed` (pre-existing `test_power` failure only) and `All checks passed!`.

- [ ] **Step 9: Commit**

```bash
git add src/tetris_agent/prompts.py tests/test_prompts.py
git commit -m "$(cat <<'EOF'
feat: routed harness — situation-scoped placements under a follow-the-technique system prompt

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

---

### Task 3: wire routing into the policies and the launchers

**Files:**
- Modify: `src/tetris_agent/model_policy.py` (imports at lines 20-27; `LLMPlacementPolicy.__init__` lines 36-81; `plan` lines 85-97; `ModelPolicy.__init__` lines 200-219)
- Modify: `src/tetris_agent/pi_policy.py:185-210` (`PiPolicy.__init__`)
- Modify: `src/tetris_agent/benchmark.py:109-132` (`build_policy`)
- Modify: `src/tetris_agent/cli.py:86-105` (`build_policy`)
- Test: `tests/test_model_policy.py`, `tests/test_pi_policy.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `prompts.system_prompt_for`, `prompts.build_user_prompt(..., situation=)` (Task 2); `situation.classify` (Task 1); `policy.Genome`.
- Produces: `LLMPlacementPolicy(..., genome: Genome | None = None)` with attributes `.genome` and `.last_situation: Situation | None`; `ModelPolicy` and `PiPolicy` accept and forward `genome=`; `benchmark.build_policy` and `cli.build_policy` pass the run's genome into model arms.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_policy.py`:

```python
def test_routed_harness_swaps_the_system_prompt_and_names_the_situation():
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT

    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "flat"})], harness="routed")
    p.plan(empty_board(), "T", "O", turn=1)
    call = p.client.messages.calls[0]
    assert call["system"][0]["text"] == ROUTED_SYSTEM_PROMPT
    assert "Situation: FLAT" in call["messages"][-1]["content"]
    assert p.last_situation.kind.value == "FLAT"


def test_routed_thresholds_come_from_the_policy_genome():
    from tetris_agent.policy import Genome

    p = policy(
        [FakeResponse({"rotation": 0, "col": 3, "reason": "x"})],
        harness="routed",
        genome=Genome(topping_out_height=1),
    )
    board = empty_board()
    board[17, 0] = True
    p.plan(board, "T", "O", turn=1)
    assert "Situation: TOPPING_OUT" in p.client.messages.calls[0]["messages"][-1]["content"]


def test_non_routed_harness_has_no_situation():
    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "x"})], harness="features")
    p.plan(empty_board(), "T", "O", turn=1)
    assert p.last_situation is None
    assert "Situation:" not in p.client.messages.calls[0]["messages"][-1]["content"]
```

Append to `tests/test_pi_policy.py`:

```python
def test_routed_system_prompt_and_situation_reach_pi():
    import subprocess

    import numpy as np

    from tetris_agent.pi_policy import PiPolicy
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT

    seen = {}

    def runner(cmd, timeout):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    p = PiPolicy(model="pi/gemma3", runner=runner, harness="routed")
    p.plan(np.zeros((18, 10), dtype=bool), "T", "O", turn=1)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1].startswith(ROUTED_SYSTEM_PROMPT)
    assert "Situation: FLAT" in cmd[-1]
```

In `tests/test_benchmark.py`, find the existing test near line 205-211 that builds a model policy through `build_policy` and asserts `"EXEMPLARS" in p.system_prompt`. Add `genome_params={"well_depth": 5}` to its `build_policy(...)` call and append the assertion `assert p.genome.well_depth == 5` to that test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_model_policy.py tests/test_pi_policy.py tests/test_benchmark.py -q 2>&1 | tail -8`
Expected: the three new model-policy tests fail (`TypeError: unexpected keyword 'genome'` / `AttributeError: last_situation` / system prompt mismatch), the pi test fails on the system-prompt assertion, and the benchmark test fails with `TypeError: build_policy() got an unexpected keyword argument` or on `.genome`.

- [ ] **Step 3: Update `LLMPlacementPolicy`**

In `src/tetris_agent/model_policy.py`, change the imports:

```python
from tetris_agent.policy import Genome, Placement
from tetris_agent.pricing import EFFORTS, cost_usd, spec
from tetris_agent.prompts import (
    PLACEMENT_SCHEMA,
    build_user_prompt,
    legal_placements,
    system_prompt_for,
)
from tetris_agent.situation import Situation, classify
```

Change `__init__`'s signature and the two lines that set the system prompt / history:

```python
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        max_tokens: int = MAX_TOKENS,
        exemplar_block: str = "",
        clock=time.monotonic,
        genome: Genome | None = None,
    ):
        self.model = model
        self.harness = harness
        self.spec = spec(model)
        # Haiku 4.5 rejects output_config.effort; its arms are effort-free by construction.
        self.effort = effort if self.spec.supports_effort else None
        self.max_tokens = max_tokens
        self._clock = clock
        # Situation thresholds for the routed harness — the same genome the heuristic plays with.
        self.genome = genome or Genome()
        # Exemplars are static across the run, so they belong in the system
        # prompt — cached — never in the per-piece user turn.
        self.system_prompt = system_prompt_for(harness) + (f"\n\n{exemplar_block}" if exemplar_block else "")
        self.name = f"{model}/{harness}" + (f"/{self.effort}" if self.effort else "")
        self._history: list[dict] = []
        self.last_situation: Situation | None = None
```

(Keep everything after `self._history` — the `usage` dict and the remaining attributes — exactly as it is.)

Change `plan`:

```python
    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        legal = legal_placements(board, piece)
        if not legal:
            return None

        situation = classify(board, piece, next_piece, self.genome) if self.harness == "routed" else None
        self.last_situation = situation
        prompt = build_user_prompt(
            self.harness, board, piece, next_piece, legal, turn, deadline_s=self.deadline_s, situation=situation
        )
        choice = self._decide(prompt, legal)
        if choice is None:
            # Every attempt failed; keep the run alive with a deterministic
            # fallback so the arm is still comparable, and count it as illegal.
            self.usage["illegal_count"] += 1
            choice = legal[0]
        return Placement(rotation=choice.rotation, col=choice.col, score=0.0)
```

Change `ModelPolicy.__init__` to accept and forward the genome:

```python
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        client=None,
        max_tokens: int = MAX_TOKENS,
        exemplar_block: str = "",
        clock=time.monotonic,
        genome: Genome | None = None,
    ):
        super().__init__(model, harness, effort, max_tokens, exemplar_block, clock=clock, genome=genome)
```

- [ ] **Step 4: Update `PiPolicy.__init__`**

In `src/tetris_agent/pi_policy.py`, add `genome=None` as the last parameter and forward it:

```python
        exemplar_block: str = "",
        clock=time.monotonic,
        hard_deadline: bool = False,
        genome=None,
    ):
        if not model.startswith(PI_PREFIX):
            raise ValueError(f"PiPolicy needs a {PI_PREFIX}* model id, got {model!r}")
        super().__init__(model, harness, effort, exemplar_block=exemplar_block, clock=clock, genome=genome)
```

- [ ] **Step 5: Thread the genome through both launchers**

`src/tetris_agent/benchmark.py`, replace `build_policy`:

```python
def build_policy(arm: Arm, genome_params: dict | None = None, exemplar_block: str = ""):
    from tetris_agent.policy import Genome, HeuristicPolicy, NoInputPolicy, RandomPolicy

    genome = Genome.from_params(genome_params or {})
    if arm.policy == "heuristic":
        return HeuristicPolicy(genome)
    if arm.policy == "no-input":
        return NoInputPolicy()
    if arm.policy == "random":
        return RandomPolicy()
    block = exemplar_block if arm.exemplars else ""
    if is_pi(arm.model):
        from tetris_agent.pi_policy import PiPolicy

        return PiPolicy(
            model=arm.model,
            harness=arm.harness,
            effort=arm.effort,
            exemplar_block=block,
            # Bounded pause: the deadline is the whole budget, kill at the mark.
            hard_deadline=arm.deadline_s is not None and not arm.live,
            genome=genome,
        )
    from tetris_agent.model_policy import ModelPolicy

    return ModelPolicy(
        model=arm.model, harness=arm.harness, effort=arm.effort, exemplar_block=block, genome=genome
    )
```

`src/tetris_agent/cli.py`, replace `build_policy`:

```python
def build_policy(cfg: Config):
    from tetris_agent.genome import load_params
    from tetris_agent.policy import Genome, HeuristicPolicy

    genome = Genome.from_params(load_params())
    if cfg.policy == "heuristic":
        return HeuristicPolicy(genome)
    from tetris_agent.pricing import is_pi

    block = ""
    if cfg.exemplars:
        from tetris_agent.traces import load_exemplar_block

        block = load_exemplar_block(cfg.exemplars)
    if is_pi(cfg.model):
        from tetris_agent.pi_policy import PiPolicy

        return PiPolicy(model=cfg.model, harness=cfg.harness, effort=cfg.effort, exemplar_block=block, genome=genome)
    from tetris_agent.model_policy import ModelPolicy

    return ModelPolicy(model=cfg.model, harness=cfg.harness, effort=cfg.effort, exemplar_block=block, genome=genome)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model_policy.py tests/test_pi_policy.py tests/test_benchmark.py -q`
Expected: all pass.

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests`
Expected: `1 failed, 284 passed` (pre-existing `test_power` failure only) and `All checks passed!`.

- [ ] **Step 8: Smoke the CLI plumbing without a model**

Run: `uv run tetris-bench --models pi/gemma3 --harnesses routed --max-pieces 3 --estimate --skip-preflight`
Expected: prints `4 arms x 1 seed(s) x 3 pieces` with an arm named `pi/gemma3/routed+live` and exits 0 (no model is called under `--estimate`).

- [ ] **Step 9: Commit**

```bash
git add src/tetris_agent/model_policy.py src/tetris_agent/pi_policy.py src/tetris_agent/benchmark.py src/tetris_agent/cli.py tests/test_model_policy.py tests/test_pi_policy.py tests/test_benchmark.py
git commit -m "$(cat <<'EOF'
feat: routed arms classify each piece and carry the genome's thresholds through both launchers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

---

### Task 4: `tetris-situations` — free corpus and class histogram

**Files:**
- Create: `src/tetris_agent/situation_corpus.py`
- Modify: `pyproject.toml:16-23` (`[project.scripts]`)
- Test: `tests/test_situation_corpus.py`

**Interfaces:**
- Consumes: `traces.mine_run(run_dir, policies=)` (reconstructs each decision's board with a checksum; returns `Exemplar` objects with `.board`, `.piece`, `.next_piece`); `situation.Kind`, `situation.classify` (Task 1); `agent.TetrisAgent`, `recorder.RunRecorder`, `policy.HeuristicPolicy`, `policy.RandomPolicy`, `emulator.Emulator`.
- Produces:
  - `generate(rom_path, out_dir, policies=("heuristic", "random"), seeds=(0, 1, 2, 3), max_pieces=150) -> list[Path]`.
  - `histogram(runs_dir, genome=None) -> dict[str, Counter]` keyed by policy name, counting `Kind` values.
  - `dominant(counts, threshold=0.9) -> str | None` — the class holding ≥ threshold of all boards.
  - `render(counts) -> str` — markdown table, kinds in priority order plus a `total` row.
  - `main(argv=None) -> int` with sub-commands `generate` and `histogram`; `histogram` exits 2 when `dominant()` fires.
  - Console script `tetris-situations`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_situation_corpus.py`:

```python
"""Situation corpus: free boards from the local policies, histogrammed by kind."""

import json
from collections import Counter

import pytest

from tetris_agent.situation_corpus import dominant, histogram, render


def _envelope(event_type, turn, data):
    return {
        "schema": "tetris.game.v1",
        "event_type": event_type,
        "turn": turn,
        "occurred_at": "2026-09-02T00:00:00+00:00",
        "data": data,
    }


def write_run(runs_dir, run_id, policy, pieces):
    """pieces: list of (piece, next_piece, rotation, col, lines_delta, feature_dict)."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    events = [_envelope("session", 0, {"phase": "start", "policy": policy, "timer_div": 0})]
    for turn, (piece, nxt, rot, col, lines, feats) in enumerate(pieces, start=1):
        events.append(_envelope("piece_spawn", turn, {"piece": piece, "next_piece": nxt}))
        events.append(_envelope("placement_decision", turn, {"rotation": rot, "col": col, "score": 0.0}))
        events.append(_envelope("piece_locked", turn, {"lines_delta": lines, "misexec": 0, "score": 0, **feats}))
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run_dir


# Hand-computed, as in test_traces.py: one O at col 0 -> heights [2,2,0,...]; a second at col 2.
O_AT_0 = {"holes": 0, "agg_height": 4, "bumpiness": 2, "max_height": 2}
THEN_O_AT_2 = {"holes": 0, "agg_height": 8, "bumpiness": 2, "max_height": 2}


def test_histogram_counts_reconstructed_boards_per_policy(tmp_path):
    write_run(
        tmp_path,
        "20260902-000001-aaaaaa",
        "heuristic",
        [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, THEN_O_AT_2)],
    )
    write_run(tmp_path, "20260902-000002-bbbbbb", "random", [("O", "O", 0, 0, 0, O_AT_0)])
    (tmp_path / "20260902-000003-cccccc").mkdir()  # no events.jsonl: ignored
    counts = histogram(tmp_path)
    assert counts["heuristic"] == Counter({"FLAT": 2})
    assert counts["random"] == Counter({"FLAT": 1})


def test_dominant_flags_a_class_at_or_above_the_threshold():
    assert dominant({"heuristic": Counter({"FLAT": 9}), "random": Counter({"MOUND": 1})}) == "FLAT"
    assert dominant({"heuristic": Counter({"FLAT": 5}), "random": Counter({"MOUND": 5})}) is None
    assert dominant({}) is None


def test_render_lists_every_kind_in_priority_order_with_totals():
    table = render({"heuristic": Counter({"FLAT": 3, "MOUND": 1}), "random": Counter({"FLAT": 1})})
    lines = table.splitlines()
    assert lines[0] == "| kind | heuristic | random | total | share |"
    kinds = [line.split("|")[1].strip() for line in lines[2:]]
    assert kinds == ["TOPPING_OUT", "TETRIS_READY", "HOLE_RISK", "MOUND", "FLAT", "total"]
    assert "| FLAT | 3 | 1 | 4 | 80.0% |" in table
    assert "| total | 4 | 1 | 5 | 100.0% |" in table


@pytest.mark.rom
def test_generate_records_replayable_runs(tmp_path, rom_path):
    from tetris_agent.situation_corpus import generate

    runs = generate(rom_path, tmp_path, policies=("heuristic",), seeds=(0,), max_pieces=5)
    assert len(runs) == 1 and (runs[0] / "events.jsonl").is_file()
    counts = histogram(tmp_path)
    assert sum(counts["heuristic"].values()) >= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_situation_corpus.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tetris_agent.situation_corpus'`.

- [ ] **Step 3: Write the module**

Create `src/tetris_agent/situation_corpus.py`:

```python
"""Situation corpus: free boards from the local policies, and the class histogram over them.

No trace with per-decision boards exists on disk (data/benchmarks/ holds only
per-run aggregates), so the corpus is generated: HeuristicPolicy and
RandomPolicy cost nothing, and a recorded run replays into verified boards
through traces.mine_run. Heuristic play gives well-formed boards, random play
gives messy ones — together they bracket what a struggling model arm faces,
without being an LLM's distribution. This validates the classifier's coverage
and its priority conflicts, not its effect on play.

    uv run tetris-situations generate --seeds 0 1 2 3 --max-pieces 150
    uv run tetris-situations histogram            # exits 2 if one class dominates
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tetris_agent.situation import Kind, classify
from tetris_agent.traces import mine_run

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/situations")
POLICIES = ("heuristic", "random")
# Spec section 06: a class holding this share of boards means there is nothing to route.
DOMINANT = 0.9


def generate(
    rom_path,
    out_dir,
    policies: tuple[str, ...] = POLICIES,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    max_pieces: int = 150,
) -> list[Path]:
    """Record one paused, headless run per (policy, seed) under out_dir; return the run dirs."""
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome, HeuristicPolicy, RandomPolicy
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    out_dir = Path(out_dir)
    runs = []
    for name in policies:
        for seed in seeds:
            policy = HeuristicPolicy() if name == "heuristic" else RandomPolicy(seed=seed)
            recorder = RunRecorder(out_dir, label=f"{name}-seed{seed}")
            emu = Emulator(rom_path)
            try:
                agent = TetrisAgent(
                    emu,
                    genome=Genome(),
                    collector=EventCollector(NoopPublisher()),
                    recorder=recorder,
                    max_pieces=max_pieces,
                    policy=policy,
                )
                fitness = agent.run(timer_div=seed)
            finally:
                emu.stop()
            logger.info("%s seed %d: %d pieces", name, seed, fitness.get("pieces_placed", 0))
            runs.append(out_dir / recorder.run_id)
    return runs


def histogram(runs_dir, genome=None) -> dict[str, Counter]:
    """{policy: Counter{kind: n}} over every verified board replayed from runs_dir."""
    run_dirs = [p for p in sorted(Path(runs_dir).iterdir()) if p.is_dir()]
    counts: dict[str, Counter] = {}
    for name in POLICIES:
        c: Counter = Counter()
        for run_dir in run_dirs:
            for ex in mine_run(run_dir, policies=(name,)):
                c[classify(ex.board, ex.piece, ex.next_piece, genome).kind.value] += 1
        counts[name] = c
    return counts


def dominant(counts: dict[str, Counter], threshold: float = DOMINANT) -> str | None:
    """The kind holding at least `threshold` of all boards across policies, or None."""
    total: Counter = Counter()
    for c in counts.values():
        total.update(c)
    n = sum(total.values())
    if not n:
        return None
    kind, top = total.most_common(1)[0]
    return kind if top / n >= threshold else None


def render(counts: dict[str, Counter]) -> str:
    """Markdown table: kinds in priority order as rows, policies plus total and share as columns."""
    names = [p for p in POLICIES if p in counts] or list(counts)
    grand = sum(sum(counts[p].values()) for p in names)
    header = "| kind | " + " | ".join(names) + " | total | share |"
    sep = "|---|" + "---|" * (len(names) + 2)
    rows = []
    for kind in Kind:
        per = [counts[p].get(kind.value, 0) for p in names]
        total = sum(per)
        share = 100 * total / grand if grand else 0.0
        rows.append(f"| {kind.value} | " + " | ".join(str(x) for x in per) + f" | {total} | {share:.1f}% |")
    totals = [sum(counts[p].values()) for p in names]
    rows.append("| total | " + " | ".join(str(x) for x in totals) + f" | {grand} | {100.0 if grand else 0.0:.1f}% |")
    return "\n".join([header, sep, *rows])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="tetris-situations", description="Board-situation corpus and histogram")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="record free heuristic/random runs to replay boards from")
    g.add_argument("--rom", default="rom/tetris.gb")
    g.add_argument("--out", default=str(CORPUS_DIR))
    g.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    g.add_argument("--seeds", nargs="+", type=lambda s: int(s, 0), default=[0, 1, 2, 3])
    g.add_argument("--max-pieces", type=int, default=150)
    h = sub.add_parser("histogram", help="classify every replayed board and tabulate by kind")
    h.add_argument("runs_dir", nargs="?", default=str(CORPUS_DIR))
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        for run in generate(args.rom, args.out, tuple(args.policies), tuple(args.seeds), args.max_pieces):
            print(run)
        return 0

    counts = histogram(args.runs_dir)
    print(render(counts))
    out = Path(args.runs_dir) / "histogram.json"
    out.write_text(json.dumps({k: dict(v) for k, v in counts.items()}, indent=2))
    print(f"\nwritten: {out}")
    top = dominant(counts)
    if top:
        print(f"\nSTOP: {top} holds >= {DOMINANT:.0%} of boards — the table has nothing to route (spec section 06).")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, add to `[project.scripts]` after the `tetris-play` line:

```toml
tetris-situations = "tetris_agent.situation_corpus:main"
```

Then run `uv sync --quiet` so the entry point is installed.

- [ ] **Step 5: Run the unit tests (ROM test is deselected by default)**

Run: `uv run pytest tests/test_situation_corpus.py -q`
Expected: `3 passed, 1 deselected`.

- [ ] **Step 6: Run the ROM integration test explicitly**

Run: `uv run pytest tests/test_situation_corpus.py -q -m rom`
Expected: `1 passed, 3 deselected` (takes a few seconds; boots PyBoy with `rom/tetris.gb`).

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests`
Expected: `1 failed, 287 passed, 9 deselected` (pre-existing `test_power` failure only) and `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/tetris_agent/situation_corpus.py tests/test_situation_corpus.py pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
feat: tetris-situations — free corpus from the local policies and the class histogram over it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

---

### Task 5: run the corpus on this box and record the histogram (the stopping point)

**Files:**
- Create: `benchmarks/2026-09-02-situation-histogram.md`
- Uses: `rom/tetris.gb` (present in this worktree, gitignored), `data/situations/` (gitignored output)

**Interfaces:**
- Consumes: `tetris-situations` (Task 4).
- Produces: the committed histogram, and a go/no-go for Task 7. **If the command exits 2 (one class ≥ 90%), stop after committing the write-up and report — do not run Task 7.**

- [ ] **Step 1: Generate the corpus**

Run (takes a few minutes; heuristic runs go the full 150 pieces, random ones top out earlier):
```bash
uv run tetris-situations generate --seeds 0 1 2 3 --max-pieces 150 2>&1 | tail -12
```
Expected: eight `data/situations/<run-id>` paths printed, preceded by `situation_corpus INFO heuristic seed N: 150 pieces` / `random seed N: <fewer> pieces` lines.

- [ ] **Step 2: Histogram it**

Run:
```bash
uv run tetris-situations histogram; echo "exit=$?"
```
Expected: the markdown table, `written: data/situations/histogram.json`, and `exit=0` — or `exit=2` with a `STOP:` line if a single class holds ≥ 90 %.

- [ ] **Step 3: Write the benchmark file**

Create `benchmarks/2026-09-02-situation-histogram.md`, pasting the table from Step 2 verbatim and filling every `<…>` from the actual output (no placeholders may remain):

```markdown
# 2026-09-02 — situation histogram over a free corpus

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

## Results

<paste the table>

Boards classified: <total>. Pieces per run: heuristic <n> each; random <list per seed>.

## Verdict

Dominant class: <none | KIND at N%>. Stopping rule (≥ 90 % in one class): <not triggered — proceed to the routed benchmark | triggered — the table has nothing to route; benchmark not run>.

Observations: <two or three sentences on what the split looks like — e.g. whether TETRIS_READY
ever fires for random play, whether HOLE_RISK is rare, whether heuristic boards sit almost
entirely in FLAT — and what that implies for the priority order in spec section 04.>
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/2026-09-02-situation-histogram.md
git commit -m "$(cat <<'EOF'
bench: situation histogram over the free heuristic/random corpus

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

- [ ] **Step 5: Report the verdict** — in the task's final message, state the dominant-class result and whether Task 7 is a go.

---

### Task 6: document the harness

**Files:**
- Modify: `README.md:21` (the harness row of the axes table) and the harness bullet list at `README.md:26-33`

- [ ] **Step 1: Update the axes table**

At `README.md:21`, change

```
| **Harness** | `board`, `legal`, `features`, `chat` |
```
to
```
| **Harness** | `board`, `legal`, `features`, `chat`, `routed` |
```

- [ ] **Step 2: Add the bullet**

Read `README.md:26-35` to find the end of the `chat` bullet (it starts at line 31 and may wrap onto the following lines). Immediately after that bullet add:

```
- **`routed`** — `legal`, plus the board's *situation* — `TOPPING_OUT`,
  `TETRIS_READY`, `HOLE_RISK`, `MOUND`, or `FLAT`, in that priority
  (`situation.py`) — with the one technique it calls for and only the
  consequence numbers that technique turns on. The general play guidance in the
  system prompt is replaced by "follow the named technique"; the thresholds are
  genome fields. Run `uv run tetris-situations` to see the class split over a
  free corpus (`benchmarks/2026-09-02-situation-histogram.md`).
```

- [ ] **Step 3: Check the rendering is sane**

Run: `sed -n '18,42p' README.md`
Expected: the table row shows five harnesses; the new bullet sits after `chat` with matching indentation.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: the routed harness

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

---

### Task 7: benchmark — which model plays best under `routed`

Only run if Task 5 reported a go. This task talks to real services (Anthropic API, Ollama, Ollama cloud) and spends a small amount of money; the cap is in the commands.

**Files:**
- Create: `benchmarks/<UTC run date>-situation-router.md` (e.g. `benchmarks/2026-09-02-situation-router.md`; use the day the runs happen, in UTC)
- Reads: `data/benchmarks/benchmark-*.json` written by each run

**Interfaces:**
- Consumes: `tetris-bench --harnesses legal features routed` (Task 3), the 08-22 baselines in `benchmarks/2026-08-22-p15-deadline-screen.md` (seed 1, 30-piece cap, +p15: heuristic 490, random 140, no-input 55).
- Produces: the matrix write-up and a verdict on spec section 08.

Design of the sweep (matches the 08-22 screen so rows are comparable): seed `1`, `--max-pieces 30`, `--decision-deadline 15` (bounded pause, arms labeled `+p15`), effort `medium` where supported, three harnesses per model, `--no-control` (baselines already published).

- [ ] **Step 1: Preflight**

Run:
```bash
[ -n "$ANTHROPIC_API_KEY" ] && echo "anthropic: key set" || echo "anthropic: NO KEY — skip the claude-* arms"
curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo "ollama: up" || echo "ollama: DOWN — start it before the pi arms"
ollama list 2>/dev/null | grep -E "cloud|gpt-oss:20b" || echo "no cloud stubs / gpt-oss:20b pulled"
nc -z 127.0.0.1 42345 && echo "tapes proxy: up" || echo "tapes proxy: down (fine — capture is optional)"
```
Expected: four status lines. Adjust the model lists below to what is actually available; note anything skipped in the write-up.

- [ ] **Step 2: Project the spend**

Run:
```bash
uv run tetris-bench --models claude-haiku-4-5 claude-sonnet-5 pi/gpt-oss:120b-cloud \
  --harnesses legal features routed --efforts medium --seeds 1 --max-pieces 30 \
  --decision-deadline 15 --no-control --estimate
```
Expected: `9 arms x 1 seed(s) x 30 pieces`, arm names ending `+p15`, `projected cost: ~$<under 1.00>`.

- [ ] **Step 3: Run the cloud and API arms**

Run (allow ~30–60 minutes; each timed-out decision costs up to 15 s):
```bash
uv run tetris-bench --models claude-haiku-4-5 claude-sonnet-5 pi/gpt-oss:120b-cloud \
  --harnesses legal features routed --efforts medium --seeds 1 --max-pieces 30 \
  --decision-deadline 15 --max-usd 3.00 --no-control --no-power 2>&1 | tee /tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/bench-cloud.log | tail -20
```
Expected: a ranked table with nine rows and `results: data/benchmarks/benchmark-<stamp>.json`. Note that path.

- [ ] **Step 4: Run the best local model, alone on the iGPU**

Evict whatever Ollama has resident first (one model at a time on this box — a saturated iGPU makes every decision time out), then run:
```bash
for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done
uv run tetris-bench --models pi/gpt-oss:20b \
  --harnesses legal features routed --efforts medium --seeds 1 --max-pieces 30 \
  --decision-deadline 15 --no-control 2>&1 | tee /tmp/claude-1000/-home-bdougie-code-pcc-labs-tetris/60f363ab-b829-4621-89fd-38a2b055018b/scratchpad/bench-local.log | tail -12
```
Expected: three rows and a second `results:` path. Note it.

- [ ] **Step 5: Pull the per-arm numbers**

Run, substituting both result paths:
```bash
uv run python - <<'EOF'
import json, glob
for path in sorted(glob.glob("data/benchmarks/benchmark-*.json"))[-2:]:
    d = json.load(open(path))
    print("==", path)
    for r in d["runs"]:
        s = r["policy_stats"]; f = r["fitness"]
        print(f'{r["arm"]:42s} score={f.get("score",0)!s:>5} lines={f.get("lines",0)} pieces={f.get("pieces_placed",0)} '
              f'decisions={s.get("decisions",0)} tok/dec={s.get("tokens_per_decision",0)} '
              f'in={s.get("input_tokens",0)} lat_ms={s.get("latency_ms_mean",0)} timeouts={s.get("timeouts",0)} '
              f'illegal={s.get("illegal_count",0)} cost=${s.get("cost_usd",0)}')
    for row in d["summary"]:
        print(f'   summary {row["arm"]:38s} race={row["race_score"]} score={row["score"]} pct15={row["pct_le_15s"]}')
EOF
```
Expected: one line per arm with race, score, decisions, tokens per decision, input tokens, mean latency, timeouts, illegal count and cost.

- [ ] **Step 6: Write the benchmark file**

Create `benchmarks/<UTC run date>-situation-router.md` in the matrix shape (see `benchmarks/README.md` and `benchmarks/2026-08-22-p15-deadline-screen.md`). Fill every `<…>` from Step 5; no placeholders may remain:

```markdown
# <date> — the situation router: `routed` vs `legal` vs `features`, per model

First benchmark of the `routed` harness (docs/superpowers/specs/2026-09-02-situation-router-design.md).
Same screen as 08-22 so rows compare: seed 1, 30-piece cap, `--decision-deadline 15` (+p15),
`medium` effort where supported, one local model resident at a time. Baselines from 08-22 at
this seed and cap: heuristic 490, random 140, no-input 55 (race). Corpus validation:
`benchmarks/2026-09-02-situation-histogram.md`.

Hypothesis (spec section 08, refined in the plan): a named technique makes the model
deliberate less — output tokens per decision and latency fall — without `race`/`score`
regressing against `legal` and `features`.

## Results (seed 1, 30-piece cap, +p15)

| arm | race | score | lines | pieces | timeouts | %≤15s | decisions | illegal | tok/decision | in tok | s/decision | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
<one row per arm, models grouped, harness order legal / features / routed, best race per model in bold>

## Per-model read

<For each model, two or three sentences: did routed cut tok/decision and latency versus features
and legal; did race/score hold; anything notable (illegal count, timeouts, downshifts).>

## Which model plays best under routing

<Rank the models by race on the routed arm; say whether the ranking differs from their
features ranking, and whether any local arm crossed the ≲300 tok/decision viability bar
from the 08-22 screen.>

## Verdict on the hypothesis

<Supported / partially / not — stated against the numbers. If routed lost on race, say where
(which situations are implicated is not measurable yet — decision events do not carry the
situation; that is the first follow-up).>

Runs: `<results path 1>`, `<results path 2>`. Skipped: <arms not run and why, or "none">.
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/*-situation-router.md
git commit -m "$(cat <<'EOF'
bench: the routed harness across models — first rows

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
)"
```

- [ ] **Step 8: Report** — the ranked routed arms, the verdict, and the two result paths, in the task's final message.

---

## Follow-ups deliberately left out of this plan

- Record `situation` on `placement_decision` events (both `agent.py` and `live_agent.py`) so model runs can be broken down per situation.
- The `MOUND` side bias (left vs centre) as an A/B once routing has a measured effect.
- Letting `evolve.py` implicate the three threshold fields in a healer rule.
- The semantic router (spec section 09) — only if the histogram shows the classes well spread *and* the fixed order is demonstrably mis-sorting.
