"""Board rendering and harness prompt construction.

The "harness" is the benchmark's independent variable alongside model and
effort: how much of the reasoning is done *for* the model before it decides.
Everything here is pure so the prompt shapes can be unit-tested without an
emulator or an API key.
"""

from dataclasses import dataclass

import numpy as np

from tetris_agent.board import COLS, Features, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width

HARNESSES = ("board", "legal", "features", "chat")

PLACEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "rotation": {"type": "integer", "enum": [0, 1, 2, 3]},
        "col": {"type": "integer", "enum": list(range(COLS))},
        "reason": {"type": "string", "description": "One short sentence explaining the choice."},
    },
    "required": ["rotation", "col", "reason"],
    "additionalProperties": False,
}

# Static across every decision in a run, so it caches. Keep the volatile board
# out of here — it goes in the user turn, after the cache breakpoint.
SYSTEM_PROMPT = """\
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

Respond with your chosen placement and one short sentence of reasoning."""


@dataclass(frozen=True)
class LegalPlacement:
    rotation: int
    col: int
    landing_row: int
    lines: int
    features: Features


def render_board(board: np.ndarray) -> str:
    return "\n".join("".join("#" if cell else "." for cell in row) for row in board)


def legal_placements(board: np.ndarray, piece: str) -> list[LegalPlacement]:
    """Every (rotation, col) the piece can actually be dropped into."""
    out = []
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is None:
                continue
            new_board, lines = place(board, shape, col, row)
            out.append(
                LegalPlacement(rotation=rot, col=col, landing_row=row, lines=lines, features=features(new_board))
            )
    return out


def _piece_art(piece: str, rotation: int) -> str:
    cells = SHAPES[(piece, rotation)]
    rows = max(r for r, _ in cells) + 1
    width = shape_width(cells)
    return "\n".join("".join("#" if (r, c) in cells else "." for c in range(width)) for r in range(rows))


def _shapes_block(piece: str) -> str:
    parts = []
    for rot in distinct_rotations(piece):
        art = _piece_art(piece, rot).replace("\n", "\n    ")
        parts.append(f"  rotation {rot} (width {shape_width(SHAPES[(piece, rot)])}):\n    {art}")
    return "\n".join(parts)


def build_user_prompt(
    harness: str,
    board: np.ndarray,
    piece: str,
    next_piece: str,
    placements: list[LegalPlacement],
    turn: int,
    deadline_s: float | None = None,
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


def build_exemplar_block(exemplars) -> str:
    """Render mined human decisions for the (cached) system prompt.

    Static across a run by construction — the block must live in the system
    prompt, where it is cached, not in the per-piece user turn, where it
    would be paid for on every decision.
    """
    if not exemplars:
        return ""
    parts = [
        "# How a strong human placed pieces",
        "",
        "Real decisions from a human player facing the same game. Imitate the",
        "judgement, not the exact moves — your boards will differ.",
    ]
    for i, ex in enumerate(exemplars, start=1):
        outcome = f"cleared {ex.lines_delta} line(s)" if ex.lines_delta else f"left {ex.holes} hole(s)"
        parts += [
            "",
            f"## Example {i} — Piece: {ex.piece}, Next: {ex.next_piece}",
            render_board(ex.board),
            f"Human chose rotation={ex.rotation} col={ex.col} — {outcome}.",
        ]
    return "\n".join(parts)
