"""Tetromino silhouettes per rotation value.

The game encodes the current piece at 0xC203 and the next piece at 0xC213:
bits 7-2 select the type, bits 1-0 the rotation state. Silhouettes here are
normalized (row, col) offsets; the ROM-marked test in test_state.py verifies
each (name, rotation) silhouette against sprites observed in the emulator.
"""

Cells = frozenset[tuple[int, int]]

# Type byte (rotation bits masked off) per the game's tetromino table.
PIECES: dict[str, int] = {"L": 0, "J": 4, "I": 8, "O": 12, "Z": 16, "S": 20, "T": 24}
_NAME_BY_ID = {v: k for k, v in PIECES.items()}

_SPAWN: dict[str, Cells] = {
    "L": frozenset({(0, 0), (0, 1), (0, 2), (1, 0)}),
    "J": frozenset({(0, 0), (0, 1), (0, 2), (1, 2)}),
    "I": frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    "O": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "Z": frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    "S": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "T": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
}


def _normalize(cells: set[tuple[int, int]]) -> Cells:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_ccw(cells: Cells) -> Cells:
    """One step of the rotation-value encoding. Verified against the ROM:
    incrementing the rot bits of 0xC203 corresponds to a counter-clockwise
    silhouette rotation (the A button *decrements* the value)."""
    max_c = max(c for _, c in cells)
    return _normalize({(max_c - c, r) for r, c in cells})


def _build_shapes() -> dict[tuple[str, int], Cells]:
    shapes: dict[tuple[str, int], Cells] = {}
    for name, spawn in _SPAWN.items():
        cells = spawn
        for rot in range(4):
            shapes[(name, rot)] = cells
            cells = _rotate_ccw(cells)
    return shapes


SHAPES: dict[tuple[str, int], Cells] = _build_shapes()


def piece_name(byte: int) -> str:
    """Translate a piece byte (0xC203 / 0xC213) to its type name."""
    return _NAME_BY_ID[byte & 0b11111100]


def distinct_rotations(name: str) -> list[int]:
    """Rotation values with unique silhouettes, in press order."""
    seen: dict[Cells, int] = {}
    for rot in range(4):
        seen.setdefault(SHAPES[(name, rot)], rot)
    return sorted(seen.values())


def shape_width(cells: Cells) -> int:
    return max(c for _, c in cells) + 1
