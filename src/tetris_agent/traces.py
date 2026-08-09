"""Mine human runs into verified (board, decision) exemplars.

A recorded run stores decisions but not boards, so the board each decision was
taken on is reconstructed by replaying the placements through the pure board
simulation. The recorded per-piece features act as a checksum: after every
replayed lock the recomputed features must match what the emulator observed.
A turn that fails the check — a tuck the straight-drop model can't express, a
misread, an early quit — is dropped along with everything after it, loudly,
so a bad reconstruction can never become an exemplar.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tetris_agent.board import COLS, ROWS, drop, features, place
from tetris_agent.pieces import SHAPES

logger = logging.getLogger(__name__)

_CHECKED = ("holes", "agg_height", "bumpiness", "max_height")


@dataclass(frozen=True)
class Exemplar:
    board: np.ndarray  # settled board *before* the placement
    piece: str
    next_piece: str
    rotation: int
    col: int
    lines_delta: int
    holes: int  # holes after the lock, per the recorded checksum


def _events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping unparseable line in %s", path)
    return out


def _policy(events: list[dict]) -> str | None:
    for e in events:
        if e.get("event_type") == "session" and e.get("data", {}).get("phase") == "start":
            return e["data"].get("policy")
    return None


def mine_run(run_dir: Path, policies: tuple[str, ...] = ("human",)) -> list[Exemplar]:
    """Exemplars from one run, or [] if it isn't a matching / readable run."""
    events = _events(run_dir)
    if _policy(events) not in policies:
        return []

    board = np.zeros((ROWS, COLS), dtype=bool)
    exemplars: list[Exemplar] = []
    spawn: dict | None = None
    decision: dict | None = None

    for e in events:
        kind, data = e.get("event_type"), e.get("data", {})
        if kind == "piece_spawn":
            spawn, decision = data, None
        elif kind == "placement_decision":
            decision = data
        elif kind == "piece_locked":
            if spawn is None or decision is None:
                logger.warning("%s: locked event without spawn/decision; stopping", run_dir)
                break
            shape = SHAPES.get((spawn["piece"], decision["rotation"]))
            row = None if shape is None else drop(board, shape, decision["col"])
            if row is None:
                logger.warning(
                    "%s turn %s: placement does not fit reconstruction; dropping rest", run_dir, e.get("turn")
                )
                break
            new_board, lines = place(board, shape, decision["col"], row)
            f = features(new_board)
            recorded = {k: data.get(k) for k in _CHECKED}
            if lines != data.get("lines_delta") or recorded != {k: getattr(f, k) for k in _CHECKED}:
                logger.warning("%s turn %s: reconstruction checksum mismatch; dropping rest", run_dir, e.get("turn"))
                break
            exemplars.append(
                Exemplar(
                    board=board,
                    piece=spawn["piece"],
                    next_piece=spawn["next_piece"],
                    rotation=decision["rotation"],
                    col=decision["col"],
                    lines_delta=lines,
                    holes=f.holes,
                )
            )
            board = new_board
            spawn, decision = None, None
    return exemplars


def mine(runs_dir: Path, policies: tuple[str, ...] = ("human",)) -> list[Exemplar]:
    """All verified exemplars under runs_dir, oldest run first."""
    out: list[Exemplar] = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if run_dir.is_dir():
            out.extend(mine_run(run_dir, policies))
    return out


def select_exemplars(pool: list[Exemplar], k: int = 8) -> list[Exemplar]:
    """Pick k exemplars spread across piece types.

    Round-robin over piece types so the block teaches the model about many
    shapes rather than k variations on one; within a type, line-clearing
    placements first — those are the decisions worth imitating.
    """
    by_piece: dict[str, list[Exemplar]] = {}
    for ex in pool:
        by_piece.setdefault(ex.piece, []).append(ex)
    for group in by_piece.values():
        group.sort(key=lambda e: (-e.lines_delta, e.holes))

    picked: list[Exemplar] = []
    while len(picked) < k:
        took = False
        for group in by_piece.values():
            if group:
                picked.append(group.pop(0))
                took = True
                if len(picked) == k:
                    break
        if not took:
            break
    return picked


def load_exemplar_block(runs_dir: Path = Path("runs"), k: int = 8) -> str:
    """Mine, select, and render — the one call the CLIs need.

    Raises rather than returning "" when no traces verify: an arm labeled +ex
    that silently ran without exemplars would corrupt the comparison.
    """
    pool = mine(runs_dir)
    if not pool:
        raise RuntimeError(
            f"no verified human traces under {runs_dir} — record one with `uv run tetris-play` "
            "(or `uv run tetris-manual`) first"
        )
    from tetris_agent.prompts import build_exemplar_block

    return build_exemplar_block(select_exemplars(pool, k))
