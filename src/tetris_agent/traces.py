"""Mine human runs into verified (board, decision) exemplars.

A recorded run stores decisions but not boards, so the board each decision was
taken on is reconstructed by replaying the placements through the pure board
simulation. The recorded per-piece features act as a checksum: after every
replayed lock the recomputed features must match what the emulator observed.
A turn that fails the check — a tuck the straight-drop model can't express, a
misread, an early quit — is dropped along with everything after it, loudly,
so a bad reconstruction can never become an exemplar.

Reviewer labels (labels.py) sit on top of the replay: an excluded turn never
becomes an exemplar, a promoted one always does and goes first — even from a
run whose policy would otherwise be ignored, since a human vouched for it.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tetris_agent.board import COLS, ROWS, drop, features, place
from tetris_agent.labels import load_labels
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
    promoted: bool = False  # a reviewer chose it; selection takes these first


@dataclass(frozen=True)
class Replayed:
    """One verified turn of a run: the decision and the board it was taken on."""

    turn: int
    board: np.ndarray
    piece: str
    next_piece: str
    rotation: int
    col: int
    lines_delta: int
    holes: int


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


def replay(events: list[dict], run_dir: Path | None = None) -> Iterator[Replayed]:
    """Verified turns, in order, stopping at the first one that fails the checksum."""
    board = np.zeros((ROWS, COLS), dtype=bool)
    spawn: dict | None = None
    decision: dict | None = None

    for e in events:
        kind, data = e.get("event_type"), e.get("data", {})
        if kind == "piece_spawn":
            spawn, decision = data, None
        elif kind == "placement_decision":
            if data.get("late"):
                continue  # the piece it describes was already placed by gravity
            decision = data
        elif kind == "piece_locked":
            if spawn is None or decision is None:
                logger.warning("%s: locked event without spawn/decision; stopping", run_dir)
                return
            shape = SHAPES.get((spawn["piece"], decision["rotation"]))
            row = None if shape is None else drop(board, shape, decision["col"])
            if row is None:
                logger.warning(
                    "%s turn %s: placement does not fit reconstruction; dropping rest", run_dir, e.get("turn")
                )
                return
            new_board, lines = place(board, shape, decision["col"], row)
            f = features(new_board)
            recorded = {k: data.get(k) for k in _CHECKED}
            if lines != data.get("lines_delta") or recorded != {k: getattr(f, k) for k in _CHECKED}:
                logger.warning("%s turn %s: reconstruction checksum mismatch; dropping rest", run_dir, e.get("turn"))
                return
            yield Replayed(
                turn=e.get("turn", 0),
                board=board,
                piece=spawn["piece"],
                next_piece=spawn["next_piece"],
                rotation=decision["rotation"],
                col=decision["col"],
                lines_delta=lines,
                holes=f.holes,
            )
            board = new_board
            spawn, decision = None, None


def mine_run(run_dir: Path, policies: tuple[str, ...] = ("human",)) -> list[Exemplar]:
    """Exemplars from one run, or [] if it isn't a matching / readable run.

    A run outside `policies` still contributes the turns a reviewer promoted,
    and nothing else; a run inside contributes every verified turn the
    reviewer did not exclude.
    """
    events = _events(run_dir)
    labels = load_labels(run_dir)
    matching = _policy(events) in policies
    if not matching and not any(label["verdict"] == "promote" for label in labels.values()):
        return []

    exemplars: list[Exemplar] = []
    for r in replay(events, run_dir):
        verdict = labels.get(r.turn, {}).get("verdict")
        if verdict == "exclude" or (not matching and verdict != "promote"):
            continue
        exemplars.append(
            Exemplar(
                board=r.board,
                piece=r.piece,
                next_piece=r.next_piece,
                rotation=r.rotation,
                col=r.col,
                lines_delta=r.lines_delta,
                holes=r.holes,
                promoted=verdict == "promote",
            )
        )
    return exemplars


def decisions(run_dir: Path) -> list[dict]:
    """Every verified decision of a run as the viewer's label deck shows it.

    The grade is the one the agent recorded (`placement_graded`) when there is
    one; a human run never graded itself, so those are graded here against
    the same oracle. Boards are encoded the way graded events carry them.

    Only the verified prefix is returned — the same turns the miner can use —
    so a run that tucked a piece on turn 6 shows five decisions, and the deck
    says so (see `placed_count`) rather than quietly offering the rest.
    """
    from tetris_agent.policy import Placement
    from tetris_agent.quality import grade

    events = _events(run_dir)
    labels = load_labels(run_dir)
    recorded = {e.get("turn"): e.get("data", {}) for e in events if e.get("event_type") == "placement_graded"}
    out = []
    for r in replay(events, run_dir):
        g = recorded.get(r.turn)
        if g is None:
            scored = grade(r.board, r.piece, r.next_piece, Placement(rotation=r.rotation, col=r.col, score=0.0))
            g = scored.to_dict() if scored is not None else None
        if g is not None:
            g = {k: v for k, v in g.items() if k not in ("board", "genome")}
        out.append(
            {
                "turn": r.turn,
                "board": ["".join("#" if cell else "." for cell in row) for row in r.board],
                "piece": r.piece,
                "next_piece": r.next_piece,
                "rotation": r.rotation,
                "col": r.col,
                "lines_delta": r.lines_delta,
                "holes": r.holes,
                "grade": g,
                # Cells the piece came to rest on, and where the oracle would
                # have put it, so the viewer can draw both without a shape table.
                "placed_cells": _cells(r.board, r.piece, r.rotation, r.col),
                "best_cells": _cells(r.board, r.piece, *g["best"]) if g else [],
                "label": labels.get(r.turn),
            }
        )
    return out


def placed_count(run_dir: Path) -> int:
    """Pieces the run actually locked, verified or not."""
    return sum(1 for e in _events(run_dir) if e.get("event_type") == "piece_locked")


def _cells(board: np.ndarray, piece: str, rotation: int, col: int) -> list[list[int]]:
    """[[row, col], ...] of `piece` dropped at (rotation, col), or [] if it doesn't fit."""
    shape = SHAPES.get((piece, rotation))
    row = None if shape is None else drop(board, shape, col)
    if row is None:
        return []
    return sorted([row + r, col + c] for r, c in shape)


def mine(runs_dir: Path, policies: tuple[str, ...] = ("human",)) -> list[Exemplar]:
    """All verified exemplars under runs_dir, oldest run first."""
    out: list[Exemplar] = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if run_dir.is_dir():
            out.extend(mine_run(run_dir, policies))
    return out


def select_exemplars(pool: list[Exemplar], k: int = 8) -> list[Exemplar]:
    """Pick k exemplars spread across piece types.

    Promoted exemplars go first, in pool order, because a reviewer picked them
    by hand. The rest fill the remaining slots round-robin over piece types so
    the block teaches the model about many shapes rather than k variations on
    one; within a type, line-clearing placements first — those are the
    decisions worth imitating.
    """
    picked: list[Exemplar] = [ex for ex in pool if ex.promoted][:k]
    by_piece: dict[str, list[Exemplar]] = {}
    for ex in pool:
        if not ex.promoted:
            by_piece.setdefault(ex.piece, []).append(ex)
    for group in by_piece.values():
        group.sort(key=lambda e: (-e.lines_delta, e.holes))

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
            "(or `uv run tetris-manual`) first, or promote decisions in the viewer's LABEL deck"
        )
    from tetris_agent.prompts import build_exemplar_block

    chosen = select_exemplars(pool, k)
    promoted = sum(e.promoted for e in chosen)
    logger.info("exemplars: %d of %d selected, %d promoted by a reviewer", len(chosen), len(pool), promoted)
    return build_exemplar_block(chosen)
