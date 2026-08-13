"""tetris.game.v1 event envelopes — the telemetry contract a Kafka bridge can tail."""

from dataclasses import asdict
from datetime import datetime, timezone

from tetris_agent.board import Features
from tetris_agent.policy import Placement

SCHEMA = "tetris.game.v1"


def _envelope(event_type: str, turn: int, data: dict) -> dict:
    return {
        "schema": SCHEMA,
        "event_type": event_type,
        "turn": turn,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


def build_session_event(phase: str, data: dict, turn: int = 0) -> dict:
    return _envelope("session", turn, {"phase": phase, **data})


def build_spawn_event(turn: int, piece: str, next_piece: str) -> dict:
    return _envelope("piece_spawn", turn, {"piece": piece, "next_piece": next_piece})


def build_decision_event(
    turn: int,
    placement: Placement,
    reason: str = "",
    latency_ms: float | None = None,
    late: bool = False,
    tokens: int | None = None,
) -> dict:
    data = asdict(placement)
    if reason:
        data["reason"] = reason
    if latency_ms is not None:
        data["latency_ms"] = round(latency_ms, 1)
    if late:
        data["late"] = True
    if tokens is not None:
        data["tokens"] = tokens
    return _envelope("placement_decision", turn, data)


def build_locked_event(turn: int, lines_delta: int, features: Features, misexec: int, score: int = 0) -> dict:
    return _envelope(
        "piece_locked", turn, {"lines_delta": lines_delta, "misexec": misexec, "score": score, **asdict(features)}
    )


def build_stuck_event(turn: int, streak: int, detail: str) -> dict:
    return _envelope("stuck", turn, {"streak": streak, "detail": detail})


def build_game_over_event(turn: int, fitness: dict) -> dict:
    return _envelope("game_over", turn, {"fitness": fitness})


class EventCollector:
    """Agent-facing façade: tracks the turn (piece) counter and publishes."""

    def __init__(self, publisher):
        self.publisher = publisher
        self.turn = 0

    def session(self, phase: str, data: dict) -> None:
        self.publisher.publish(build_session_event(phase, data, self.turn))

    def spawn(self, piece: str, next_piece: str) -> None:
        self.turn += 1
        self.publisher.publish(build_spawn_event(self.turn, piece, next_piece))

    def decision(
        self,
        placement: Placement,
        reason: str = "",
        latency_ms: float | None = None,
        late: bool = False,
        turn: int | None = None,
        tokens: int | None = None,
    ) -> None:
        self.publisher.publish(
            build_decision_event(self.turn if turn is None else turn, placement, reason, latency_ms, late, tokens)
        )

    def locked(self, lines_delta: int, features: Features, misexec: int, score: int = 0) -> None:
        self.publisher.publish(build_locked_event(self.turn, lines_delta, features, misexec, score))

    def stuck(self, streak: int, detail: str) -> None:
        self.publisher.publish(build_stuck_event(self.turn, streak, detail))

    def game_over(self, fitness: dict) -> None:
        self.publisher.publish(build_game_over_event(self.turn, fitness))
