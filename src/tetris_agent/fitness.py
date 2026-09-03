"""Per-run fitness — the healer's only input signal."""

from tetris_agent.board import Features


class FitnessTracker:
    def __init__(self):
        self._holes: list[int] = []
        self._misexec_count = 0
        self._streak = 0
        self._max_streak = 0
        self._max_stack = 0
        self._topped_out = False
        self._regrets: list[float] = []
        self._ranks: list[int] = []

    def on_lock(self, features: Features, misexec: int) -> None:
        self._holes.append(features.holes)
        self._misexec_count += misexec
        self._streak = self._streak + 1 if misexec > 0 else 0
        self._max_streak = max(self._max_streak, self._streak)
        self._max_stack = max(self._max_stack, features.max_height)

    def on_game_over(self) -> None:
        self._topped_out = True

    def on_grade(self, grade) -> None:
        """One decision scored against the oracle. Late and fallback placements
        never reach here, so the averages describe choices the model actually made."""
        self._regrets.append(grade.regret_norm)
        self._ranks.append(grade.rank)

    def compute(self, score: int, lines: int, level: int) -> dict:
        pieces = len(self._holes)
        graded = len(self._regrets)
        return {
            "score": score,
            "lines": lines,
            "level": level,
            "pieces_placed": pieces,
            "avg_holes": round(sum(self._holes) / pieces, 3) if pieces else 0.0,
            "max_stack_height": self._max_stack,
            "misexec_count": self._misexec_count,
            "max_misexec_streak": self._max_streak,
            "topped_out": self._topped_out,
            "graded_decisions": graded,
            "mean_regret": round(sum(self._regrets) / graded, 4) if graded else None,
            "top1_rate": round(sum(1 for r in self._ranks if r == 1) / graded, 3) if graded else None,
            "top3_rate": round(sum(1 for r in self._ranks if r <= 3) / graded, 3) if graded else None,
        }


def race_score(fitness: dict) -> float:
    """Scalar the healer races on: score dominated, survival as tiebreak."""
    return float(fitness.get("score", 0)) + 5.0 * float(fitness.get("pieces_placed", 0))
