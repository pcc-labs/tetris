"""Jev: typed judgments about a placement decision.

Jev is TypeSafe's "System One" model (https://docs.typesafe.ai). One POST
carries a piece of state and a map of typed questions; the answer to each
comes back constrained to the options supplied, with a probability
distribution. No text generation, so nothing to parse and nothing to recover
from prose — and it is fast enough to re-ask as the reviewer steps between
pieces, the way the TypeSafe Typewriter demo re-asks on every keystroke.

Here the state is one decision as the viewer shows it: the pre-decision board,
the piece, the placement taken, and the oracle's grade of it. The questions
are the reviewer's own — would this teach a model anything, and should it be
promoted into the exemplar block or kept out of it. Jev proposes; the label
that the miner reads is still written by the human (see labels.py).

The question specs are the single source of truth: the viewer renders whatever
`/api/jev/questions` reports, so adding one here is the whole change.
"""

import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
API_KEY_VAR = "TYPESAFE_API_KEY"
API_KEY_CONSOLE_URL = "https://console.typesafe.ai/settings/keys"
DEFAULT_MODEL = "jev-latest"

# Display order. The verdict leads because it is what the reviewer acts on;
# the rest explain it.
GROUPS = [
    {"id": "verdict", "title": "Verdict", "blurb": "What Jev would do with this decision."},
    {"id": "board", "title": "On the board", "blurb": "What the placement did."},
]

SPECS = [
    {
        "id": "verdict",
        "label": "Verdict",
        "group": "verdict",
        "question": {
            "type": "choice",
            "instructions": (
                "This is one Tetris placement, graded against a two-ply lookahead oracle. "
                "Should it be shown to a language model as an example of good play, kept out "
                "of the examples, or neither?"
            ),
            "criteria": {
                "promote": "A placement worth imitating: show it as an exemplar",
                "neutral": "Ordinary; neither teaches nor misleads",
                "exclude": "Misleading or bad; keep it out of the exemplars",
            },
        },
    },
    {
        "id": "exemplar_worthy",
        "label": "Worth teaching",
        "group": "verdict",
        "question": {
            "type": "noul",
            "instructions": "Would a model that imitated this placement play better Tetris for it?",
            "criteria": {"true": "Imitating this improves play", "false": "Nothing here to learn"},
        },
    },
    {
        "id": "quality",
        "label": "Quality",
        "group": "verdict",
        "question": {
            "type": "score",
            "instructions": "How good is this placement, judged on the board it was taken on?",
            "criteria": ["Blunder", "Weak", "Reasonable", "Strong", "Best available"],
        },
    },
    {
        "id": "creates_hole",
        "label": "Creates a hole",
        "group": "board",
        "question": {
            "type": "noul",
            "instructions": "Does this placement leave an empty cell covered by a settled cell?",
        },
    },
    {
        "id": "sets_up_clear",
        "label": "Sets up a clear",
        "group": "board",
        "question": {
            "type": "noul",
            "instructions": "Does this placement clear a line or leave the board one piece from clearing one?",
        },
    },
    {
        "id": "risk",
        "label": "Risk",
        "group": "board",
        "question": {
            "type": "score",
            "instructions": "How close to topping out does the board look after this placement?",
            "criteria": ["Flat and low", "Comfortable", "Building up", "Tall and uneven", "About to top out"],
        },
    },
]

QUESTIONS = {spec["id"]: spec["question"] for spec in SPECS}

# What the browser is told: everything but the question text, which it never
# needs (the answers arrive keyed by id, and the labels live here).
PUBLIC_QUESTIONS = [
    {
        "id": spec["id"],
        "label": spec["label"],
        "group": spec["group"],
        "type": spec["question"]["type"],
        "criteria": spec["question"].get("criteria"),
    }
    for spec in SPECS
]


def model() -> str:
    return os.environ.get("TYPESAFE_MODEL", "").strip() or DEFAULT_MODEL


def api_key() -> str:
    """Trimmed, so a var set to the empty string counts as absent."""
    return os.environ.get(API_KEY_VAR, "").strip()


def is_configured() -> bool:
    return bool(api_key())


def setup_state() -> dict:
    """Whether a key exists and where to get one. Never the key itself."""
    return {"configured": is_configured(), "keyVar": API_KEY_VAR, "consoleUrl": API_KEY_CONSOLE_URL}


def decision_state(decision: dict) -> dict:
    """The state Jev is asked about: a decision record as `labels.decisions` builds it.

    Structured rather than prose so nothing is lost in narration. The oracle's
    grade goes in too — the reviewer sees it on the same screen, and a verdict
    that ignored it would be the odd one out.
    """
    grade = decision.get("grade") or {}
    state = {
        "game": "Game Boy Tetris, 10 columns by 18 rows, row 0 at the top",
        "board_before": decision["board"],
        "board_legend": "'.' empty, '#' settled",
        "piece": decision["piece"],
        "next_piece": decision.get("next_piece"),
        "placement": {"rotation": decision["rotation"], "col": decision["col"]},
        "outcome": {"lines_cleared": decision.get("lines_delta"), "holes_after": decision.get("holes")},
    }
    if grade:
        state["oracle"] = {
            "best_placement": {"rotation": grade["best"][0], "col": grade["best"][1]},
            "rank": f"{grade['rank']} of {grade['legal_count']} legal placements",
            "regret": grade["regret"],
            "regret_norm": grade["regret_norm"],
            "note": "regret is best_value - chosen_value; regret_norm scales it 0..1 against the worst placement",
        }
    return state


def ask(state, questions: dict | None = None, timeout_s: float = 10.0) -> dict:
    """One call: every question, evaluated in parallel against `state`.

    Returns the API body as-is: {"model", "answers": {id: answer}, "usage"}.
    Answer shapes — noul: {"type","noul"}; choice: {"type","choice",
    "probabilities","confidence"}; score: {"type","score","legend",
    "probabilities","confidence"}.
    """
    key = api_key()
    if not key:
        raise RuntimeError(f"{API_KEY_VAR} is not set")
    body = json.dumps({"state": state, "model": model(), "questions": questions or QUESTIONS}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:300]
        raise RuntimeError(f"TypeSafe {err.code}: {detail}") from err
