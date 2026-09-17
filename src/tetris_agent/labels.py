"""Reviewer labels on recorded decisions, one file per run.

`runs/<id>/labels.json` maps a turn to a verdict the exemplar miner acts on:

    {"12": {"verdict": "promote", "note": "", "source": "human",
            "jev": {...answers as returned...}, "labeled_at": "..."}}

`promote` forces the decision into the exemplar block; `exclude` keeps it out
even from a human run that would otherwise contribute every turn. A turn with
no entry is treated as before — the miner's own rules apply. Jev's answers
ride along for the record but decide nothing: the verdict is the human's
(source "human"), or Jev's proposal the human accepted (source "jev").
"""

import json
from datetime import datetime, timezone
from pathlib import Path

VERDICTS = ("promote", "exclude")
FILENAME = "labels.json"


def load_labels(run_dir: Path) -> dict[int, dict]:
    path = Path(run_dir) / FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for turn, label in raw.items():
        if isinstance(label, dict) and label.get("verdict") in VERDICTS:
            out[int(turn)] = label
    return out


def _write(run_dir: Path, labels: dict[int, dict]) -> None:
    path = Path(run_dir) / FILENAME
    path.write_text(json.dumps({str(t): labels[t] for t in sorted(labels)}, indent=1) + "\n")


def set_label(
    run_dir: Path, turn: int, verdict: str, note: str = "", source: str = "human", jev: dict | None = None
) -> dict:
    """Write one verdict, replacing any earlier one on the same turn."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, not {verdict!r}")
    labels = load_labels(run_dir)
    label = {
        "verdict": verdict,
        "note": note,
        "source": source,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }
    if jev is not None:
        label["jev"] = jev
    labels[int(turn)] = label
    _write(run_dir, labels)
    return label


def clear_label(run_dir: Path, turn: int) -> bool:
    """Remove the verdict on `turn`. False if there was none."""
    labels = load_labels(run_dir)
    if int(turn) not in labels:
        return False
    del labels[int(turn)]
    _write(run_dir, labels)
    return True
