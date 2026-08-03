"""Persistent genome: the parameter channel between healer and agent.

A plain JSON file with {"current": params, "history": [...]} — pokemon-kafka
stored this as an HTML comment in notes.md parsed by regex; don't do that.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tetris_agent.policy import Genome

logger = logging.getLogger(__name__)

GENOME_PATH = Path("data/genome.json")


def load_params(path: str | Path = GENOME_PATH) -> dict:
    params = Genome().to_params()
    path = Path(path)
    if path.exists():
        try:
            stored = json.loads(path.read_text())
            params.update(stored.get("current", {}))
        except (json.JSONDecodeError, AttributeError):
            logger.warning("genome file %s is corrupt; using defaults", path)
    return params


def append_genome(params: dict, source: str, path: str | Path = GENOME_PATH) -> None:
    path = Path(path)
    store = {"current": {}, "history": []}
    if path.exists():
        try:
            store = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("genome file %s is corrupt; rewriting", path)
    store["current"] = params
    store.setdefault("history", []).append(
        {"params": params, "source": source, "ts": datetime.now(timezone.utc).isoformat()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.rename(path)
