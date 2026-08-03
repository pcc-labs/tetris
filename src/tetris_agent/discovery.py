"""Capability healing: when parameter tuning is exhausted, hand the problem to
a Claude Code session in an isolated git worktree. Its proposal only ships if
it passes tests, lint, and a measured fitness race — a human merges the PR.

This is the only component allowed to shell out beyond git/gh: it drives the
`claude` CLI itself.
"""

import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tetris_agent.evolve import decide
from tetris_agent.fitness import race_score

logger = logging.getLogger(__name__)

COOLDOWN_HOURS = 24
QUEUE_PATH = Path("data/discovery_queue.json")
STATE_PATH = Path("data/discovery_state.json")
HEALER_STATE_PATH = Path("data/healer_state.json")
GATE_MAX_PIECES = 150
GATE_SEEDS = (0x00, 0x40)

RULE_CODE_MAP = {
    "early-topout": ["src/tetris_agent/policy.py", "src/tetris_agent/board.py"],
    "hole-thrash": ["src/tetris_agent/policy.py", "src/tetris_agent/board.py"],
    "misexecution": ["src/tetris_agent/controller.py", "src/tetris_agent/emulator.py"],
}

PROMPT_TEMPLATE = """\
You are working on tetris-agent, a self-healing Game Boy Tetris agent.

The healer rule `{rule}` keeps firing and parameter tuning is exhausted:
the latest race scored control={control} vs best variant={winner_score}.

Fitness of the failing run:
{fitness}

Recent race history:
{races}

The rule implicates these files — start there, but follow the evidence:
{files}

Improve the agent's capability so this failure mode stops occurring. Rules:
- Add or update tests for what you change (TDD).
- Run `uv run pytest` and `uv run ruff check .` before finishing.
- Do NOT touch data/genome.json, data/healer_state.json, or any file under data/.
- Do NOT change fitness definitions or healer rules to make the problem disappear.
- Keep the change minimal and focused on the failure mode.
"""


def load_queue(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("discovery queue %s is corrupt; ignoring", path)
        return []


def _load_json(path: str | Path, default: dict) -> dict:
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("%s is corrupt; resetting", path)
    return default


def _save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def pick_entry(queue: list[dict], state: dict) -> dict | None:
    """Oldest un-attempted entry, if the global 24h attempt cooldown allows."""
    last = state.get("last_attempt_at")
    if last and datetime.now(timezone.utc) - datetime.fromisoformat(last) < timedelta(hours=COOLDOWN_HOURS):
        return None
    pending = [e for e in queue if not e.get("attempted")]
    return min(pending, key=lambda e: e["queued_at"]) if pending else None


def build_bundle(entry: dict, healer_state: dict) -> dict:
    races = [r for r in healer_state.get("races", []) if r.get("rule") == entry["rule"]][-5:]
    return {"entry": entry, "races": races, "files": RULE_CODE_MAP.get(entry["rule"], [])}


def build_prompt(bundle: dict) -> str:
    entry = bundle["entry"]
    return PROMPT_TEMPLATE.format(
        rule=entry["rule"],
        control=entry.get("race", {}).get("control"),
        winner_score=entry.get("race", {}).get("winner_score"),
        fitness=json.dumps(entry["fitness"], indent=2),
        races=json.dumps(bundle["races"], indent=2),
        files="\n".join(f"- {f}" for f in bundle["files"]),
    )


def worktree_add(repo_root: str | Path, branch: str) -> Path:
    worktree = Path(repo_root) / ".worktrees" / branch.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", str(worktree), "-b", branch],
        cwd=repo_root, check=True, capture_output=True, text=True,
    )
    return worktree


def propose(worktree: Path, prompt: str) -> bool:
    result = subprocess.run(
        ["claude", "-p", prompt, "--permission-mode", "acceptEdits", "--max-turns", "40"],
        cwd=worktree, capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        logger.warning("claude proposal failed: %s", result.stderr[-500:])
    return result.returncode == 0


def _mean_gate_score(directory: str | Path, rom_path: str | Path) -> float:
    scores = []
    for seed in GATE_SEEDS:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            "uv", "--directory", str(directory), "run", "tetris-agent",
            "--rom", str(Path(rom_path).resolve()),
            "--max-pieces", str(GATE_MAX_PIECES), "--timer-div", str(seed),
            "--no-self-heal", "--no-record", "--no-telemetry", "--output-json", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        scores.append(race_score(json.loads(Path(out_path).read_text())))
    return sum(scores) / len(scores)


def run_gates(worktree: Path, rom_path: str | Path, entry: dict) -> dict:
    tests = subprocess.run(
        ["uv", "run", "pytest", "-m", "not rom", "-q"], cwd=worktree, capture_output=True, text=True
    )
    if tests.returncode != 0:
        return {"passed": False, "reason": "pytest", "detail": tests.stdout[-1000:]}
    lint = subprocess.run(["uv", "run", "ruff", "check", "."], cwd=worktree, capture_output=True, text=True)
    if lint.returncode != 0:
        return {"passed": False, "reason": "ruff", "detail": lint.stdout[-1000:]}
    candidate = _mean_gate_score(worktree, rom_path)
    baseline = _mean_gate_score(Path.cwd(), rom_path)
    if not decide(baseline, candidate):
        return {"passed": False, "reason": "fitness", "baseline": baseline, "candidate": candidate}
    return {"passed": True, "baseline": baseline, "candidate": candidate}


def push_and_pr(worktree: Path, branch: str, entry: dict) -> str:
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", f"discovery: address {entry['rule']}"],
        cwd=worktree, check=True, capture_output=True, text=True,
    )
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=worktree, check=True, capture_output=True, text=True)
    result = subprocess.run(
        ["gh", "pr", "create", "--fill", "--head", branch],
        cwd=worktree, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def cleanup(worktree: Path, branch: str) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], capture_output=True, text=True)
    subprocess.run(["git", "branch", "-D", branch], capture_output=True, text=True)


def run(
    queue_path: str | Path = QUEUE_PATH,
    state_path: str | Path = STATE_PATH,
    healer_state_path: str | Path = HEALER_STATE_PATH,
    repo_root: str | Path = ".",
    rom_path: str | Path = "rom/tetris.gb",
    dry_run: bool = False,
) -> dict:
    """One discovery attempt; never raises."""
    try:
        queue = load_queue(queue_path)
        state = _load_json(state_path, {})
        entry = pick_entry(queue, state)
        if entry is None:
            return {"status": "idle"}
        bundle = build_bundle(entry, _load_json(healer_state_path, {}))
        prompt = build_prompt(bundle)
        if dry_run:
            print(prompt)
            return {"status": "dry-run", "rule": entry["rule"]}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch = f"discovery/{entry['rule']}-{stamp}"
        worktree = worktree_add(repo_root, branch)

        entry["attempted"] = True
        state["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(queue_path, queue)
        _save_json(state_path, state)

        if propose(worktree, prompt):
            gates = run_gates(worktree, rom_path, entry)
        else:
            gates = {"passed": False, "reason": "propose"}
        detail = {k: v for k, v in gates.items() if k != "passed"}
        if gates["passed"]:
            url = push_and_pr(worktree, branch, entry)
            logger.info("discovery PR opened: %s", url)
            return {"status": "pr", "rule": entry["rule"], "url": url, **detail}
        cleanup(worktree, branch)
        return {"status": "failed", "rule": entry["rule"], **detail}
    except Exception:
        logger.exception("discovery run failed")
        return {"status": "error"}


def main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="tetris-discovery")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="attempt the oldest queued escalation")
    run_parser.add_argument("--rom", default="rom/tetris.gb")
    run_parser.add_argument("--dry-run", action="store_true", help="print the prompt without spawning anything")
    args = parser.parse_args(argv)

    report = run(rom_path=args.rom, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 0
