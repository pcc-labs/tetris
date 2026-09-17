#!/usr/bin/env python3
"""An Ollama host on a Daytona GPU sandbox — the pi/ arms when the Framework is out of reach.

The harness is host-agnostic already: preflight, `pi`, and the viewer all talk to
whatever `TETRIS_OLLAMA_URL` names. This boots a GPU sandbox that answers that URL
and tears it down again; nothing downstream changes.

    uv sync --group daytona
    uv run --group daytona scripts/daytona_host.py snapshot --models gemma4       # once per model list
    uv run --group daytona scripts/daytona_host.py up --models gemma4             # ~seconds
    eval "$(uv run --group daytona scripts/daytona_host.py env)"
    uv run tetris-bench --models pi/gemma4 --harnesses routed --seeds 1
    uv run --group daytona scripts/daytona_host.py down                           # or the TTL reaps it

`snapshot` builds docker/daytona/Dockerfile server-side with the weights pulled at
build time — the stereOS move: pay for the pull once, then every `up` is a boot,
not a download. The snapshot name is a function of the model list and the
Dockerfile, so `up --models ...` finds the matching one or tells you to build it.
`up --snapshot daytona-gpu` is the slow path on Daytona's stock GPU image (install
plus pull on every boot); `up` pulls whatever the box turns out to be missing
either way.

The host is public while it is up — plain `TETRIS_OLLAMA_URL` consumers (urllib,
pi) send no auth headers — and holds nothing but weights. An obscure hostname and
the TTL bound the exposure; a forgotten host reaps itself instead of billing
overnight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "docker" / "daytona" / "Dockerfile"
STATE_FILE = REPO / "data" / "daytona" / "host.json"
LEDGER_FILE = REPO / "data" / "daytona" / "hosts.jsonl"  # one line per host session, with its cost
# daytona.io/pricing, read 2026-09-17: pay-as-you-go compute plus the preemptible
# GPU column. The bench reports every arm's cost_usd; the sandbox that served a
# pi/ arm is a cost too, and this is what turns its hours into dollars.
USD_PER_VCPU_HOUR = 0.0504
USD_PER_GIB_HOUR = 0.0162
USD_PER_DISK_GIB_HOUR = 0.000108  # after the first 5 GiB
USD_PER_GPU_HOUR = {"H100": 2.27, "H200": 2.61, "RTX-PRO-6000": 1.74, "RTX-4090": 0.57, "RTX-5090": 0.74}
PI_MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

STOCK_GPU_SNAPSHOT = "daytona-gpu"  # Daytona's own image; pokemon-kafka probed it as an H100 80GB
OLLAMA_PORT = 11434
MODELS_DIR = "/models"
INSTALL_CMD = "command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh"
# OLLAMA_KEEP_ALIVE: a matrix idles between arms for longer than the 5-minute
# default, and reloading 20 GB per arm is a visible dent in the first decision.
# setsid + </dev/null: an exec'd command only returns once nothing holds its
# stdio, and a bare `nohup … &` still inherits stdin — the serve step then sits
# out its whole timeout and comes back as a 408 (live, 2026-09-17).
# The baked image owns /models; the stock daytona-gpu image runs as user
# `daytona` and cannot create it (live, 2026-09-17), so fall back to $HOME.
SERVE_CMD = (
    f"M={MODELS_DIR}; {{ mkdir -p $M && [ -w $M ]; }} 2>/dev/null || M=$HOME/.ollama/models; mkdir -p $M; "
    "OLLAMA_MODELS=$M OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=1h "
    "setsid nohup ollama serve > /tmp/ollama.log 2>&1 < /dev/null &"
)
# `break`, never `exit`: these snippets are joined into one exec'd script.
WAIT_SNIPPET = (
    f"ok=0; for i in $(seq 1 30); do curl -s http://127.0.0.1:{OLLAMA_PORT} >/dev/null"
    ' && { ok=1; break; }; sleep 1; done; [ "$ok" = 1 ]'
)
# ollama logs one `inference compute` line per device it will use; `library=cpu`
# there means the GPU reservation did not reach the runtime.
GPU_LINE_CMD = "grep -m1 'inference compute' /tmp/ollama.log || echo 'no inference compute line yet'"
# A promptless generate loads the weights and returns. Measured 2026-09-17: the
# first real call after boot took 58.6 s for two tokens, the second 0.6 s — every
# decision inside that window is a timeout, so the load is paid here, not in
# the first arm's clock.
WARM_CMD = (
    'curl -s -m 900 http://127.0.0.1:{port}/api/generate -d \'{{"model": "{tag}", "keep_alive": "1h"}}\' >/dev/null'
)


def _load_dotenv(path: Path = REPO / ".env") -> None:
    """DAYTONA_API_KEY from the repo's .env when the shell didn't export it."""
    if os.environ.get("DAYTONA_API_KEY") or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def normalize_tags(models: list[str]) -> list[str]:
    """Arm ids or bare tags -> sorted, deduplicated Ollama tags."""
    tags = sorted({m.strip().removeprefix("pi/") for m in models if m.strip()})
    if not tags:
        raise SystemExit("no models given — e.g. --models gemma4,gpt-oss:20b")
    return tags


def snapshot_name(tags: list[str], dockerfile: str, bake: bool = True) -> str:
    """Immutable, content-addressed: Daytona rejects mutable tags, and a rebuilt
    Dockerfile must not silently reuse a stale image. An unbaked image carries no
    weights, so its name carries no model list."""
    if not bake:
        return f"tetris-ollama-base-{hashlib.sha256(dockerfile.encode()).hexdigest()[:8]}"
    digest = hashlib.sha256((dockerfile + "\n" + "\n".join(tags)).encode()).hexdigest()[:8]
    slug = "-".join(re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") for t in tags)
    return f"tetris-ollama-{slug}-{digest}"[:63]


def pull_command(tags: list[str]) -> str:
    """One build-time RUN: serve, pull every tag, stop. The FUSE-free local disk
    of the builder takes ollama's partial writes; the finished blobs ship in the image."""
    pulls = " && ".join(f"ollama pull {t}" for t in tags)
    return (
        f"({SERVE_CMD}) && {WAIT_SNIPPET} && {pulls} && pkill -x ollama; "
        f"for i in $(seq 1 15); do pgrep -x ollama >/dev/null || break; sleep 1; done; "
        f"chmod -R a+rX {MODELS_DIR}"
    )


def _daytona(client=None):
    if client is not None:
        return client
    _load_dotenv()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is not set — export it or put it in .env (gitignored)")
    from daytona import Daytona  # imported here: optional dependency

    return Daytona()


def shape_of(sandbox) -> dict:
    """What Daytona actually provisioned, read off the sandbox: the numbers the bill is made of."""
    gpu_type = getattr(sandbox, "gpu_type", None)
    return {
        "cpu": float(getattr(sandbox, "cpu", 0) or 0),
        "memory_gib": float(getattr(sandbox, "memory", 0) or 0),
        "disk_gib": float(getattr(sandbox, "disk", 0) or 0),
        "gpu": float(getattr(sandbox, "gpu", 0) or 0),
        "gpu_type": getattr(gpu_type, "value", gpu_type) if gpu_type else None,
    }


def usd_per_hour(shape: dict) -> float:
    gpu_rate = USD_PER_GPU_HOUR.get(shape.get("gpu_type") or "", 0.0)
    return (
        shape.get("cpu", 0) * USD_PER_VCPU_HOUR
        + shape.get("memory_gib", 0) * USD_PER_GIB_HOUR
        + max(shape.get("disk_gib", 0) - 5, 0) * USD_PER_DISK_GIB_HOUR
        + shape.get("gpu", 0) * gpu_rate
    )


def created_epoch(sandbox) -> float:
    """The sandbox's own creation time: billing starts there, not when `up` finishes
    provisioning — the pull and the warm-up are on the meter too."""
    raw = getattr(sandbox, "created_at", None)
    if raw:
        from datetime import datetime

        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def save_state(
    sandbox_id: str,
    url: str,
    snapshot: str,
    tags: list[str],
    shape: dict | None = None,
    created_at: float | None = None,
) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {
                "sandbox_id": sandbox_id,
                "url": url,
                "snapshot": snapshot,
                "models": tags,
                "shape": shape or {},
                "created_at": created_at or time.time(),
            }
        )
    )


def cost_so_far(state: dict, now: float | None = None) -> tuple[float, float]:
    """(hours, usd) for a recorded host, at the rates above."""
    hours = ((now or time.time()) - state["created_at"]) / 3600
    return hours, hours * usd_per_hour(state.get("shape") or {})


def record_session(state: dict, ended: str) -> dict:
    """Append the finished host to the ledger and return the line."""
    hours, usd = cost_so_far(state)
    line = {
        "sandbox_id": state["sandbox_id"],
        "snapshot": state["snapshot"],
        "models": state["models"],
        "shape": state.get("shape") or {},
        "started_at": state["created_at"],
        "hours": round(hours, 4),
        "usd": round(usd, 4),
        "ended": ended,
    }
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_FILE.open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def load_state() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def remote_tags(url: str, timeout_s: float = 5.0) -> list[str] | None:
    """Model tags the host reports, or None if it does not answer."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=timeout_s) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return sorted(m["name"] for m in payload.get("models", []) if m.get("name"))


def unknown_to_pi(tags: list[str], models_json: Path = PI_MODELS_JSON) -> list[str]:
    """Tags pi's ollama provider has no entry for: --thinking is silently dropped
    for those (README, "Reasoning level"), so say so before a matrix runs at
    Ollama's default effort while claiming otherwise."""
    try:
        known = {m["id"] for m in json.loads(models_json.read_text())["providers"]["ollama"]["models"]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return list(tags)
    return [t for t in tags if t not in known and t.removesuffix(":latest") not in known]


def _snapshot_exists(d, name: str) -> bool:
    """Only not-found means no; a 401 or a network error must surface as itself."""
    from daytona import DaytonaNotFoundError

    try:
        d.snapshot.get(name)
    except DaytonaNotFoundError:
        return False
    return True


def build_snapshot(
    tags: list[str],
    name: str | None = None,
    gpu_type: str = "H100",
    disk: int = 80,
    bake: bool = True,
    client=None,
) -> str:
    """bake=False leaves the weights out: an org tier with a 5 GB snapshot cap
    (measured 2026-09-17, gemma4 alone is 11 GB) can only hold the Ollama binary,
    and `up` pulls at boot instead."""
    from daytona import CreateSnapshotParams, GpuType, Image, Resources

    dockerfile = DOCKERFILE.read_text()
    name = name or snapshot_name(tags, dockerfile, bake=bake)
    d = _daytona(client)
    if _snapshot_exists(d, name):
        print(f"[snapshot] {name} already exists — nothing to build", file=sys.stderr)
        return name
    image = Image.from_dockerfile(DOCKERFILE)
    if bake:
        image = image.run_commands(pull_command(tags))
    what = f"with {', '.join(tags)} baked in" if bake else "without weights (pulled at boot)"
    print(f"[snapshot] building {name} on Daytona {what} ...", file=sys.stderr)
    d.snapshot.create(
        CreateSnapshotParams(
            name=name,
            image=image,
            # 16 vCPU on purpose: ollama's per-token CPU work throttled decode to 34 tok/s
            # at cpu=4 against 58 tok/s on the 16-vCPU stock image (H100, gemma4, 2026-09-17).
            resources=Resources(cpu=16, memory=32, disk=disk, gpu=1, gpu_type=GpuType(gpu_type)),
        ),
        on_logs=lambda chunk: print(chunk, end="", flush=True, file=sys.stderr),
    )
    print(f"[snapshot] built {name}", file=sys.stderr)
    return name


def up(tags: list[str], snapshot: str | None = None, ttl_minutes: int = 120, client=None) -> str:
    """Boot the host, pull anything it is missing, print and return its URL."""
    from daytona import CreateSandboxFromSnapshotParams

    existing = load_state()
    if existing:
        raise SystemExit(f"host already up ({existing['url']}) — run `down` first, one host at a time")
    d = _daytona(client)
    if snapshot is None:
        # Baked first (boot only), then the weights-free base (boot + pull).
        dockerfile = DOCKERFILE.read_text()
        candidates = [snapshot_name(tags, dockerfile), snapshot_name(tags, dockerfile, bake=False)]
        snapshot = next((c for c in candidates if _snapshot_exists(d, c)), None)
        if snapshot is None:
            raise SystemExit(
                f"no snapshot for {', '.join(tags)} (looked for {' and '.join(candidates)}) — build one with\n"
                f"  uv run --group daytona scripts/daytona_host.py snapshot --models {','.join(tags)}"
                " [--no-bake]\n"
                f"or take the slow path with --snapshot {STOCK_GPU_SNAPSHOT}"
            )
    elif snapshot != STOCK_GPU_SNAPSHOT and not _snapshot_exists(d, snapshot):
        raise SystemExit(f"snapshot {snapshot!r} does not exist")

    sandbox = d.create(
        CreateSandboxFromSnapshotParams(
            snapshot=snapshot,
            labels={"purpose": "tetris-ollama-host"},
            public=True,
            ephemeral=True,
            ttl_minutes=ttl_minutes,
        ),
        timeout=600,
    )

    def step(cmd: str, timeout: int, what: str):
        print(f"[host] {what} ...", file=sys.stderr)
        r = sandbox.process.exec(cmd, timeout=timeout)
        if r.exit_code != 0:
            raise SystemExit(f"{what} failed: {r.result[-400:]}")
        return r

    try:
        step(INSTALL_CMD, 300, "install")  # a no-op on the baked image
        step(f"{SERVE_CMD}\n{WAIT_SNIPPET}", 90, "serve")
        have = step("ollama list", 60, "list").result
        for tag in tags:
            if not re.search(rf"^{re.escape(tag)}(:latest)?\s", have, re.M):
                print(f"[host] {tag} not in the image — pulling", file=sys.stderr)
                step(f"ollama pull {tag}", 1800, f"pull {tag}")
        gpu_line = step(GPU_LINE_CMD, 30, "gpu check").result.strip()
        for tag in tags:
            t0 = time.monotonic()
            step(WARM_CMD.format(port=OLLAMA_PORT, tag=tag), 960, f"load {tag} onto the GPU")
            print(f"[host] {tag} loaded in {time.monotonic() - t0:.0f}s", file=sys.stderr)
    except BaseException:
        d.delete(sandbox)  # a half-built host must not survive to bill idle
        raise

    url = sandbox.get_preview_link(OLLAMA_PORT).url
    shape = shape_of(sandbox)
    save_state(sandbox.id, url, snapshot, tags, shape, created_epoch(sandbox))
    print(
        f"[host] shape: {shape['cpu']:.0f} vCPU, {shape['memory_gib']:.0f} GiB, {shape['disk_gib']:.0f} GiB disk, "
        f"{shape['gpu']:.0f}x {shape['gpu_type'] or 'no GPU'} = ${usd_per_hour(shape):.2f}/h "
        f"(${usd_per_hour(shape) * ttl_minutes / 60:.2f} if the TTL reaps it)",
        file=sys.stderr,
    )
    print(f"[host] {gpu_line}", file=sys.stderr)
    if "library=cpu" in gpu_line:
        print(
            "[host] WARNING: ollama sees no GPU — decisions will crawl; check the snapshot's gpu reservation",
            file=sys.stderr,
        )
    seen = remote_tags(url)
    if seen is None:
        print(
            f"[host] WARNING: {url}/api/tags does not answer from here yet — preflight will say so too", file=sys.stderr
        )
    else:
        print(f"[host] serving {', '.join(seen)}", file=sys.stderr)
    missing = unknown_to_pi(tags)
    if missing:
        print(
            f"[host] pi has no entry for {', '.join(missing)} in {PI_MODELS_JSON} — --efforts will be "
            "silently ignored for those until they are listed under the ollama provider",
            file=sys.stderr,
        )
    print(f"[host] up: {url} (ttl {ttl_minutes}m)", file=sys.stderr)
    print(f'[host] next: eval "$(uv run --group daytona {Path(__file__).relative_to(REPO)} env)"', file=sys.stderr)
    print(url)
    return url


def env() -> int:
    """Shell lines that point the harness at the host: `eval "$(... env)"`."""
    state = load_state()
    if not state:
        print("# no host up — run `daytona_host.py up --models ...` first", file=sys.stderr)
        return 1
    print(f"export TETRIS_OLLAMA_URL={state['url']}")
    return 0


def status() -> int:
    state = load_state()
    if not state:
        print("no host recorded")
        return 1
    hours, usd = cost_so_far(state)
    seen = remote_tags(state["url"])
    print(f"{state['url']}  snapshot={state['snapshot']}  up {hours * 60:.0f} min  ~${usd:.2f} so far")
    print(
        "unreachable — reaped by its TTL? run `down` to clear the record"
        if seen is None
        else f"serving {', '.join(seen)}"
    )
    return 0 if seen is not None else 1


def down(client=None) -> int:
    """Tear the host down by state file; idempotent."""
    state = load_state()
    if not state:
        print("[host] no host recorded — nothing to do", file=sys.stderr)
        return 0
    d = _daytona(client)
    try:
        d.delete(d.get(state["sandbox_id"]))
        ended = "deleted"
        print(f"[host] deleted {state['sandbox_id']}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - already gone (TTL) is success, not failure
        ended = "already gone"
        print(f"[host] delete skipped ({type(exc).__name__}) — likely already reaped", file=sys.stderr)
    line = record_session(state, ended)
    print(
        f"[host] {line['hours'] * 60:.0f} min ≈ ${line['usd']:.2f} — recorded in {LEDGER_FILE.name}",
        file=sys.stderr,
    )
    STATE_FILE.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ollama on a Daytona GPU sandbox for the pi/ arms")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="build the image with the weights baked in (once per model list)")
    p_snap.add_argument("--models", required=True, help="comma-separated Ollama tags or pi/ arm ids")
    p_snap.add_argument("--name", help="override the content-addressed snapshot name")
    p_snap.add_argument("--gpu-type", default="H100", help="H100 (default), H200, RTX-PRO-6000, RTX-4090, RTX-5090")
    p_snap.add_argument("--disk", type=int, default=80, help="GiB; weights plus headroom")
    p_snap.add_argument(
        "--no-bake",
        action="store_true",
        help="leave the weights out (for an org tier whose snapshot cap is below the model); `up` pulls at boot",
    )

    p_up = sub.add_parser("up", help="boot the host and print its URL")
    p_up.add_argument("--models", required=True, help="comma-separated Ollama tags or pi/ arm ids")
    p_up.add_argument(
        "--snapshot", help=f"snapshot to boot (default: the one `snapshot` built; {STOCK_GPU_SNAPSHOT} = slow path)"
    )
    p_up.add_argument("--ttl", type=int, default=120, help="minutes before the sandbox self-reaps")

    sub.add_parser("env", help="print `export TETRIS_OLLAMA_URL=...` for eval")
    sub.add_parser("status", help="is the recorded host answering?")
    sub.add_parser("down", help="delete the host")
    args = parser.parse_args(argv)

    if args.cmd == "snapshot":
        build_snapshot(
            normalize_tags(args.models.split(",")),
            name=args.name,
            gpu_type=args.gpu_type,
            disk=args.disk,
            bake=not args.no_bake,
        )
        return 0
    if args.cmd == "up":
        up(normalize_tags(args.models.split(",")), snapshot=args.snapshot, ttl_minutes=args.ttl)
        return 0
    if args.cmd == "env":
        return env()
    if args.cmd == "status":
        return status()
    return down()


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main())
