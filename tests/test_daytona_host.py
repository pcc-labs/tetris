"""scripts/daytona_host.py against a fake SDK — the whole surface the driver touches."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "daytona_host.py"


@pytest.fixture
def host(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("daytona_host", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "host.json")
    monkeypatch.setattr(mod, "LEDGER_FILE", tmp_path / "hosts.jsonl")
    monkeypatch.setattr(mod, "remote_tags", lambda url, timeout_s=5.0: ["gemma4:latest"])
    pi_models = tmp_path / "models.json"
    pi_models.write_text(json.dumps({"providers": {"ollama": {"models": [{"id": "gemma4"}, {"id": "gpt-oss:20b"}]}}}))
    monkeypatch.setattr(mod, "PI_MODELS_JSON", pi_models)
    return mod


class Exec:
    def __init__(self, exit_code, result=""):
        self.exit_code, self.result = exit_code, result


class FakeSandbox:
    cpu, memory, disk, gpu = 4, 16, 40, 1
    gpu_type = types.SimpleNamespace(value="H100")
    created_at = "2026-09-17T14:00:00.000Z"

    def __init__(self, listed="gemma4:latest  abc  17 GB\n", fail_on=None):
        self.id = "sb-1"
        self.commands = []
        self.listed = listed
        self.fail_on = fail_on
        self.process = self
        self.deleted = False

    def exec(self, command, cwd=None, env=None, timeout=None):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return Exec(1, "boom")
        if command == "ollama list":
            return Exec(0, "NAME  ID  SIZE\n" + self.listed)
        if "inference compute" in command:
            return Exec(0, 'inference compute id=GPU-0 library=cuda name="NVIDIA H100 80GB"')
        return Exec(0, "")

    def get_preview_link(self, port):
        return types.SimpleNamespace(url=f"https://{port}-sb-1.proxy.daytona.works")


class FakeClient:
    def __init__(self, sandbox=None, snapshots=("tetris-ollama-any",)):
        self.sandbox = sandbox or FakeSandbox()
        self.params = None
        self.deleted = []
        self.snapshot = self
        self._snapshots = set(snapshots)
        self.created_snapshot = None

    # Doubles as daytona.Daytona.create (sandbox params) and daytona.snapshot.create (image params).
    def create(self, params, *, timeout=None, on_logs=None):
        if hasattr(params, "image"):
            self.created_snapshot = params
            self._snapshots.add(params.name)
            return None
        self.params = params
        return self.sandbox

    def delete(self, sandbox):
        self.deleted.append(sandbox.id)

    def get(self, name):
        if name in self._snapshots:
            return types.SimpleNamespace(name=name)
        if name == self.sandbox.id:
            return self.sandbox
        raise LookupError(name)


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = types.ModuleType("daytona")

    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class GpuType:
        def __init__(self, value):
            self.value = value

    class Image:
        def __init__(self, text):
            self.text = text

        @staticmethod
        def from_dockerfile(path):
            return Image(Path(path).read_text())

        def run_commands(self, *cmds):
            self.text += "".join(f"RUN {c}\n" for c in cmds)
            return self

    sdk.CreateSandboxFromSnapshotParams = Params
    sdk.CreateSnapshotParams = Params
    sdk.Resources = Params
    sdk.GpuType = GpuType
    sdk.Image = Image
    sdk.DaytonaNotFoundError = LookupError
    sdk.Daytona = lambda: (_ for _ in ()).throw(AssertionError("tests inject a client"))
    monkeypatch.setitem(sys.modules, "daytona", sdk)
    return sdk


def test_tags_drop_the_arm_prefix_and_dedupe(host):
    assert host.normalize_tags(["pi/gemma4", "gemma4", " gpt-oss:20b "]) == ["gemma4", "gpt-oss:20b"]
    with pytest.raises(SystemExit):
        host.normalize_tags([" "])


def test_snapshot_name_is_content_addressed(host):
    a = host.snapshot_name(["gemma4", "gpt-oss:20b"], "FROM ubuntu\n")
    assert a.startswith("tetris-ollama-gemma4-gpt-oss-20b-") and len(a) <= 63
    assert a == host.snapshot_name(["gemma4", "gpt-oss:20b"], "FROM ubuntu\n")
    assert a != host.snapshot_name(["gemma4", "gpt-oss:20b"], "FROM ubuntu:24.04\n")  # Dockerfile change -> new name
    assert a != host.snapshot_name(["gemma4"], "FROM ubuntu\n")


def test_pull_command_serves_pulls_every_tag_and_stops(host):
    cmd = host.pull_command(["gemma4", "gpt-oss:20b"])
    assert "ollama serve" in cmd and "ollama pull gemma4" in cmd and "ollama pull gpt-oss:20b" in cmd
    assert "pkill -x ollama" in cmd and "exit" not in cmd


def test_snapshot_build_bakes_the_pull_and_reserves_a_gpu(host, fake_sdk):
    client = FakeClient(snapshots=())
    name = host.build_snapshot(["gemma4"], gpu_type="H100", client=client)
    p = client.created_snapshot
    assert p.name == name and name.startswith("tetris-ollama-gemma4-")
    assert "ollama pull gemma4" in p.image.text and "install.sh" in p.image.text
    assert p.resources.gpu == 1 and p.resources.gpu_type.value == "H100"


def test_snapshot_build_is_idempotent(host, fake_sdk, capsys):
    client = FakeClient(snapshots=(host.snapshot_name(["gemma4"], host.DOCKERFILE.read_text()),))
    host.build_snapshot(["gemma4"], client=client)
    assert client.created_snapshot is None
    assert "already exists" in capsys.readouterr().err


def test_up_boots_public_ephemeral_host_and_records_it(host, fake_sdk, capsys):
    client = FakeClient(snapshots=(host.snapshot_name(["gemma4"], host.DOCKERFILE.read_text()),))
    url = host.up(["gemma4"], ttl_minutes=45, client=client)

    assert url == "https://11434-sb-1.proxy.daytona.works"
    p = client.params
    assert p.public is True and p.ephemeral is True and p.ttl_minutes == 45
    assert p.labels == {"purpose": "tetris-ollama-host"}
    cmds = client.sandbox.commands
    assert any("ollama serve" in c for c in cmds)
    assert not any(c.startswith("ollama pull") for c in cmds)  # baked in, nothing to pull
    assert json.loads(host.STATE_FILE.read_text())["sandbox_id"] == "sb-1"
    assert capsys.readouterr().out.strip() == url
    assert client.deleted == []


def test_up_pulls_what_the_image_is_missing(host, fake_sdk):
    client = FakeClient(sandbox=FakeSandbox(listed="gemma4:latest  x  1 GB\n"))
    host.up(["gemma4", "gpt-oss:20b"], snapshot=host.STOCK_GPU_SNAPSHOT, client=client)
    assert "ollama pull gpt-oss:20b" in client.sandbox.commands
    assert "ollama pull gemma4" not in client.sandbox.commands


def test_up_refuses_a_snapshot_that_was_never_built(host, fake_sdk):
    with pytest.raises(SystemExit, match="no snapshot for gemma4"):
        host.up(["gemma4"], client=FakeClient(snapshots=()))
    with pytest.raises(SystemExit, match="does not exist"):
        host.up(["gemma4"], snapshot="tetris-ollama-nope", client=FakeClient(snapshots=()))


def test_unbaked_snapshot_has_no_weights_and_up_falls_back_to_it(host, fake_sdk):
    client = FakeClient(snapshots=())
    name = host.build_snapshot(["gemma4"], bake=False, client=client)
    assert name.startswith("tetris-ollama-base-")
    assert name == host.snapshot_name([], host.DOCKERFILE.read_text(), bake=False)  # model list not in the name
    assert "RUN (" not in client.created_snapshot.image.text  # no pull step
    assert "install.sh" in client.created_snapshot.image.text
    # The base image is what `up` boots when no baked one exists, and it pulls there.
    client.sandbox = FakeSandbox(listed="")
    host.up(["gemma4"], client=client)
    assert client.params.snapshot == name
    assert "ollama pull gemma4" in client.sandbox.commands


def test_up_deletes_a_half_built_host(host, fake_sdk):
    client = FakeClient(sandbox=FakeSandbox(fail_on="ollama serve"))
    with pytest.raises(SystemExit, match="serve failed"):
        host.up(["gemma4"], snapshot=host.STOCK_GPU_SNAPSHOT, client=client)
    assert client.deleted == ["sb-1"]
    assert host.load_state() is None


def test_up_is_one_host_at_a_time(host, fake_sdk):
    host.save_state("sb-0", "https://old", "snap", ["gemma4"])
    with pytest.raises(SystemExit, match="already up"):
        host.up(["gemma4"], client=FakeClient())


def test_up_warns_when_ollama_sees_no_gpu(host, fake_sdk, capsys):
    class CpuSandbox(FakeSandbox):
        def exec(self, command, cwd=None, env=None, timeout=None):
            if "inference compute" in command:
                return Exec(0, "inference compute id=cpu library=cpu")
            return super().exec(command, cwd, env, timeout)

    host.up(["gemma4"], snapshot=host.STOCK_GPU_SNAPSHOT, client=FakeClient(sandbox=CpuSandbox()))
    assert "sees no GPU" in capsys.readouterr().err


def test_env_and_down_round_trip(host, fake_sdk, capsys):
    client = FakeClient()
    host.up(["gemma4"], snapshot=host.STOCK_GPU_SNAPSHOT, client=client)
    capsys.readouterr()
    assert host.env() == 0
    assert capsys.readouterr().out.strip() == "export TETRIS_OLLAMA_URL=https://11434-sb-1.proxy.daytona.works"
    assert host.down(client=client) == 0
    assert client.deleted == ["sb-1"] and host.load_state() is None
    assert host.down(client=client) == 0  # idempotent
    assert host.env() == 1


def test_down_survives_a_host_the_ttl_already_reaped(host, fake_sdk, capsys):
    host.save_state("gone", "https://x", "snap", ["gemma4"])
    assert host.down(client=FakeClient()) == 0
    assert "already reaped" in capsys.readouterr().err and host.load_state() is None


def test_unknown_to_pi_reads_the_ollama_provider(host, tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"providers": {"ollama": {"models": [{"id": "gemma4"}]}}}))
    assert host.unknown_to_pi(["gemma4", "gemma4:latest", "gpt-oss:20b"], models) == ["gpt-oss:20b"]
    assert host.unknown_to_pi(["gemma4"], tmp_path / "missing.json") == ["gemma4"]


def test_dotenv_fills_only_a_missing_key(host, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# keys\nDAYTONA_API_KEY='dtn_x'\nOTHER=1\n")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    host._load_dotenv(env)
    import os

    assert os.environ["DAYTONA_API_KEY"] == "dtn_x"
    monkeypatch.setenv("DAYTONA_API_KEY", "shell")
    host._load_dotenv(env)
    assert os.environ["DAYTONA_API_KEY"] == "shell"


def test_host_cost_is_priced_from_what_was_provisioned(host):
    shape = host.shape_of(FakeSandbox())
    assert shape == {"cpu": 4.0, "memory_gib": 16.0, "disk_gib": 40.0, "gpu": 1.0, "gpu_type": "H100"}
    rate = host.usd_per_hour(shape)
    assert rate == pytest.approx(4 * 0.0504 + 16 * 0.0162 + 35 * 0.000108 + 2.27)
    assert host.usd_per_hour({**shape, "gpu": 0, "gpu_type": None}) == pytest.approx(
        4 * 0.0504 + 16 * 0.0162 + 35 * 0.000108
    )


def test_down_records_the_session_cost(host, fake_sdk, capsys):
    client = FakeClient()
    host.up(["gemma4"], snapshot=host.STOCK_GPU_SNAPSHOT, client=client)
    import time

    state = host.load_state()
    state["created_at"] = time.time() - 1800  # half an hour ago
    host.STATE_FILE.write_text(json.dumps(state))
    host.down(client=client)
    line = json.loads(host.LEDGER_FILE.read_text().splitlines()[-1])
    assert line["sandbox_id"] == "sb-1" and line["ended"] == "deleted"
    assert line["hours"] == pytest.approx(0.5, abs=1e-3)
    assert line["usd"] == pytest.approx(0.5 * host.usd_per_hour(state["shape"]), abs=1e-3)
    assert "30 min" in capsys.readouterr().err


def test_state_starts_the_meter_at_the_sandboxs_creation(host, fake_sdk):
    host.up(["gemma4"], snapshot=host.STOCK_GPU_SNAPSHOT, client=FakeClient())
    from datetime import datetime, timezone

    assert host.load_state()["created_at"] == datetime(2026, 9, 17, 14, tzinfo=timezone.utc).timestamp()
