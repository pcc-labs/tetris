import json

from tetris_agent.genome import append_genome, load_params
from tetris_agent.policy import Genome


def test_load_missing_file_returns_defaults(tmp_path):
    params = load_params(tmp_path / "genome.json")
    assert params == Genome().to_params()


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "genome.json"
    tuned = {**Genome().to_params(), "w_holes": -0.9}
    append_genome(tuned, source="healer:hole-thrash", path=path)
    assert load_params(path)["w_holes"] == -0.9
    append_genome({**tuned, "w_holes": -1.1}, source="healer:hole-thrash", path=path)
    stored = json.loads(path.read_text())
    assert len(stored["history"]) == 2
    assert stored["history"][-1]["source"] == "healer:hole-thrash"
    assert load_params(path)["w_holes"] == -1.1


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "genome.json"
    path.write_text("{not json")
    assert load_params(path) == Genome().to_params()


def test_partial_stored_params_merge_over_defaults(tmp_path):
    path = tmp_path / "genome.json"
    path.write_text(json.dumps({"current": {"w_lines": 2.0}, "history": []}))
    params = load_params(path)
    assert params["w_lines"] == 2.0
    assert params["w_holes"] == Genome().w_holes
