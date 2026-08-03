import random

import pytest

from tetris_agent.evolve import decide, run_race, sample_variants
from tetris_agent.policy import Genome


def test_sample_variants_only_perturbs_implicated_keys():
    base = Genome().to_params()
    variants = sample_variants(base, ["w_holes"], n=5, rng=random.Random(7))
    for v in variants:
        assert v["w_holes"] != base["w_holes"]
        for key in base:
            if key != "w_holes":
                assert v[key] == base[key]


def test_sample_variants_deterministic_given_seed():
    base = Genome().to_params()
    a = sample_variants(base, ["w_lines", "w_holes"], n=3, rng=random.Random(42))
    b = sample_variants(base, ["w_lines", "w_holes"], n=3, rng=random.Random(42))
    assert a == b


def test_sample_variants_int_params_stay_positive_ints():
    base = Genome().to_params()
    variants = sample_variants(base, ["ticks_per_press"], n=10, rng=random.Random(1))
    for v in variants:
        assert isinstance(v["ticks_per_press"], int)
        assert v["ticks_per_press"] >= 1


def test_run_race_averages_seeds_and_picks_winner():
    base = {"w_holes": -0.3}
    good = {"w_holes": -0.9}
    bad = {"w_holes": 0.1}

    def fake_runner(params, timer_div, rom_path, max_pieces=150):
        table = {-0.3: 100, -0.9: 200, 0.1: 10}
        return {"score": table[params["w_holes"]] + timer_div, "pieces_placed": 0}

    result = run_race(base, [good, bad], seeds=[0, 10], rom_path="unused", runner=fake_runner)
    assert result["control"] == 105.0
    assert result["winner"]["params"] == good
    assert result["winner"]["score"] == 205.0
    assert len(result["candidates"]) == 2


def test_decide_requires_margin():
    assert decide(100.0, 106.0) is True
    assert decide(100.0, 104.0) is False
    assert decide(0.0, 5.0) is False  # tiny scores need an absolute improvement
    assert decide(0.0, 15.0) is True


@pytest.mark.rom
def test_run_agent_in_process_smoke(rom_path):
    from tetris_agent.evolve import run_agent
    from tetris_agent.policy import Genome

    fitness = run_agent(Genome().to_params(), timer_div=0, rom_path=rom_path, max_pieces=10)
    assert fitness["pieces_placed"] == 10
