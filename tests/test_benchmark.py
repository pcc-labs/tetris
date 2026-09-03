import json

import pytest

from tetris_agent.benchmark import (
    Arm,
    ArmResult,
    build_policy,
    estimate_cost,
    expand_arms,
    main,
    render_table,
    run_matrix,
    summarize,
    write_results,
)


def fitness(score=100, pieces=30, lines=2, holes=1.0, topped=False):
    return {
        "score": score,
        "lines": lines,
        "level": 0,
        "pieces_placed": pieces,
        "avg_holes": holes,
        "max_stack_height": 5,
        "misexec_count": 0,
        "max_misexec_streak": 0,
        "topped_out": topped,
    }


def test_expand_arms_includes_control_and_skips_effort_for_haiku():
    arms = expand_arms(["claude-opus-5", "claude-haiku-4-5"], ["board"], ["low", "high"])
    names = [a.name for a in arms]
    assert names[0] == "heuristic"
    assert "claude-opus-5/board/low+live" in names
    assert "claude-opus-5/board/high+live" in names
    # Haiku rejects effort, so it collapses to a single effort-free arm.
    assert "claude-haiku-4-5/board+live" in names
    assert not any(n.startswith("claude-haiku-4-5/board/") for n in names)


def test_expand_arms_can_omit_the_control():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"], include_control=False)
    assert all(a.policy == "model" for a in arms)


def test_estimate_cost_ignores_the_free_control_arm():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"])
    with_control = estimate_cost(arms, [0], max_pieces=10)
    without = estimate_cost([a for a in arms if a.policy == "model"], [0], max_pieces=10)
    assert with_control == without > 0


def test_expand_arms_collapses_effort_for_non_thinking_pi_models():
    arms = expand_arms(["claude-opus-5", "pi/gpt-oss:20b", "pi/gemma3"], ["board"], ["low", "high"])
    names = [a.name for a in arms]
    assert "pi/gpt-oss:20b/board/low+live" in names
    assert "pi/gpt-oss:20b/board/high+live" in names
    # gemma3 has no thinking support, so it collapses like haiku does.
    assert "pi/gemma3/board+live" in names
    assert not any(n.startswith("pi/gemma3/board/") for n in names)


def test_pi_arms_are_free_in_the_estimate():
    paid = expand_arms(["claude-opus-5"], ["board"], ["low"], include_control=False)
    mixed = expand_arms(["claude-opus-5", "pi/gpt-oss:20b"], ["board"], ["low"], include_control=False)
    assert estimate_cost(mixed, [0], max_pieces=10) == estimate_cost(paid, [0], max_pieces=10) > 0
    all_pi = expand_arms(["pi/gpt-oss:20b"], ["board"], ["low"], include_control=False)
    assert estimate_cost(all_pi, [0], max_pieces=10) == 0.0


def test_build_policy_dispatches_pi_models_to_pi_policy():
    from tetris_agent.pi_policy import PiPolicy

    policy = build_policy(Arm("model", "pi/gemma3", "board", None))
    assert isinstance(policy, PiPolicy)
    assert policy.ollama_model == "gemma3"


def test_run_matrix_stops_when_the_budget_cap_is_reached():
    arms = [Arm("model", "claude-opus-5", "board", "low"), Arm("model", "claude-fable-5", "board", "low")]
    calls = []

    def fake_runner(arm, seed, rom_path, max_pieces):
        calls.append(arm.name)
        return ArmResult(arm=arm.name, seed=seed, fitness=fitness(), policy_stats={"cost_usd": 1.0})

    results = run_matrix(arms, [0, 1], rom_path="x", max_usd=1.5, runner=fake_runner)
    # First run spends $1.00; the second is allowed (spend < cap); the third is not.
    assert len(results) == 2
    assert len(calls) == 2


def test_run_matrix_without_a_cap_runs_everything():
    arms = [Arm("heuristic")]
    results = run_matrix(
        arms,
        [0, 1, 2],
        rom_path="x",
        runner=lambda a, s, r, m: ArmResult(arm=a.name, seed=s, fitness=fitness(), policy_stats={}),
    )
    assert len(results) == 3


def test_summarize_averages_seeds_and_ranks_by_race_score():
    results = [
        ArmResult("weak", 0, fitness(score=0, pieces=10), {"cost_usd": 0.5, "illegal_count": 3}),
        ArmResult("strong", 0, fitness(score=400, pieces=30), {"cost_usd": 1.0}),
        ArmResult("strong", 1, fitness(score=200, pieces=30), {"cost_usd": 1.0}),
    ]
    rows = summarize(results)
    assert [r["arm"] for r in rows] == ["strong", "weak"]
    assert rows[0]["score"] == 300.0  # averaged across the two seeds
    assert rows[0]["runs"] == 2
    assert rows[0]["cost_usd"] == 2.0  # summed, not averaged
    assert rows[1]["illegal"] == 3


def test_summarize_counts_errored_runs_without_dividing_by_zero():
    rows = summarize([ArmResult("broken", 0, {}, {}, error="boom")])
    assert rows[0]["errors"] == 1
    assert rows[0]["score"] == 0.0


def test_unmeasured_energy_renders_na_not_zero():
    """A local arm that was never sampled is unknown, not free — the old $0.0000."""
    rows = summarize([ArmResult("a", 0, fitness(), {"cost_usd": 0.0})])
    assert rows[0]["energy_wh"] == "n/a"
    assert rows[0]["energy_usd"] == "n/a"


def test_measured_energy_sums_across_seeds_and_prices():
    from tetris_agent.pricing import energy_usd

    runs = [
        ArmResult("a", 0, fitness(), {"energy_wh": 4.0}),
        ArmResult("a", 1, fitness(), {"energy_wh": 6.0}),
    ]
    rows = summarize(runs)
    assert rows[0]["energy_wh"] == 10.0  # summed like cost_usd, not averaged
    assert rows[0]["energy_usd"] == pytest.approx(energy_usd(10.0))


def test_energy_columns_appear_in_the_table():
    rows = summarize([ArmResult("a", 0, fitness(), {"energy_wh": 2.5})])
    table = render_table(rows)
    assert "energy_wh" in table and "energy_usd" in table


def test_render_table_lists_every_arm():
    rows = summarize([ArmResult("a", 0, fitness(), {}), ArmResult("b", 0, fitness(score=5), {})])
    table = render_table(rows)
    assert "race_score" in table
    assert "a" in table and "b" in table
    assert len(table.splitlines()) == 4  # header + separator + two arms


def test_write_results_persists_summary_and_runs(tmp_path):
    results = [ArmResult("a", 0, fitness(), {"cost_usd": 0.25})]
    path = write_results(results, summarize(results), out_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert saved["summary"][0]["arm"] == "a"
    assert saved["runs"][0]["seed"] == 0


def test_estimate_mode_spends_nothing(capsys):
    assert main(["--models", "claude-opus-5", "--estimate", "--max-pieces", "5"]) == 0
    out = capsys.readouterr().out
    assert "projected cost" in out
    assert "heuristic" in out


def test_expand_arms_includes_floor_chance_and_ceiling_baselines():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"])
    names = [a.name for a in arms]
    # Ordered weakest-first so the table reads as a ladder.
    assert names[:3] == ["heuristic", "random", "no-input"]
    assert "claude-opus-5/board/low+live" in names


def test_build_policy_dispatches_the_baseline_arms():
    from tetris_agent.policy import HeuristicPolicy, NoInputPolicy, RandomPolicy

    assert isinstance(build_policy(Arm("no-input")), NoInputPolicy)
    assert isinstance(build_policy(Arm("random")), RandomPolicy)
    assert isinstance(build_policy(Arm("heuristic")), HeuristicPolicy)


def test_baseline_arms_cost_nothing_in_the_estimate():
    arms = expand_arms([], [], [])
    assert [a.name for a in arms] == ["heuristic", "random", "no-input"]
    assert estimate_cost(arms, [0], max_pieces=50) == 0.0


def test_exemplar_arms_carry_a_visible_marker():
    from tetris_agent.benchmark import Arm, expand_arms

    arm = Arm(policy="model", model="claude-sonnet-5", harness="features", effort="low", exemplars=True)
    assert arm.name == "claude-sonnet-5/features/low+ex"
    arms = expand_arms(["claude-sonnet-5"], ["features"], ["low"], include_control=False, exemplars=True)
    assert all(a.exemplars for a in arms if a.policy == "model")


def test_build_policy_threads_exemplar_block_into_pi_arms():
    from tetris_agent.benchmark import Arm, build_policy

    arm = Arm(policy="model", model="pi/gemma3", harness="features", effort=None, exemplars=True)
    p = build_policy(arm, genome_params={"well_depth": 5}, exemplar_block="EXEMPLARS")
    assert "EXEMPLARS" in p.system_prompt
    assert p.genome.well_depth == 5


def test_live_arms_carry_a_visible_marker():
    arm = Arm(policy="model", model="claude-sonnet-5", harness="features", effort="low", live=True)
    assert arm.name == "claude-sonnet-5/features/low+live"
    both = Arm(policy="model", model="claude-sonnet-5", harness="features", effort="low", exemplars=True, live=True)
    assert both.name == "claude-sonnet-5/features/low+ex+live"


def test_fixed_effort_arms_carry_a_visible_marker():
    """An effort-axis matrix pins each arm's tier; the id must not collide with ladder rows."""
    from tetris_agent.benchmark import Arm, expand_arms

    arm = Arm(policy="model", model="claude-sonnet-5", harness="features", effort="low", live=True, fixed_effort=True)
    assert arm.name == "claude-sonnet-5/features/low+live+fixed"
    arms = expand_arms(["pi/gpt-oss:20b"], ["features"], ["off", "low"], include_control=False, fixed_effort=True)
    assert [a.name for a in arms] == [
        "pi/gpt-oss:20b/features/off+live+fixed",
        "pi/gpt-oss:20b/features/low+live+fixed",
    ]


def test_build_policy_pins_the_effort_ladder_for_fixed_arms():
    from tetris_agent.benchmark import Arm, build_policy

    arm = Arm(policy="model", model="pi/gpt-oss:20b", harness="features", effort="low", live=True, fixed_effort=True)
    assert build_policy(arm)._effort_ladder() == ["low"]


def test_expand_arms_defaults_model_arms_to_live_and_baselines_not():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"])
    names = [a.name for a in arms]
    assert names[:3] == ["heuristic", "random", "no-input"]
    assert "claude-opus-5/board/low+live" in names
    assert all(not a.live for a in arms if a.policy != "model")


def test_expand_arms_live_false_restores_legacy_names():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"], live=False)
    assert "claude-opus-5/board/low" in [a.name for a in arms]


def test_run_arm_live_builds_a_paced_emulator_and_live_agent(monkeypatch):
    from tetris_agent.benchmark import run_arm

    captured = {}

    class FakeEmu:
        def __init__(self, rom, headless=True, speed=0, **kw):
            captured["emu"] = {"headless": headless, "speed": speed}

        def stop(self):
            captured["stopped"] = True

    class FakeLiveAgent:
        def __init__(self, emu, genome=None, collector=None, recorder=None, max_pieces=0, policy=None, **kwargs):
            self.live_stats = {"late": 2, "worker_errors": 0, "in_flight_at_end": False}

        def run(self, timer_div=None, level=0):
            captured["level"] = level
            return fitness()

    monkeypatch.setattr("tetris_agent.emulator.Emulator", FakeEmu)
    monkeypatch.setattr("tetris_agent.live_agent.LiveTetrisAgent", FakeLiveAgent)
    arm = Arm("model", "pi/gemma3", "features", None, live=True)

    result = run_arm(arm, seed=0, rom_path="rom.gb", max_pieces=5, level=3)

    assert captured["emu"] == {"headless": True, "speed": 1}
    assert captured["level"] == 3
    assert captured["stopped"] is True
    assert result.policy_stats["late"] == 2


def test_run_arm_paused_keeps_the_uncapped_emulator(monkeypatch):
    from tetris_agent.benchmark import run_arm

    captured = {}

    class FakeEmu:
        def __init__(self, rom, headless=True, speed=0, **kw):
            captured["emu"] = {"headless": headless, "speed": speed}

        def stop(self):
            pass

    class FakeAgent:
        def __init__(self, emu, genome=None, collector=None, recorder=None, max_pieces=0, policy=None, **kwargs):
            pass

        def run(self, timer_div=None):
            return fitness()

    monkeypatch.setattr("tetris_agent.emulator.Emulator", FakeEmu)
    monkeypatch.setattr("tetris_agent.agent.TetrisAgent", FakeAgent)
    arm = Arm("model", "pi/gemma3", "features", None, live=False)

    result = run_arm(arm, seed=0, rom_path="rom.gb", max_pieces=5)

    assert result.error == ""  # the agent really ran; nothing was swallowed
    assert captured["emu"]["speed"] == 0
    assert "late" not in result.policy_stats


def test_summarize_sums_late_decisions():
    results = [
        ArmResult("live-arm", 0, fitness(), {"late": 3}),
        ArmResult("live-arm", 1, fitness(), {"late": 2}),
    ]
    assert summarize(results)[0]["late"] == 5


def test_render_table_includes_late_column():
    table = render_table(summarize([ArmResult("a", 0, fitness(), {})]))
    assert "late" in table.splitlines()[0].split()


def test_estimate_paused_has_no_live_marker(capsys):
    assert main(["--models", "claude-opus-5", "--estimate", "--max-pieces", "5", "--paused"]) == 0
    out = capsys.readouterr().out
    assert "+live" not in out
    assert "wall-clock" not in out


def test_estimate_live_marks_arms_and_warns_about_wall_clock(capsys):
    assert main(["--models", "claude-opus-5", "--estimate", "--max-pieces", "5"]) == 0
    out = capsys.readouterr().out
    assert "+live" in out
    assert "wall-clock" in out


def test_summarize_reports_mean_tokens_per_second():
    results = [
        ArmResult("a", 0, fitness(), {"tokens_per_second": 30.0}),
        ArmResult("a", 1, fitness(), {"tokens_per_second": 10.0}),
    ]
    assert summarize(results)[0]["tok_s"] == 20.0


def test_render_table_includes_tok_s_column():
    table = render_table(summarize([ArmResult("a", 0, fitness(), {})]))
    assert "tok_s" in table.splitlines()[0].split()


# ---- bounded pause (+p15) and --watch --------------------------------------


def test_paused_deadline_arms_carry_a_visible_marker():
    arm = Arm("model", "pi/gpt-oss:20b", "features", "medium", live=False, deadline_s=15.0)
    assert arm.name == "pi/gpt-oss:20b/features/medium+p15"
    assert Arm("model", "pi/gemma3", "board", None, live=False).name == "pi/gemma3/board"


def test_expand_arms_threads_the_deadline_into_model_arms():
    arms = expand_arms(["pi/gpt-oss:20b"], ["features"], ["medium"], include_control=False, live=False, deadline_s=15.0)
    assert [a.deadline_s for a in arms] == [15.0]
    assert arms[0].name.endswith("+p15")


def test_build_policy_arms_pi_with_a_hard_deadline_only_when_bounded():
    bounded = build_policy(Arm("model", "pi/gemma3", "board", None, live=False, deadline_s=15.0))
    live = build_policy(Arm("model", "pi/gemma3", "board", None, live=True))
    assert bounded.hard_deadline is True
    assert live.hard_deadline is False


def test_run_arm_passes_deadline_and_meta_to_the_paused_agent(monkeypatch):
    from tetris_agent.benchmark import run_arm

    captured = {}

    class FakeEmu:
        def __init__(self, rom, headless=True, speed=0, **kw):
            captured["emu"] = {"headless": headless, "speed": speed}

        def stop(self):
            pass

    class FakeAgent:
        def __init__(self, emu, genome=None, collector=None, recorder=None, max_pieces=0, policy=None, **kwargs):
            captured["agent_kwargs"] = kwargs
            captured["collector"] = collector
            self.paused_stats = {"timeouts": 3, "decision_latencies_ms": [5000.0]}

        def run(self, timer_div=None):
            return fitness()

    monkeypatch.setattr("tetris_agent.emulator.Emulator", FakeEmu)
    monkeypatch.setattr("tetris_agent.agent.TetrisAgent", FakeAgent)
    arm = Arm("model", "pi/gemma3", "features", None, live=False, deadline_s=15.0)

    result = run_arm(arm, seed=7, rom_path="rom.gb", max_pieces=5)

    kwargs = captured["agent_kwargs"]
    assert kwargs["decision_deadline_s"] == 15.0
    assert kwargs["session_meta"]["model"] == "pi/gemma3"
    assert kwargs["session_meta"]["arm"] == arm.name
    assert kwargs["session_meta"]["seed"] == 7
    assert kwargs["session_meta"]["max_pieces"] == 5
    assert result.policy_stats["timeouts"] == 3


def test_run_arm_watch_streams_events_and_paces_the_emulator(monkeypatch):
    from tetris_agent.benchmark import run_arm

    captured = {}
    sent = []

    class FakeStreamer:
        def send_event(self, event):
            sent.append(event)

        def send_frame(self, turn, png):
            sent.append(("frame", turn))

    class FakeEmu:
        def __init__(self, rom, headless=True, speed=0, **kw):
            captured["emu"] = {"headless": headless, "speed": speed}
            self.frame_hook = None

        def stop(self):
            pass

    class FakeAgent:
        def __init__(self, emu, genome=None, collector=None, recorder=None, max_pieces=0, policy=None, **kwargs):
            captured["collector"] = collector
            captured["frame_sinks"] = kwargs.get("frame_sinks")
            self.paused_stats = {}

        def run(self, timer_div=None):
            captured["collector"].session("start", {})
            return fitness()

    monkeypatch.setattr("tetris_agent.emulator.Emulator", FakeEmu)
    monkeypatch.setattr("tetris_agent.agent.TetrisAgent", FakeAgent)
    arm = Arm("model", "pi/gemma3", "features", None, live=False, deadline_s=15.0)

    run_arm(arm, seed=0, rom_path="rom.gb", max_pieces=5, streamer=FakeStreamer())

    # Watching means real time: the paused emulator is paced, not uncapped.
    assert captured["emu"]["speed"] == 1
    assert any(isinstance(e, dict) and e.get("event_type") == "session" for e in sent)
    assert captured["frame_sinks"], "frames must stream to the viewer"


def test_summarize_reports_timeouts_and_deadline_hit_rates():
    results = [
        ArmResult(
            arm="pi/x/features+p15",
            seed=0,
            fitness=fitness(),
            policy_stats={
                "timeouts": 1,
                "decision_latencies_ms": [5000.0, 12000.0, 20000.0],
            },
        )
    ]
    row = summarize(results)[0]
    assert row["timeouts"] == 1
    assert row["pct_le_10s"] == 33.3
    assert row["pct_le_15s"] == 66.7


def test_render_table_includes_the_deadline_columns():
    rows = summarize([ArmResult(arm="a", seed=0, fitness=fitness(), policy_stats={})])
    table = render_table(rows)
    assert "timeouts" in table and "pct_le_15s" in table


def test_lookahead_arm_is_opt_in_and_absent_by_default():
    """A command run yesterday must produce the same rows tomorrow."""
    default = [a.name for a in expand_arms(["claude-opus-5"], ["board"], ["low"])]
    assert "lookahead" not in default

    opted = [a.name for a in expand_arms(["claude-opus-5"], ["board"], ["low"], lookahead_control=True)]
    assert "lookahead" in opted
    assert opted[: len(default)] == default  # appended, nothing reordered


def test_build_policy_dispatches_the_lookahead_arm():
    from tetris_agent.policy import LookaheadPolicy

    assert isinstance(build_policy(Arm("lookahead")), LookaheadPolicy)


def test_lookahead_arm_costs_nothing_in_the_estimate():
    arms = expand_arms([], [], [], lookahead_control=True)
    assert estimate_cost(arms, [0], max_pieces=50) == 0.0


def test_decision_deadline_flag_implies_paused_and_names_the_arms(capsys):
    rc = main(
        [
            "--models",
            "pi/gemma3",
            "--harnesses",
            "board",
            "--decision-deadline",
            "15",
            "--no-control",
            "--estimate",
            "--skip-preflight",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "+p15" in out
    assert "+live" not in out
