import json

from tetris_agent.benchmark import (
    Arm,
    ArmResult,
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
    assert "claude-opus-5/board/low" in names
    assert "claude-opus-5/board/high" in names
    # Haiku rejects effort, so it collapses to a single effort-free arm.
    assert "claude-haiku-4-5/board" in names
    assert not any(n.startswith("claude-haiku-4-5/board/") for n in names)


def test_expand_arms_can_omit_the_control():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"], include_control=False)
    assert all(a.policy == "model" for a in arms)


def test_estimate_cost_ignores_the_free_control_arm():
    arms = expand_arms(["claude-opus-5"], ["board"], ["low"])
    with_control = estimate_cost(arms, [0], max_pieces=10)
    without = estimate_cost([a for a in arms if a.policy == "model"], [0], max_pieces=10)
    assert with_control == without > 0


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
