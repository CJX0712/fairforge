"""Tests: benchmark runner + pipeline end-to-end + CLI.

Author: 晨星
"""

import json

import numpy as np

from fairforge.data.generator import generate_scenario
from fairforge.pipeline.benchmark import (
    DEFAULT_MITIGATORS,
    environment_fingerprint,
    run_benchmark,
    run_single,
    save_benchmark,
)


def test_environment_fingerprint_fields():
    fp = environment_fingerprint()
    assert {
        "platform",
        "python",
        "numpy",
        "scikit_learn",
        "fairlearn",
        "optuna",
        "backends_available",
    } <= set(fp)


def test_run_benchmark_small_matrix(tmp_path):
    entries = run_benchmark(
        scenarios=["mixed"],
        mitigators=["reweighing", "reject_option"],
        seed=42,
        n_per_group=250,
    )
    assert len(entries) == 2
    for e in entries:
        assert {
            "scenario",
            "mitigator",
            "backend",
            "used_fallback",
            "accuracy",
            "metrics",
            "n_trials",
            "elapsed_s",
            "seed",
            "dependency_versions",
            "environment_fingerprint",
        } <= set(e)
        assert set(e["metrics"]) == {
            "demographic_parity_diff",
            "equal_opportunity_diff",
            "equalized_odds_diff",
            "average_odds_diff",
            "disparate_impact",
            "theil_index",
            "calibration_by_group",
        }
        assert 0.0 <= e["accuracy"] <= 1.0
    out = tmp_path / "bench.json"
    save_benchmark(entries, out, extra={"flagship": {"ok": True}})
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["flagship"] == {"ok": True}
    assert len(payload["entries"]) == 2


def test_run_single_deterministic_seed():
    ds = generate_scenario("mixed", n_per_group=300, seed=42)
    _, p1, _ = run_single(ds, "threshold_optimizer", seed=42)
    _, p2, _ = run_single(ds, "threshold_optimizer", seed=42)
    assert np.array_equal(p1, p2)


def test_default_mitigator_list_complete():
    assert set(DEFAULT_MITIGATORS) == {
        "reweighing",
        "exponentiated_gradient",
        "grid_search",
        "threshold_optimizer",
        "group_threshold",
        "reject_option",
    }


def test_cli_backend_command(capsys):
    from fairforge.cli import main

    assert main(["backend"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "available" in payload and "fairlearn" in payload["available"]


def test_cli_data_command(capsys):
    from fairforge.cli import main

    assert main(["data", "--scenario", "mixed", "--n-per-group", "100"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_samples"] == 200
    assert payload["group_counts"] == {"0": 100, "1": 100}


def test_cli_run_command(tmp_path, capsys):
    from fairforge.cli import main

    out = tmp_path / "cli_bench.json"
    assert (
        main(["run", "--scenario", "mixed", "--mitigator", "reject_option", "--out", str(out)]) == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1


def test_version_and_author():
    import fairforge

    assert fairforge.__version__ == "0.1.0"
    assert fairforge.__author__ == "晨星"
