"""Target-band schedule + offline calibration split."""

import pytest

from uni_agent.taskpilot import TargetBand, band_for_iteration, resolve_band
from uni_agent.taskpilot.calibrate import classify_candidates, solve_rates

SCHEDULE = {
    "iterations_1_to_4": {"target_resolve_rate": 0.5},
    "iteration_5": {"min_exclusive": 0.0, "max_inclusive": 0.5},
}


def _row(instance_id):
    return {"extra_info": {"tools_kwargs": {"task": {"metadata": {"instance_id": instance_id}}}}}


def test_default_band_is_the_learnable_region():
    band = TargetBand()
    assert not band.contains(0.0)  # nothing solved: too hard
    assert not band.contains(1.0)  # everything solved: saturated
    assert band.contains(1 / 8)
    assert band.contains(7 / 8)
    assert band.distance_to_target(0.25) == 0.0  # no target => age ordering only


def test_iterations_1_to_4_keep_the_region_and_prefer_the_target():
    band = resolve_band(SCHEDULE, 3)
    assert band.contains(1 / 8) and band.contains(7 / 8)
    assert band.distance_to_target(0.5) < band.distance_to_target(7 / 8)


def test_iteration_5_hard_caps_the_upper_bound():
    band = resolve_band(SCHEDULE, 5)
    assert band.contains(0.5)
    assert not band.contains(0.625)
    assert not band.contains(0.0)


def test_uncovered_iteration_without_default_is_rejected():
    with pytest.raises(ValueError, match="no target_band entry"):
        resolve_band(SCHEDULE, 6)
    assert resolve_band({**SCHEDULE, "default": {"min_exclusive": 0.25}}, 6).min_exclusive == 0.25


@pytest.mark.parametrize(
    "spec",
    [
        {"min_exclusive": 0.6, "max_inclusive": 0.4},  # inverted
        {"target_resolve_rate": 0.9, "max_inclusive": 0.5},  # target outside its own band
        {"target": 0.5},  # typo'd key
    ],
)
def test_malformed_bands_are_rejected(spec):
    with pytest.raises(ValueError):
        TargetBand.from_mapping(spec)


def test_shipped_recipe_config_matches_the_paper_schedule():
    config = "examples/frognano/configs/taskpilot.yaml"
    assert band_for_iteration(config, 1).target_resolve_rate == 0.5
    assert band_for_iteration(config, 5).max_inclusive == 0.5


def test_calibration_splits_the_pool_by_solve_rate():
    results = (
        [{"instance_id": "mixed", "resolved": i < 4} for i in range(8)]
        + [{"instance_id": "saturated", "resolved": True} for _ in range(8)]
        + [{"instance_id": "too_hard", "resolved": False} for _ in range(8)]
    )
    rates = solve_rates(results)
    assert rates == {"mixed": 0.5, "saturated": 1.0, "too_hard": 0.0}

    rows = [_row("mixed"), _row("saturated"), _row("too_hard"), _row("never_run")]
    accepted, off_band, report = classify_candidates(rows, rates, TargetBand())
    assert [_row_id(row) for row in accepted] == ["mixed"]
    assert [_row_id(row) for row in off_band] == ["saturated", "too_hard"]
    assert {entry["instance_id"]: entry["disposition"] for entry in report} == {
        "mixed": "accepted",
        "saturated": "saturated",
        "too_hard": "too_hard",
        "never_run": "uncalibrated",
    }


def _row_id(row):
    return row["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"]
