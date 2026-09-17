"""Success-gated log-length reward shaping."""

import math
from dataclasses import dataclass, field

import pytest

from uni_agent.framework.length_penalty import LengthPenaltyConfig, apply_length_penalty


@dataclass
class _Traj:
    """Minimal stand-in for a gateway Trajectory (only the fields the shaper reads)."""

    reward_score: float
    response_mask: list = field(default_factory=list)
    finished: bool | None = True
    reward_metrics: dict = field(default_factory=dict)


def _traj(reward, num_tokens, finished=True):
    return _Traj(reward_score=reward, response_mask=[1] * num_tokens, finished=finished)


CONFIG = LengthPenaltyConfig(free_tokens=100, alpha=0.1)


def test_success_within_budget_is_untouched():
    assert apply_length_penalty([_traj(1.0, 100)], CONFIG)[0].reward_score == 1.0


def test_success_over_budget_pays_a_log_penalty():
    shaped = apply_length_penalty([_traj(1.0, 300)], CONFIG)[0].reward_score
    assert shaped == pytest.approx(1.0 - 0.1 * math.log(3.0))
    assert shaped < apply_length_penalty([_traj(1.0, 200)], CONFIG)[0].reward_score


def test_failures_are_never_shaped():
    assert apply_length_penalty([_traj(0.0, 10_000)], CONFIG)[0].reward_score == 0.0


def test_correct_but_truncated_gets_partial_credit():
    shaped = apply_length_penalty([_traj(1.0, 10_000, finished=False)], CONFIG)[0].reward_score
    assert shaped == 0.5


def test_unknown_completion_is_shaped_as_a_success():
    """`finished` is tri-state; None means unknown, which must not read as truncated."""
    shaped = apply_length_penalty([_traj(1.0, 300, finished=None)], CONFIG)[0].reward_score
    assert shaped == pytest.approx(1.0 - 0.1 * math.log(3.0))


def test_disabled_by_default_and_requires_a_budget():
    assert LengthPenaltyConfig.from_config({}) is None
    assert LengthPenaltyConfig.from_config({"length_penalty": {"enable": False, "free_tokens": 10}}) is None
    assert LengthPenaltyConfig.from_config({"length_penalty": {"enable": True, "free_tokens": 10}}).free_tokens == 10
    with pytest.raises(ValueError, match="free_tokens"):
        LengthPenaltyConfig.from_config({"length_penalty": {"enable": True}})
