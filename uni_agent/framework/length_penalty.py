"""Success-gated log-length penalty over a session's trajectories.

Binary test-pass rewards say nothing about how many tokens the solution cost, so a
long-horizon SWE policy drifts toward rambling trajectories. This shapes only
*successful* ones, leaving a free budget and then charging logarithmically::

    R(n) = 1                     n <= free_tokens
    R(n) = 1 - alpha*ln(n/free)  n >  free_tokens

``n`` counts model-generated tokens (reasoning + text + tool calls, excluding tool
observations), which is exactly the trajectory's response mask. A trajectory that
scored but never terminated (``finished is False``) is credited ``truncated_reward``
instead: it solved the task without committing to an answer, so it is neither a clean
success nor a failure. ``finished is None`` means "unknown" and is left unshaped.

Failures stay at their unshaped reward, and ``reward_metrics`` is left alone --
group filtering (see :mod:`uni_agent.taskpilot.sampler`) therefore keeps banding on
raw pass/fail rather than on shaped reward.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["LengthPenaltyConfig", "apply_length_penalty"]


@dataclass(frozen=True)
class LengthPenaltyConfig:
    """Args:
    free_tokens: Model-generated token budget that costs nothing.
    alpha: Penalty strength on ``ln(n / free_tokens)``.
    truncated_reward: Reward for a successful but unfinished (budget-truncated) trajectory.
    """

    free_tokens: int
    alpha: float = 0.1
    truncated_reward: float = 0.5

    @classmethod
    def from_config(cls, af_cfg: Any) -> LengthPenaltyConfig | None:
        """Build from ``actor_rollout_ref.rollout.custom.agent_framework.length_penalty``, or None when off."""
        cfg = af_cfg.get("length_penalty") if af_cfg else None
        if not cfg or not bool(cfg.get("enable", False)):
            return None
        free_tokens = int(cfg.get("free_tokens", 0))
        if free_tokens <= 0:
            raise ValueError("agent_framework.length_penalty.free_tokens must be positive when the penalty is enabled")
        return cls(
            free_tokens=free_tokens,
            alpha=float(cfg.get("alpha", 0.1)),
            truncated_reward=float(cfg.get("truncated_reward", 0.5)),
        )


def apply_length_penalty(trajectories: list, config: LengthPenaltyConfig) -> list:
    """Return ``trajectories`` with successful rewards shaped by their generated length."""
    shaped = []
    for traj in trajectories:
        reward = traj.reward_score
        if reward is None or reward <= 0:
            shaped.append(traj)
            continue
        if traj.finished is False:
            reward = config.truncated_reward
        else:
            num_tokens = sum(traj.response_mask) if traj.response_mask else 0
            if num_tokens > config.free_tokens:
                reward *= 1.0 - config.alpha * math.log(num_tokens / config.free_tokens)
        shaped.append(replace(traj, reward_score=float(reward)))
    return shaped
