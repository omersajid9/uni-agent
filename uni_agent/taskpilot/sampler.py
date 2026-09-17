"""TaskPilot sampler: keep only the groups that are trainable *for this policy*.

Plugged in as verl's custom sampler, so no trainer changes are needed::

    trainer.v1.sampler.custom_sampler.path=pkg://uni_agent.taskpilot.sampler
    trainer.v1.sampler.custom_sampler.name=TaskPilotSampler
    ++trainer.v1.sampler.sampler_kwargs.config_path=examples/frognano/configs/taskpilot.yaml
    ++trainer.v1.sampler.sampler_kwargs.iteration=1

Every step, rollout produces ``n`` trajectories per task; their mean binary
reward is that task's solve rate ``p̂`` under the *current* checkpoint. Groups
outside the iteration's :class:`~uni_agent.taskpilot.band.TargetBand` are evicted
and replaced by fresh prompts (verl's dynamic-sampling refill path), and when
more in-band groups are available than the batch needs, the ones closest to
``target_resolve_rate`` are selected first. Calibration and training therefore
share one set of rollouts -- no extra generation pass.

Filtering reads the *raw* pass/fail metric (``acc``), so reward shaping such as a
length penalty never moves a group in or out of the band.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import OmegaConf

from uni_agent.taskpilot.band import TargetBand, load_band_schedule, resolve_band
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

logger = logging.getLogger(__name__)

__all__ = ["TaskPilotSampler"]


class TaskPilotSampler(ReplayBufferAsync):
    """Band-filtered replay buffer for the async v1 trainers.

    Args:
        sampler_kwargs: ``trainer.v1.sampler.sampler_kwargs``. Recognized keys:
            ``config_path`` (recipe yaml holding the ``taskpilot`` block),
            ``iteration`` (which band of the schedule applies; default 1),
            ``metric`` (per-trajectory reward key to average; default ``acc``).
        **kwargs: Forwarded verbatim to :class:`ReplayBufferAsync`.
    """

    def __init__(self, *, sampler_kwargs=None, **kwargs) -> None:
        if OmegaConf.is_config(sampler_kwargs):
            options = OmegaConf.to_container(sampler_kwargs, resolve=True)
        else:
            options = dict(sampler_kwargs or {})
        config_path = options.get("config_path")
        taskpilot = load_band_schedule(config_path) if config_path else {}
        iteration = int(options.get("iteration", taskpilot.get("iteration", 1)))
        metric = str(options.get("metric", taskpilot.get("metric", "acc")))

        super().__init__(sampler_kwargs=sampler_kwargs, filter_groups_metric=metric, **kwargs)
        self.band: TargetBand = resolve_band(taskpilot.get("target_band"), iteration)
        # partition_id => {uid: p̂}; mirrors the base cache's lifetime (cleared per group eviction).
        self._resolve_rates: dict[str, dict[str, float]] = defaultdict(dict)
        logger.info("TaskPilot iteration %d: keeping groups with %s in %s", iteration, metric, self.band.describe())

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        """Return the finished groups to evict: those whose ``p̂`` falls outside the band."""
        if partition_id == "val":
            return set(), Counter()

        resolve_rates = self._resolve_rates[partition_id]
        finished_uids = self.finished_keys[partition_id]
        for uid in resolve_rates.keys() - finished_uids:
            del resolve_rates[uid]

        resolve_rates.update(self._solve_rates(partition_id, finished_uids - resolve_rates.keys()))
        rejected = {uid: rate for uid, rate in resolve_rates.items() if not self.band.contains(rate)}
        return set(rejected), Counter(rejected.values())

    def _solve_rates(self, partition_id: str, uids: set[str]) -> dict[str, float]:
        """Mean binary reward per group, read from each trajectory's ``reward_extra_info``."""
        if not uids:
            return {}

        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in uids]
        rewards_by_uid: dict[str, list[float]] = defaultdict(list)
        if trajectory_keys:
            extra_fields_list = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields"],
            )
            for key, extra_fields in zip(trajectory_keys, extra_fields_list, strict=True):
                extra_fields = getattr(extra_fields, "data", extra_fields)
                reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
                if self.filter_groups_metric not in reward_extra_info:
                    raise RuntimeError(
                        f"TaskPilot metric {self.filter_groups_metric!r} missing from group {key.split('_')[0]}; "
                        "the task must report it (e.g. task_runner report_reward=True with TaskResult.accuracy)"
                    )
                rewards_by_uid[key.split("_")[0]].append(float(reward_extra_info[self.filter_groups_metric]))

        missing = uids - rewards_by_uid.keys()
        if missing:
            raise RuntimeError(f"finished groups carry no trajectories: {sorted(missing)[:5]}")
        return {uid: float(np.mean(rewards)) for uid, rewards in rewards_by_uid.items()}

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        """Fill the batch with in-band groups, closest to the target solve rate first.

        Staleness stays the tiebreaker (and the only key for validation or when no
        target is configured), so oldest-prompt-first ordering is preserved.
        """
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])
        resolve_rates = self._resolve_rates[partition_id]
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda key: (
                self.band.distance_to_target(resolve_rates[key]) if key in resolve_rates else 0.0,
                prompt_global_steps_snapshot.get(key, 0),
            ),
        )
        selected = ordered_keys[:batch_size]
        if selected and resolve_rates:
            rates = [resolve_rates[key] for key in selected if key in resolve_rates]
            if rates:
                logger.info(
                    "TaskPilot selected %d/%d groups, solve rate mean=%.3f min=%.3f max=%.3f",
                    len(selected),
                    len(sampleable_keys),
                    float(np.mean(rates)),
                    min(rates),
                    max(rates),
                )
        return selected, partition_snapshot, prompt_global_steps_snapshot

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        super()._clear_groups(partition_id, uids)
        for uid in uids:
            self._resolve_rates[partition_id].pop(uid, None)
