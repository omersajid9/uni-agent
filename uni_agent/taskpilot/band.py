"""Target bands: which empirical solve rates count as *trainable* this iteration.

A band is a policy-relative learnability window over a task's rollout solve rate
``p̂ = (1/n) Σ R_j`` (``R_j ∈ {0, 1}``). Groups below the band are too hard, groups
above it are saturated; only the ones inside carry gradient signal, so only they
are worth spending an update on.

The schedule is data, not code: one entry per training iteration, keyed by
``iteration_<k>`` or ``iterations_<a>_to_<b>`` (plus an optional ``default``)::

    taskpilot:
      metric: acc
      iteration: 1
      target_band:
        iterations_1_to_4:
          target_resolve_rate: 0.5
        iteration_5:
          min_exclusive: 0.0
          max_inclusive: 0.5

``target_resolve_rate`` is a *soft* preference (keep the whole learnable region,
but fill the batch with the groups closest to the target);
``min_exclusive`` / ``max_inclusive`` are a *hard* filter (anything outside is
dropped and replaced). The two compose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["TargetBand", "band_for_iteration", "load_band_schedule", "resolve_band"]

_ITERATION_KEY = re.compile(r"^iterations?_(\d+)(?:_to_(\d+))?$")


@dataclass(frozen=True)
class TargetBand:
    """A learnability window over the group solve rate ``p̂``.

    Args:
        min_exclusive: Lower bound, exclusive. ``0.0`` drops groups nothing solved.
        max_inclusive: Upper bound, inclusive. ``None`` means "strictly below 1.0",
            i.e. the full learnable region ``0 < p̂ < 1``.
        target_resolve_rate: Preferred solve rate inside the band, used to rank
            groups when more are available than the batch needs.
    """

    min_exclusive: float = 0.0
    max_inclusive: float | None = None
    target_resolve_rate: float | None = None

    def __post_init__(self) -> None:
        upper = 1.0 if self.max_inclusive is None else self.max_inclusive
        if not 0.0 <= self.min_exclusive < upper <= 1.0:
            raise ValueError(f"target band requires 0 <= min_exclusive < max_inclusive <= 1, got {self}")
        if self.target_resolve_rate is not None and not self.contains(self.target_resolve_rate):
            raise ValueError(f"target_resolve_rate={self.target_resolve_rate} lies outside its own band {self}")

    def contains(self, resolve_rate: float) -> bool:
        """Whether a group with this solve rate is trainable under the band."""
        below_upper = resolve_rate <= self.max_inclusive if self.max_inclusive is not None else resolve_rate < 1.0
        return resolve_rate > self.min_exclusive and below_upper

    def distance_to_target(self, resolve_rate: float) -> float:
        """Ranking key inside the band; ``0.0`` for every group when no target is set."""
        if self.target_resolve_rate is None:
            return 0.0
        return abs(resolve_rate - self.target_resolve_rate)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> TargetBand:
        unknown = set(mapping) - {"min_exclusive", "max_inclusive", "target_resolve_rate"}
        if unknown:
            raise ValueError(
                f"unknown target-band keys {sorted(unknown)}; "
                "expected min_exclusive / max_inclusive / target_resolve_rate"
            )
        return cls(
            min_exclusive=float(mapping.get("min_exclusive", 0.0)),
            max_inclusive=None if mapping.get("max_inclusive") is None else float(mapping["max_inclusive"]),
            target_resolve_rate=(
                None if mapping.get("target_resolve_rate") is None else float(mapping["target_resolve_rate"])
            ),
        )

    def describe(self) -> str:
        upper = "1.0)" if self.max_inclusive is None else f"{self.max_inclusive}]"
        target = "" if self.target_resolve_rate is None else f" target={self.target_resolve_rate}"
        return f"({self.min_exclusive}, {upper}{target}"


def load_band_schedule(config_path: str | Path) -> dict[str, Any]:
    """Read the ``taskpilot`` block of a recipe config file."""
    config = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8")) or {}
    taskpilot = config.get("taskpilot", config)
    if not isinstance(taskpilot, dict):
        raise ValueError(f"{config_path}: expected a mapping under 'taskpilot'")
    return taskpilot


def resolve_band(target_band: dict[str, Any] | None, iteration: int) -> TargetBand:
    """Select the band that applies to ``iteration`` from a ``target_band`` mapping.

    An empty / missing mapping yields the default band -- the full learnable
    region ``0 < p̂ < 1``, which is what plain dynamic sampling already does.
    """
    if not target_band:
        return TargetBand()

    default: dict[str, Any] | None = None
    for key, spec in target_band.items():
        if key == "default":
            default = spec
            continue
        match = _ITERATION_KEY.match(str(key))
        if match is None:
            raise ValueError(
                f"unparsable target_band key {key!r}; use 'iteration_<k>', 'iterations_<a>_to_<b>', or 'default'"
            )
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        if low <= iteration <= high:
            return TargetBand.from_mapping(spec or {})

    if default is None:
        raise ValueError(f"no target_band entry covers iteration {iteration} and no 'default' entry is set")
    return TargetBand.from_mapping(default or {})


def band_for_iteration(config_path: str | Path, iteration: int) -> TargetBand:
    """Convenience wrapper: load a recipe config and resolve its band for ``iteration``."""
    return resolve_band(load_band_schedule(config_path).get("target_band"), iteration)
