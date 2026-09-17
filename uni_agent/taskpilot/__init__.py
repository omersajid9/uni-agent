"""TaskPilot: policy-relative task selection (target bands over rollout solve rates).

* :mod:`.band` -- the target-band schedule (one learnability window per iteration).
* :mod:`.sampler` -- online per-step filtering, wired in as verl's custom sampler.
* :mod:`.calibrate` -- offline band split of a candidate pool between iterations.

``sampler`` imports verl, so it is not re-exported here: ``band`` and ``calibrate``
stay importable in verl-free contexts (data prep, tests).
"""

from uni_agent.taskpilot.band import TargetBand, band_for_iteration, load_band_schedule, resolve_band

__all__ = ["TargetBand", "band_for_iteration", "load_band_schedule", "resolve_band"]
