# ruff: noqa: E501
"""Offline calibration: split a candidate task pool by policy-relative solve rate.

Online (per-step) band filtering lives in :mod:`uni_agent.taskpilot.sampler`. This
module is its offline twin, used *between* iterations to keep the pool healthy:
score every candidate with the current checkpoint, keep the tasks inside the
iteration's band, and route the rest to a refine / drop queue.

Calibration rollouts are just inference, so the existing harness produces them::

    # 1. n rollouts per candidate against the checkpoint you are about to train from
    BASE_URL=http://localhost:8000/v1 MODEL=Qwen3-4B \
        python examples/inference/parallel_infer_api.py \
        --data-path candidates.parquet \
        --task-config examples/frognano/task_config_leaf.yaml \
        --n 8 --result-path calibration/iter1.json

    # 2. band split (accepted -> next RL batch, off-band -> refine or drop)
    python -m uni_agent.taskpilot.calibrate \
        --data-path candidates.parquet \
        --results calibration/iter1.json \
        --config examples/frognano/configs/taskpilot.yaml \
        --iteration 1 --out-dir calibration/iter1
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from uni_agent.taskpilot.band import TargetBand, band_for_iteration

logger = logging.getLogger(__name__)

__all__ = ["classify_candidates", "instance_id_of", "solve_rates"]


def instance_id_of(row: dict[str, Any]) -> str:
    """The dataset row's instance id, as reported by the rollout harness."""
    return row["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"]


def solve_rates(results: list[dict[str, Any]]) -> dict[str, float]:
    """Per-instance ``p̂``: the mean of the binary ``resolved`` flags of its rollouts.

    Rollouts that raised (sandbox / eval failures) count as unresolved, matching how
    a failed trajectory scores during training.
    """
    outcomes: dict[str, list[float]] = defaultdict(list)
    for result in results:
        outcomes[result["instance_id"]].append(float(bool(result.get("resolved"))))
    return {instance_id: sum(flags) / len(flags) for instance_id, flags in outcomes.items()}


def classify_candidates(
    rows: list[dict[str, Any]], rates: dict[str, float], band: TargetBand
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (accepted, off-band, uncalibrated) and build the per-row report."""
    accepted, off_band, report = [], [], []
    for row in rows:
        instance_id = instance_id_of(row)
        rate = rates.get(instance_id)
        if rate is None:
            disposition = "uncalibrated"
        elif band.contains(rate):
            disposition = "accepted"
        elif rate <= band.min_exclusive:
            disposition = "too_hard"
        else:
            disposition = "saturated"

        report.append({"instance_id": instance_id, "resolve_rate": rate, "disposition": disposition})
        if disposition == "accepted":
            accepted.append(row)
        elif disposition != "uncalibrated":
            off_band.append(row)
    return accepted, off_band, report


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset

    Dataset.from_list(rows).to_parquet(str(path))
    logger.info("wrote %d tasks to %s", len(rows), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Band-split a candidate task pool by calibrated solve rate.")
    parser.add_argument("--data-path", required=True, help="Candidate pool in Uni-Agent parquet format.")
    parser.add_argument("--results", required=True, help="--result-path JSON produced by parallel_infer_api.py.")
    parser.add_argument("--config", required=True, help="Recipe yaml holding the 'taskpilot' block.")
    parser.add_argument("--iteration", type=int, required=True, help="Which target band of the schedule applies.")
    parser.add_argument("--out-dir", required=True, help="Destination for accepted / off_band / calibration files.")
    args = parser.parse_args()

    from datasets import load_dataset

    band = band_for_iteration(args.config, args.iteration)
    rows = load_dataset("parquet", data_files=args.data_path, split="train").to_list()
    results = json.loads(Path(args.results).expanduser().read_text(encoding="utf-8"))["results"]

    rates = solve_rates(results)
    accepted, off_band, report = classify_candidates(rows, rates, band)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in report), encoding="utf-8")
    if accepted:
        _write_parquet(accepted, out_dir / "accepted.parquet")
    if off_band:
        _write_parquet(off_band, out_dir / "off_band.parquet")

    counts: dict[str, int] = defaultdict(int)
    for entry in report:
        counts[entry["disposition"]] += 1
    logger.info(
        "iteration %d band %s over %d candidates: %s",
        args.iteration,
        band.describe(),
        len(rows),
        dict(sorted(counts.items())),
    )
    if not accepted:
        raise SystemExit("no candidate landed in the target band; refine the pool or widen the band")


if __name__ == "__main__":
    main()
