#!/usr/bin/env python
"""Collect per-scene LERF-OVS metric TSVs into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CF3 LERF-OVS scene metrics.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[0] if rows else None


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_path = Path(args.output) if args.output else run_root / "metrics.tsv"
    rows = []
    fieldnames = [
        "scene",
        "mIoU",
        "mAcc",
        "num_frames",
        "num_queries",
        "mask_thresh",
        "iteration",
        "model_path",
        "elapsed_sec",
        "status",
    ]

    for scene in args.scenes:
        metric_path = run_root / scene / "eval_lerf_ovs" / "metrics.tsv"
        row = read_metric(metric_path)
        if row is None:
            row = {
                "scene": scene,
                "mIoU": "nan",
                "mAcc": "nan",
                "num_frames": "0",
                "num_queries": "0",
                "mask_thresh": "",
                "iteration": "",
                "model_path": str(run_root / scene),
                "elapsed_sec": "",
                "status": "missing",
            }
        else:
            row["status"] = "ok"
        rows.append({key: row.get(key, "") for key in fieldnames})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    per_object_path = run_root / "per_object_metrics.tsv"
    per_object_fields = [
        "scene",
        "frame",
        "category",
        "iou",
        "loc_acc",
        "pred_px",
        "gt_px",
        "max_relevance",
        "mean_relevance",
        "status",
    ]
    with per_object_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_object_fields, delimiter="\t")
        writer.writeheader()
        for scene in args.scenes:
            scene_path = run_root / scene / "eval_lerf_ovs" / "per_object_metrics.tsv"
            if not scene_path.exists():
                continue
            with scene_path.open("r", newline="") as scene_file:
                reader = csv.DictReader(scene_file, delimiter="\t")
                for row in reader:
                    writer.writerow({key: row.get(key, "") for key in per_object_fields})

    print(f"Wrote {output_path}")
    print(f"Wrote {per_object_path}")


if __name__ == "__main__":
    main()
