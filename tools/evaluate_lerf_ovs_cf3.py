#!/usr/bin/env python
"""Evaluate CF3 LERF-OVS masks with mIoU and localization accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from openclip_encoder import OpenCLIPNetwork  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CF3 on LERF-OVS labels.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--metrics_tsv", type=str, default="")
    parser.add_argument("--per_object_tsv", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--save_relevance", action="store_true")
    args = parser.parse_args()
    args.dataset = model.extract(args)
    args.pipeline = pipeline.extract(args)
    return args


def segmentation_to_polygons(segmentation) -> list[list[tuple[float, float]]]:
    if not segmentation:
        return []
    if isinstance(segmentation[0], (int, float)):
        coords = segmentation
        return [[(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]]
    if isinstance(segmentation[0], list) and segmentation and segmentation[0]:
        if isinstance(segmentation[0][0], (int, float)):
            if len(segmentation[0]) == 2:
                return [[(float(x), float(y)) for x, y in segmentation]]
            polygons = []
            for coords in segmentation:
                polygons.append(
                    [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
                )
            return polygons
    return []


def polygon_mask(width: int, height: int, objects: list[dict]) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask_image)
    for obj in objects:
        for polygon in segmentation_to_polygons(obj.get("segmentation", [])):
            if len(polygon) >= 3:
                drawer.polygon(polygon, outline=1, fill=1)
    return np.asarray(mask_image, dtype=bool)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def load_annotations(gt_folder: str) -> list[tuple[str, str, int, int, np.ndarray]]:
    label_paths = sorted(Path(gt_folder).glob("*.json"))
    annotations = []
    for label_path in label_paths:
        with label_path.open("r") as f:
            label = json.load(f)

        info = label.get("info", {})
        image_stem = Path(info.get("name") or label_path.stem).stem
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        by_category: dict[str, list[dict]] = defaultdict(list)
        for obj in label.get("objects", []):
            category = obj.get("category")
            if category:
                by_category[str(category)].append(obj)

        for category, objects in sorted(by_category.items()):
            gt_mask = polygon_mask(width, height, objects)
            annotations.append((image_stem, category, width, height, gt_mask))
    return annotations


@torch.no_grad()
def render_feature_map(view, gaussians: GaussianModel, pipeline, background: torch.Tensor) -> torch.Tensor:
    gaussians.compact_feature_field.normalize_features()
    latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
    latent_map = render(
        view,
        gaussians.compact_feature_field,
        pipeline,
        background,
        override_color=latent_features,
    )["render"]
    feature_map = gaussians.compact_feature_field.decode_featuremap(latent_map).float()
    feature_map = F.normalize(feature_map, p=2, dim=0)
    feature_map[torch.isnan(feature_map)] = 0.0
    return feature_map


def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    safe_state(args.quiet)
    annotations = load_annotations(args.gt_folder)
    if not annotations:
        raise RuntimeError(f"No label JSON files found in {args.gt_folder}")

    dataset = args.dataset
    dataset.eval = False
    dataset.resolution = -1
    dataset.foundation_model = dataset.foundation_model or "langsplat"

    gaussians = GaussianModel(dataset.sh_degree, 512)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = {view.image_name: view for view in scene.getTrainCameras() + scene.getTestCameras()}

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    device = torch.device("cuda")
    clip_model = OpenCLIPNetwork(device)
    eval_dir = Path(dataset.model_path) / "eval_lerf_ovs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (eval_dir / "masks").mkdir(exist_ok=True)
    if args.save_relevance:
        (eval_dir / "relevance").mkdir(exist_ok=True)

    per_object_rows = []
    ious = []
    loc_hits = []
    render_cache: dict[str, torch.Tensor] = {}
    start = perf_counter()

    for image_stem, category, gt_width, gt_height, gt_mask in tqdm(annotations, desc=f"Eval {args.scene}"):
        view = views.get(image_stem)
        if view is None:
            per_object_rows.append(
                {
                    "scene": args.scene,
                    "frame": image_stem,
                    "category": category,
                    "iou": "nan",
                    "loc_acc": "nan",
                    "pred_px": "nan",
                    "gt_px": int(gt_mask.sum()),
                    "max_relevance": "nan",
                    "mean_relevance": "nan",
                    "status": "missing_view",
                }
            )
            continue

        if image_stem not in render_cache:
            render_cache[image_stem] = render_feature_map(view, gaussians, args.pipeline, background)
        feature_map = render_cache[image_stem]
        _, height, width = feature_map.shape
        gt_mask_eval = resize_mask(gt_mask, height, width)

        clip_model.set_positives([category])
        sem_map = feature_map.permute(1, 2, 0).unsqueeze(0)
        relevance_map = clip_model.get_max_across(sem_map)[0, 0].detach().float()
        pred_mask = (relevance_map > args.mask_thresh).cpu().numpy().astype(bool)

        intersection = np.logical_and(pred_mask, gt_mask_eval).sum()
        union = np.logical_or(pred_mask, gt_mask_eval).sum()
        iou = float(intersection / union) if union > 0 else 0.0

        max_index = int(torch.argmax(relevance_map).item())
        max_y, max_x = divmod(max_index, width)
        loc_acc = bool(gt_mask_eval[max_y, max_x])

        ious.append(iou)
        loc_hits.append(float(loc_acc))
        row = {
            "scene": args.scene,
            "frame": image_stem,
            "category": category,
            "iou": f"{iou:.6f}",
            "loc_acc": int(loc_acc),
            "pred_px": int(pred_mask.sum()),
            "gt_px": int(gt_mask_eval.sum()),
            "max_relevance": f"{float(relevance_map.max().item()):.6f}",
            "mean_relevance": f"{float(relevance_map.mean().item()):.6f}",
            "status": "ok",
        }
        per_object_rows.append(row)

        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{image_stem}_{category}")
        if args.save_masks:
            Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L").save(
                eval_dir / "masks" / f"{safe_name}.png"
            )
        if args.save_relevance:
            np.save(eval_dir / "relevance" / f"{safe_name}.npy", relevance_map.cpu().numpy())

    elapsed = perf_counter() - start
    metrics = {
        "scene": args.scene,
        "mIoU": float(np.mean(ious)) if ious else float("nan"),
        "mAcc": float(np.mean(loc_hits)) if loc_hits else float("nan"),
        "num_frames": len({row["frame"] for row in per_object_rows if row["status"] == "ok"}),
        "num_queries": len(ious),
        "mask_thresh": args.mask_thresh,
        "iteration": args.iteration,
        "model_path": dataset.model_path,
        "elapsed_sec": elapsed,
    }

    return metrics, per_object_rows


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metrics, per_object_rows = evaluate(args)

    eval_dir = Path(args.dataset.model_path) / "eval_lerf_ovs"
    metrics_path = Path(args.metrics_tsv) if args.metrics_tsv else eval_dir / "metrics.tsv"
    per_object_path = Path(args.per_object_tsv) if args.per_object_tsv else eval_dir / "per_object_metrics.tsv"

    metric_fields = [
        "scene",
        "mIoU",
        "mAcc",
        "num_frames",
        "num_queries",
        "mask_thresh",
        "iteration",
        "model_path",
        "elapsed_sec",
    ]
    write_tsv(
        metrics_path,
        [
            {
                **metrics,
                "mIoU": f"{metrics['mIoU']:.6f}",
                "mAcc": f"{metrics['mAcc']:.6f}",
                "elapsed_sec": f"{metrics['elapsed_sec']:.2f}",
            }
        ],
        metric_fields,
    )
    write_tsv(
        per_object_path,
        per_object_rows,
        [
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
        ],
    )
    with (eval_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"{metrics['scene']}: mIoU={metrics['mIoU']:.4f} "
        f"mAcc={metrics['mAcc']:.4f} queries={metrics['num_queries']}"
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
