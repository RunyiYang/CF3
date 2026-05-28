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
import cv2
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
    parser.add_argument(
        "--mask_protocol",
        choices=("raw", "lerf"),
        default="lerf",
        help="raw thresholds relevance directly; lerf matches LangSplat/LERF smoothing and truncation.",
    )
    parser.add_argument(
        "--loc_metric",
        choices=("mask", "bbox"),
        default="bbox",
        help="bbox matches the LangSplat/LERF localization protocol.",
    )
    parser.add_argument(
        "--normalize_decoded_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize decoded 512D features before OpenCLIP relevance.",
    )
    parser.add_argument("--eval_subdir", type=str, default="eval_lerf_ovs")
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


def object_bboxes(objects: list[dict]) -> np.ndarray:
    boxes = []
    for obj in objects:
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_arr.size == 4:
            boxes.append(bbox_arr)
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack(boxes, axis=0)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def resize_bboxes(bboxes: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    if bboxes.size == 0:
        return bboxes
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    resized = bboxes.copy()
    resized[:, [0, 2]] *= scale_x
    resized[:, [1, 3]] *= scale_y
    return resized


def load_annotations(gt_folder: str) -> list[tuple[str, str, int, int, np.ndarray, np.ndarray]]:
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
            bboxes = object_bboxes(objects)
            annotations.append((image_stem, category, width, height, gt_mask, bboxes))
    return annotations


@torch.no_grad()
def render_feature_map(
    view,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    normalize_decoded_features: bool,
) -> torch.Tensor:
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
    if normalize_decoded_features:
        feature_map = F.normalize(feature_map, p=2, dim=0)
    feature_map[torch.isnan(feature_map)] = 0.0
    return feature_map


def average_filter_relevance(relevance_map: torch.Tensor, scale: int = 30) -> torch.Tensor:
    np_relev = relevance_map.detach().cpu().numpy()
    kernel = np.ones((scale, scale), dtype=np.float32) / float(scale**2)
    avg_filtered = cv2.filter2D(np_relev, -1, kernel)
    return torch.from_numpy(avg_filtered).to(relevance_map.device)


def lerf_protocol_mask(relevance_map: torch.Tensor, mask_thresh: float) -> np.ndarray:
    avg_filtered = average_filter_relevance(relevance_map)
    filtered_map = 0.5 * (avg_filtered + relevance_map)

    output = filtered_map - torch.min(filtered_map)
    output = output / (torch.max(output) + 1e-9)
    output = output * 2.0 - 1.0
    output = torch.clip(output, 0, 1)

    pred_mask = (output.detach().cpu().numpy() > mask_thresh).astype(np.uint8)
    return smooth_mask(pred_mask).astype(bool)


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    smoothed = mask.copy()
    scale = 3
    for i in range(h):
        y0 = max(0, i - scale)
        y1 = min(i + scale + 1, h - 1)
        for j in range(w):
            x0 = max(0, j - scale)
            x1 = min(j + scale + 1, w - 1)
            square = mask[y0:y1, x0:x1]
            smoothed[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return smoothed


def locate_peaks(relevance_map: torch.Tensor, protocol: str) -> np.ndarray:
    if protocol == "lerf":
        loc_map = average_filter_relevance(relevance_map)
    else:
        loc_map = relevance_map
    loc_np = loc_map.detach().cpu().numpy()
    max_value = loc_np.max()
    ys, xs = np.nonzero(loc_np == max_value)
    return np.stack([xs, ys], axis=1)


def any_point_in_bboxes(points_xy: np.ndarray, bboxes: np.ndarray) -> bool:
    for x, y in points_xy:
        for x1, y1, x2, y2 in bboxes.reshape(-1, 4):
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
    return False


def any_point_in_mask(points_xy: np.ndarray, mask: np.ndarray) -> bool:
    height, width = mask.shape
    for x, y in points_xy:
        if 0 <= x < width and 0 <= y < height and mask[y, x]:
            return True
    return False


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
    eval_dir = Path(dataset.model_path) / args.eval_subdir
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

    for image_stem, category, gt_width, gt_height, gt_mask, gt_bboxes in tqdm(annotations, desc=f"Eval {args.scene}"):
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
                    "peak_x": "nan",
                    "peak_y": "nan",
                    "max_relevance": "nan",
                    "mean_relevance": "nan",
                    "status": "missing_view",
                }
            )
            continue

        if image_stem not in render_cache:
            render_cache[image_stem] = render_feature_map(
                view,
                gaussians,
                args.pipeline,
                background,
                normalize_decoded_features=args.normalize_decoded_features,
            )
        feature_map = render_cache[image_stem]
        _, height, width = feature_map.shape
        gt_mask_eval = resize_mask(gt_mask, height, width)
        gt_bboxes_eval = resize_bboxes(gt_bboxes, gt_width, gt_height, width, height)

        clip_model.set_positives([category])
        sem_map = feature_map.permute(1, 2, 0).unsqueeze(0)
        relevance_map = clip_model.get_max_across(sem_map)[0, 0].detach().float()
        if args.mask_protocol == "lerf":
            pred_mask = lerf_protocol_mask(relevance_map, args.mask_thresh)
        else:
            pred_mask = (relevance_map > args.mask_thresh).cpu().numpy().astype(bool)

        intersection = np.logical_and(pred_mask, gt_mask_eval).sum()
        union = np.logical_or(pred_mask, gt_mask_eval).sum()
        iou = float(intersection / union) if union > 0 else 0.0

        peak_points = locate_peaks(relevance_map, args.mask_protocol)
        max_x, max_y = peak_points[0]
        if args.loc_metric == "bbox":
            loc_acc = any_point_in_bboxes(peak_points, gt_bboxes_eval)
        else:
            loc_acc = any_point_in_mask(peak_points, gt_mask_eval)

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
            "peak_x": max_x,
            "peak_y": max_y,
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
        "mask_protocol": args.mask_protocol,
        "loc_metric": args.loc_metric,
        "normalize_decoded_features": args.normalize_decoded_features,
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

    eval_dir = Path(args.dataset.model_path) / args.eval_subdir
    metrics_path = Path(args.metrics_tsv) if args.metrics_tsv else eval_dir / "metrics.tsv"
    per_object_path = Path(args.per_object_tsv) if args.per_object_tsv else eval_dir / "per_object_metrics.tsv"

    metric_fields = [
        "scene",
        "mIoU",
        "mAcc",
        "num_frames",
        "num_queries",
        "mask_thresh",
        "mask_protocol",
        "loc_metric",
        "normalize_decoded_features",
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
            "peak_x",
            "peak_y",
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
