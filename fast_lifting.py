# Modified from the original codebase of:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This file is part of CF3.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from scene import Scene
import random
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_feature, render_collect_features
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import numpy as np
import torch.nn as nn
import time

def render_save_features(model_path: str, name: str, iteration: int, views: list, gaussians: GaussianModel, pipeline: PipelineParams, background, frames=None):

    elapsed_time = 0
    count = 0

    feature_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_vis")
    rendered_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    makedirs(feature_vis_path, exist_ok=True)
    makedirs(rendered_feature_path, exist_ok=True)

    with torch.no_grad():

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if frames is not None and view.image_name not in frames:
                continue
            count += 1
            start_time = time.time()

            # original feature rendering
            render_pkg = render_feature(view, gaussians, pipeline, background)
            feature_map = render_pkg["feature_map"]
            torch.cuda.synchronize()
            elapsed_time += time.time() - start_time

            gt_feature_map = view.semantic_feature.cuda()

            render_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            C, H, W = feature_map.shape
            if feature_map.shape != gt_feature_map.shape:
                gt_feature_map = nn.functional.interpolate(gt_feature_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

            # save feature map
            torch.save(feature_map.half(), os.path.join(rendered_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

            # save pca visualization
            fmap = nn.functional.normalize(feature_map[None, ...], dim=1)
            gt_fmap = nn.functional.normalize(gt_feature_map[None, ...], dim=1)

            # change nan to 0
            fmap[torch.isnan(fmap)] = 0.0
            gt_fmap[torch.isnan(gt_fmap)] = 0.0

            # concatenate feature map
            concat_fmap = torch.cat([gt_fmap, fmap], dim=3)    # (1, C, H, 3*W)
            pca = sklearn.decomposition.PCA(3, random_state=42)
            feature_concat = concat_fmap.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy()

            pca.fit(gt_fmap.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy())
            transformed = pca.transform(feature_concat)

            feature_pca_mean = torch.tensor(feature_concat.mean(0)).float().cuda()
            feature_pca_components = torch.tensor(pca.components_).float().cuda()

            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            del feature_concat
            
            vis_feature = (concat_fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T   # (H * 3*W, 3)
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((concat_fmap.shape[2], concat_fmap.shape[3], 3)).cpu()    # (H, 3*W, 3)

            gt_feature_vis = vis_feature[:, :W, :]
            feature_vis = vis_feature[:, W:, :]
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(gt_feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title("Ground Truth Feature")

            plt.subplot(1, 2, 2)
            plt.imshow(feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title(f"Rendered FeatureGS")

            plt.suptitle(f"View {idx:05d} Feature Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(feature_vis_path, f'{idx:05d}_feature_vis_concat_sub.png'))
            plt.close()

        fps = count / elapsed_time
        print(f"Rendering time: {elapsed_time:.2f}s, FPS: {fps:.2f}")

    return fps

def set_features(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, iteration : int, skip_train : bool, skip_test : bool, output_path : str):
    
    semantic_feature_size = None
    if dataset.foundation_model == "lseg":
        semantic_feature_size = 512
    elif dataset.foundation_model == "maskclip":
        semantic_feature_size = 512
    elif dataset.foundation_model == "clip":
        semantic_feature_size = 768
    elif dataset.foundation_model == "dino":
        semantic_feature_size = 384
    elif dataset.foundation_model == "langsplat":
        semantic_feature_size = 512

    gaussians = GaussianModel(dataset.sh_degree, semantic_feature_size)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    makedirs(output_path, exist_ok=True)
    print("Output path: ", output_path)
    os.system(f"cp -r {dataset.model_path}/cfg_args {output_path}")
    os.system(f"cp -r {dataset.model_path}/cameras.json {output_path}")
    os.system(f"cp -r {dataset.model_path}/input.ply {output_path}")

    viewpoint_stack = scene.getTrainCameras().copy()
    
    gaussians.lift_feature_setup()

    accum_features = torch.zeros((gaussians.get_num_gaussians, 1, gaussians.semantic_feature_size), dtype=torch.float, device="cuda").requires_grad_(False)
    accum_features_squared = torch.zeros((gaussians.get_num_gaussians, 1, gaussians.semantic_feature_size), dtype=torch.float, device="cuda").requires_grad_(False)
    accum_contribution = torch.zeros((gaussians.get_num_gaussians), dtype=torch.float, device="cuda").requires_grad_(False)

    with torch.no_grad():
        for view in tqdm(viewpoint_stack):
            gt_feature_map = view.semantic_feature.cuda().float()
            gt_feature_map = torch.nn.functional.normalize(gt_feature_map, dim=0)
            gt_feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(view.image_height, view.image_width), mode='bilinear', align_corners=False).squeeze(0)

            render_pkg = render_collect_features(view, gaussians, pipe, background, gt_feature_map)

            accum_features += render_pkg["accum_features"]
            accum_features_squared += render_pkg["accum_features_square"]
            accum_contribution += render_pkg["importance_score"]

        gaussians.set_semantic_feature(accum_contribution, accum_features, accum_features_squared)

        gaussians._semantic_feature = torch.nn.functional.normalize(gaussians._semantic_feature, dim=2)

    with torch.no_grad():
        # save the model
        point_cloud_path = os.path.join(output_path, f"point_cloud/iteration_{scene.loaded_iter}")
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # render the point cloud
        if not skip_test:
            render_save_features(output_path, f"test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, background)
        if not skip_train:
            render_save_features(output_path, f"train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, background)

        # number of gaussians
        with open(os.path.join(point_cloud_path, "num_gaussians.txt"), 'w') as f:
            f.write("number of gaussians: " + str(gaussians.get_num_gaussians) + "\n" + "number of light semantic gaussians: " + str(gaussians.compact_feature_field.get_num_gaussians))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", "-o", default="", type=str)
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    print("smoothing: ", args.smoothing)

    if args.output == "":
        args.output = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)
    set_features(model.extract(args), op.extract(args), pipeline.extract(args), args.iteration, args.skip_train, args.skip_test, args.output) ###

    print("Done")