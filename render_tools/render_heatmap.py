#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_feature, render_collect_features, render_gof
from gaussian_renderer import GaussianModel
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from utils.clip_utils import CLIPEditor
import time
import yaml
from models.networks import CNN_decoder, MLP_encoder

def feature_visualize_saving(feature, gt_feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    gt_fmap = gt_feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    gt_fmap = nn.functional.normalize(gt_fmap, dim=1)

    # concatenate feature map
    concat_fmap = torch.cat([fmap, gt_fmap], dim=3)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    gt_samples = gt_fmap.permute(0, 2, 3, 1).reshape(-1, gt_fmap.shape[1])[::3].cpu().numpy()
    # transformed = pca.fit_transform(f_samples)
    pca.fit(gt_samples)

    f_samples = concat_fmap.permute(0, 2, 3, 1).reshape(-1, gt_fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.transform(f_samples)

    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()

    q1, q99 = np.percentile(transformed, [0, 100])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    
    vis_feature = (concat_fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((concat_fmap.shape[2], concat_fmap.shape[3], 3)).cpu()
    return vis_feature


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, novel_view : bool, 
                video : bool , edit_config: str, novel_video : bool, multi_interpolate : bool, num_views : int): 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 512)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        features_dc = gaussians.compact_feature_field._features_dc.clone().detach().squeeze(1)
        features = gaussians.compact_feature_field.autoencoder.decoder(features_dc) # torch.Size([N, 512])

        


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true") ###
    parser.add_argument("--novel_video", action="store_true") ###
    parser.add_argument('--edit_config', default="no editing", type=str)
    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--num_views", default=200, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.novel_view, 
                args.video, args.edit_config, args.novel_video, args.multi_interpolate, args.num_views) ###