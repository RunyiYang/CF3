# Modified from the original codebase of:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This file is part of CF3.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from scene import Scene
import random
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_feature, render_collect_features
import torchvision
from utils.general_utils import safe_state, build_rotation
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene.gaussian_model import GaussianModel, CompactFeatureField
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
from utils.loss_utils import l1_loss, ssim, tv_loss, custom_loss, custom_loss2, l2_loss
from simple_knn._C import knnGraphCUDA, mergeKNN
from random import randint
# Uncomment if you wish to use TensorBoard
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ExponentialLR

def collect_fps_importance(model_path: str, name: str, iteration: int, views: list, gaussians: GaussianModel, pipeline: PipelineParams, background):

    # measure time
    lightgs_time = 0
    featuregs_time = 0

    feature_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_vis")
    makedirs(feature_vis_path, exist_ok=True)

    # measure time
    with torch.no_grad():
        gaussians.compact_feature_field.normalize_features()
        importance_score = torch.ones(gaussians.compact_feature_field.get_num_gaussians, device="cuda")
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if idx == 0:
                continue
            
            # read time
            lightgs_start = time.time()
            latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
            render_pkg_light = render(view, gaussians.compact_feature_field, pipeline, background, override_color=latent_features)
            cf3_feature_map = gaussians.compact_feature_field.decode_featuremap(render_pkg_light["render"])
            torch.cuda.synchronize()
            lightgs_time += time.time() - lightgs_start
            importance_score += render_pkg_light["importance_score"]

            # original feature rendering
            featuregs_start = time.time()
            render_pkg = render_feature(view, gaussians, pipeline, background)
            torch.cuda.synchronize()
            featuregs_time += time.time() - featuregs_start

    light_fps = (len(views)-1) / lightgs_time
    feature_fps = (len(views)-1) / featuregs_time

    return light_fps, feature_fps, importance_score


def render_save_features(model_path: str, name: str, iteration: int, views: list, gaussians: GaussianModel, pipeline: PipelineParams, background, selected_frames=None):

    cossim_sum = 0
    l1_sum = 0

    feature_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_vis")
    cf3_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    makedirs(feature_vis_path, exist_ok=True)
    makedirs(cf3_feature_map_path, exist_ok=True)

    with torch.no_grad():
        latent_features = gaussians.compact_feature_field.normalize_features()

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            if selected_frames is not None and view.image_name not in selected_frames:
                continue

            latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
            latent_feature_map = render(view, gaussians.compact_feature_field, pipeline, background, override_color=latent_features)["render"]
            cf3_feature_map = gaussians.compact_feature_field.decode_featuremap(latent_feature_map).half()
            
            # original feature rendering
            render_pkg = render_feature(view, gaussians, pipeline, background)
            feature_map = render_pkg["feature_map"].half()
            gt_feature_map = view.semantic_feature.cuda().half()

            render_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            # save feature map
            torch.save(cf3_feature_map.half(), os.path.join(cf3_feature_map_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

            # For visualization, downsample to 1/4 resolution
            C, H, W = cf3_feature_map.shape
            H = H // 4
            W = W // 4

            if cf3_feature_map.shape != gt_feature_map.shape:
                gt_feature_map = nn.functional.interpolate(gt_feature_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
                cf3_feature_map = nn.functional.interpolate(cf3_feature_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
                feature_map = nn.functional.interpolate(feature_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

            # compute similarity
            cos_sim_light = F.cosine_similarity(cf3_feature_map.unsqueeze(0), gt_feature_map.unsqueeze(0), dim=1)
            cos_sim = F.cosine_similarity(feature_map.unsqueeze(0), gt_feature_map.unsqueeze(0), dim=1)

            # compute l1 loss
            l1_loss_light = l1_loss(cf3_feature_map.unsqueeze(0), gt_feature_map.unsqueeze(0))
            l1_loss_feature = l1_loss(feature_map.unsqueeze(0), gt_feature_map.unsqueeze(0))

            # save pca visualization
            fmap_light = nn.functional.normalize(cf3_feature_map[None, ...], dim=1).detach().cpu()
            fmap = nn.functional.normalize(feature_map[None, ...], dim=1).detach().cpu()
            gt_fmap = nn.functional.normalize(gt_feature_map[None, ...], dim=1).detach().cpu()

            # change nan to 0
            fmap[torch.isnan(fmap)] = 0.0
            fmap_light[torch.isnan(fmap_light)] = 0.0
            gt_fmap[torch.isnan(gt_fmap)] = 0.0

            # concatenate feature map
            concat_fmap = torch.cat([gt_fmap, fmap, fmap_light], dim=3)    # (1, C, H, 3*W)
            pca = sklearn.decomposition.PCA(3, random_state=42)
            feature_concat = concat_fmap.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy()

            pca.fit(gt_fmap.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy())
            transformed = pca.transform(feature_concat)

            feature_pca_mean = torch.tensor(feature_concat.mean(0)).half()
            feature_pca_components = torch.tensor(pca.components_).half()

            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            del feature_concat
            
            vis_feature = (concat_fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T   # (H * 3*W, 3)
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((concat_fmap.shape[2], concat_fmap.shape[3], 3)).cpu()    # (H, 3*W, 3)

            gt_feature_vis = vis_feature[:, :W, :]
            feature_vis = vis_feature[:, W:2*W, :]
            light_feature_vis = vis_feature[:, 2*W:, :]
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(gt_feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title("Ground Truth Feature")

            plt.subplot(1, 3, 2)
            plt.imshow(feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title(f"Rendered FeatureGS\ncossim: {cos_sim.mean().item():.3f} l1: {l1_loss_feature.item():.3f}")

            plt.subplot(1, 3, 3)
            plt.imshow(light_feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title(f"Light FeatureGS\ncossim: {cos_sim_light.mean().item():.3f} l1: {l1_loss_light.item():.3f}")

            plt.suptitle(f"View {idx:05d} Feature Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(feature_vis_path, f'{idx:05d}_feature_vis_concat_sub.png'))
            plt.close()

            # save pca visualization for each feature map
            gt_feature_vis = gt_feature_vis.permute(2, 0, 1)        # (3, H, W)
            feature_vis = feature_vis.permute(2, 0, 1)              # (3, H, W)
            light_feature_vis = light_feature_vis.permute(2, 0, 1)    # (3, H, W)

            # Save each visualization using torchvision
            gt_save_path = os.path.join(feature_vis_path, f'{idx:05d}_gt_feature_vis.png')
            torchvision.utils.save_image(gt_feature_vis, gt_save_path)

            rendered_save_path = os.path.join(feature_vis_path, f'{idx:05d}_rendered_feature_vis.png')
            torchvision.utils.save_image(feature_vis, rendered_save_path)

            light_save_path = os.path.join(feature_vis_path, f'{idx:05d}_light_feature_vis.png')
            torchvision.utils.save_image(light_feature_vis, light_save_path)
        
            cossim_sum += cos_sim.mean().item()
            l1_sum += l1_loss_feature.item()


def ensure_rotation_matrix(V):
    """
    Ensure that an orthogonal matrix V is a proper rotation matrix.
    V should be of shape (n, 3, 3).
    """
    dets = torch.det(V)  # Shape: (n,)
    mask = dets < 0
    V[mask, :, -1] = -V[mask, :, -1]

    return V

def rotation_matrix_to_quaternion(R):
    """
    Convert a batch of rotation matrices to quaternions in PyTorch.
    
    Args:
        R (torch.Tensor): Tensor of shape (N, 3, 3) containing N rotation matrices.
    
    Returns:
        torch.Tensor: Tensor of shape (N, 4) with quaternions [w, x, y, z].
    """
    R = R.float()
    N = R.shape[0]
    
    # Compute the trace of each rotation matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Compute the four values corresponding to w^2, x^2, y^2, z^2 (scaled by 4)
    s = torch.stack([
        trace + 1.0,                                 # Related to w
        1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],  # Related to x
        1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2],  # Related to y
        1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]   # Related to z
    ], dim=1)  # Shape: (N, 4)
    
    # Ensure non-negative values for square root (should be the case for valid rotation matrices)
    s = torch.clamp(s, min=0.0)
    
    # Find the index of the largest value for each matrix
    idx = torch.argmax(s, dim=1)  # Shape: (N,)
    
    # Case 0: w is the largest component
    w0 = torch.sqrt(s[:, 0]) / 2.0
    x0 = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w0.clamp(min=1e-8))
    y0 = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w0.clamp(min=1e-8))
    z0 = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w0.clamp(min=1e-8))
    q0 = torch.stack([w0, x0, y0, z0], dim=1)  # Shape: (N, 4)
    
    # Case 1: x is the largest component
    x1 = torch.sqrt(s[:, 1]) / 2.0
    w1 = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * x1.clamp(min=1e-8))
    y1 = (R[:, 0, 1] + R[:, 1, 0]) / (4.0 * x1.clamp(min=1e-8))
    z1 = (R[:, 0, 2] + R[:, 2, 0]) / (4.0 * x1.clamp(min=1e-8))
    q1 = torch.stack([w1, x1, y1, z1], dim=1)  # Shape: (N, 4)
    
    # Case 2: y is the largest component
    y2 = torch.sqrt(s[:, 2]) / 2.0
    w2 = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * y2.clamp(min=1e-8))
    x2 = (R[:, 0, 1] + R[:, 1, 0]) / (4.0 * y2.clamp(min=1e-8))
    z2 = (R[:, 1, 2] + R[:, 2, 1]) / (4.0 * y2.clamp(min=1e-8))
    q2 = torch.stack([w2, x2, y2, z2], dim=1)  # Shape: (N, 4)
    
    # Case 3: z is the largest component
    z3 = torch.sqrt(s[:, 3]) / 2.0
    w3 = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * z3.clamp(min=1e-8))
    x3 = (R[:, 0, 2] + R[:, 2, 0]) / (4.0 * z3.clamp(min=1e-8))
    y3 = (R[:, 1, 2] + R[:, 2, 1]) / (4.0 * z3.clamp(min=1e-8))
    q3 = torch.stack([w3, x3, y3, z3], dim=1)  # Shape: (N, 4)
    
    # Stack all possible quaternions
    q_all = torch.stack([q0, q1, q2, q3], dim=0)  # Shape: (4, N, 4)
    
    # Select the correct quaternion for each batch element
    quaternion = q_all[idx, torch.arange(N), :]  # Shape: (N, 4)
    
    # Normalize quaternions to ensure unit length
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True).clamp(min=1e-8)
    
    return quaternion

def adaptive_sparsification(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, scene: Scene, gaussians: GaussianModel, save_iterations: int, output_path: str):

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ##############################################################################################################
    # Training to align surface normals and adjust opacity
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    viewpoint_stack = None

    gaussians.detach_setup()

    opt.scaling_lr *= 0.1
    opt.rotation_lr *= 0.1
    gaussians.compact_feature_field.training_setup(opt)
    gaussians.compact_feature_field.scheduler = ExponentialLR(gaussians.compact_feature_field.optimizer, gamma=opt.gamma)

    if gaussians.compact_feature_field._global_contribution is None:
        gaussians.compact_feature_field._global_contribution = torch.zeros((gaussians.compact_feature_field.get_num_gaussians, 1), device="cuda")

    if pipe.finetune_decoder:
        gaussians.compact_feature_field.autoencoder.train()
        gaussians.compact_feature_field.autoencoder_optimizer = torch.optim.Adam(gaussians.compact_feature_field.autoencoder.parameters(), lr=opt.ae_lr * 0.1)
        # gaussians.light_semantic_gs.autoencoder_scheduler = ExponentialLR(gaussians.light_semantic_gs.autoencoder_optimizer, gamma=0.5)

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.compact_feature_field.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
    
            if iteration < opt.merge_until_iter:

                if pipe.pruning:
                    with torch.no_grad():
                        print("image_shape: ", viewpoint_stack[0].image_width, viewpoint_stack[0].image_height)
                        # prune points based on global contribution
                        contribution = gaussians.compact_feature_field.get_global_contribution.squeeze()
                        opacity = gaussians.compact_feature_field.get_opacity.squeeze()
                        mask = (contribution < opt.contrib_threshold) | (opacity < opt.alpha_threshold)
                        mask = mask.squeeze()

                        survivor_indices = torch.nonzero(~mask).squeeze().cuda()
                        prune_indices = torch.nonzero(mask).squeeze().cuda()
                        knn = knnGraphCUDA(gaussians.compact_feature_field.get_xyz)[0] # knn indices
                        labels = -torch.ones(mask.shape[0], dtype=torch.long, device="cuda")
                        labels[survivor_indices] = survivor_indices
                        neighbor_indices = knn[prune_indices]
                        neighbor_labels = labels[neighbor_indices]

                        # Create a mask for neighbors that already have a valid label.
                        valid_mask = (neighbor_labels != -1)
                        has_valid = valid_mask.any(dim=1)  # Shape: (num_unassigned,)
                        # Convert the boolean mask to float lets us use argmax, which returns the first maximum (i.e. first True).
                        first_valid_idx = valid_mask.float().argmax(dim=1)  # Shape: (num_unassigned,)
                        
                        # Extract the label of the first valid neighbor for each unassigned point.
                        new_labels = neighbor_labels[torch.arange(neighbor_labels.shape[0]), first_valid_idx]
                        
                        # Only update those points that have at least one valid neighbor.
                        labels[prune_indices[has_valid]] = new_labels[has_valid]
                        labels = labels[prune_indices]

                        gaussians.compact_feature_field.prune_points(mask, labels)

                gaussians.compact_feature_field._global_contribution = torch.zeros(gaussians.compact_feature_field.get_global_contribution.shape, device="cuda")
        
        # Pick a random Camera
        view = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Light feature rendering
        latent_features = gaussians.compact_feature_field.normalize_features()
        latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
        render_pkg_light_feature = render(view, gaussians.compact_feature_field, pipe, background, override_color=latent_features)
        latent_rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg_light_feature["render"], render_pkg_light_feature["viewspace_points"], render_pkg_light_feature["visibility_filter"], render_pkg_light_feature["radii"]

        cf3_feature_map = gaussians.compact_feature_field.decode_featuremap(latent_rendering)
        gaussians.compact_feature_field._global_contribution += render_pkg_light_feature["importance_score"].reshape(-1, 1)

        gt_feature_map = None

        # render feature map
        with torch.no_grad():
            render_pkg = render_feature(view, gaussians, pipe, background)
            feature_map = render_pkg["feature_map"]
            feature_map = F.normalize(feature_map, p=2, dim=0)
            feature_map[torch.isnan(feature_map)] = 0.0

        # Loss
        loss = 0

        if opt.lambda_norm > 0:
            norm_map = (1 - cf3_feature_map.norm(dim=0))**2
            norm_loss = norm_map.mean()
            loss += norm_loss * opt.lambda_norm

        # might be buggy for langsplat features
        # if dataset.foundation_model != "langsplat":
        #     cf3_feature_map = F.normalize(cf3_feature_map, p=2, dim=0)
        #     cf3_feature_map[torch.isnan(cf3_feature_map)] = 0.0

        # default training target is the rendered feature map
        if pipe.use_render_feature:
            loss += l1_loss(cf3_feature_map, feature_map.detach())

        if pipe.use_gt_feature:
            gt_feature_map = view.semantic_feature.cuda()
            gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
            gt_feature_map[torch.isnan(gt_feature_map)] = 0.0

            if cf3_feature_map.shape != gt_feature_map.shape:
                cf3_feature_map = nn.functional.interpolate(cf3_feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)

            mask = (gt_feature_map != 0)
            loss += l1_loss(cf3_feature_map[mask], gt_feature_map[mask])

        if pipe.use_render_depth:
            depth_map = render_pkg["depth"]
            light_depth_map = render_pkg_light_feature["depth"]
            loss += opt.lambda_depth * l1_loss(light_depth_map, depth_map)

        loss.backward()
        torch.cuda.empty_cache()
        iter_end.record()

        with torch.no_grad():

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 500 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.write(f"Loss: {ema_loss_for_log:.{7}f}")
            if iteration % 10 == 0:
                progress_bar.update(10)

            # Keep track of max radii in image-space for pruning
            gaussians.compact_feature_field.max_radii2D[visibility_filter] = torch.max(gaussians.compact_feature_field.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.compact_feature_field.add_merge_stats(viewspace_point_tensor, visibility_filter)

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.compact_feature_field.optimizer.step()
            gaussians.compact_feature_field.optimizer.zero_grad(set_to_none = True)

            if pipe.finetune_decoder:
                gaussians.compact_feature_field.autoencoder_optimizer.step()
                gaussians.compact_feature_field.autoencoder_optimizer.zero_grad(set_to_none = True)
            
        with torch.no_grad():
            # Adaptive merging
            if iteration % opt.merge_interval == 0 and iteration < opt.merge_until_iter and pipe.merging:
                print("xyz_grad: ", gaussians.compact_feature_field.xyz_gradient_accum.max(), "feature grad: ", gaussians.compact_feature_field.feature_gradient_accum.max())
                grads = (gaussians.compact_feature_field.xyz_gradient_accum + gaussians.compact_feature_field.feature_gradient_accum) / gaussians.compact_feature_field.denom
                grads[grads.isnan()] = opt.merge_grad_threshold

                # only merge points with no gradients
                merge_grad_mask = torch.norm(grads, dim=1) < opt.merge_grad_threshold
                grad_mask_indices = torch.arange(0, merge_grad_mask.shape[0], device="cuda")[merge_grad_mask]

                points = gaussians.compact_feature_field.get_xyz[merge_grad_mask]
                cov3d = gaussians.compact_feature_field.get_covariance()[merge_grad_mask]
                alpha = gaussians.compact_feature_field.get_opacity[merge_grad_mask]

                # Find gaussians to merge
                cov3d = cov3d[:, ]    # (xx, xy, xz, yy, yz, zz)
                merge_indices, merge_xyz, merge_cov3d, merge_alpha = mergeKNN(points, cov3d, alpha)

                # Create a mask for found matches
                merge_mask = (merge_indices >= 0).squeeze()

                # Get the source indices (i.e. the indices for which there is a valid match), src_indices can have duplicates / tgt_indices are going to be pruned
                tgt_indices = torch.arange(merge_indices.shape[0], device=merge_indices.device)[merge_mask]
                src_indices = merge_indices[merge_mask].squeeze()

                # Avoid duplicating connections (since matching is symmetric),
                duplicate_mask = src_indices < tgt_indices
                src_indices = src_indices[duplicate_mask]
                tgt_indices = tgt_indices[duplicate_mask]
                src_original_indices = grad_mask_indices[src_indices]
                tgt_original_indices = grad_mask_indices[tgt_indices]

                merge_xyz = merge_xyz[src_indices]
                merge_cov3d = merge_cov3d[src_indices]
                merge_alpha = merge_alpha[src_indices]

                cov_matrices = torch.stack([
                    torch.stack([merge_cov3d[:, 0], merge_cov3d[:, 1], merge_cov3d[:, 2]], dim=1),
                    torch.stack([merge_cov3d[:, 1], merge_cov3d[:, 3], merge_cov3d[:, 4]], dim=1),
                    torch.stack([merge_cov3d[:, 2], merge_cov3d[:, 4], merge_cov3d[:, 5]], dim=1)
                ], dim=1)  # Shape: (n, 3, 3)

                # compute rotation quaternions and scaling factors
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)  # eigenvalues: (n, 3), eigenvectors: (n, 3, 3), ascending
                scaling_factors = torch.sqrt(torch.abs(eigenvalues))  # Shape: (n, 3)
                rotation_matrices = ensure_rotation_matrix(eigenvectors)  # Shape: (n, 3, 3) 
                rotation_quats = rotation_matrix_to_quaternion(rotation_matrices)  # Shape: (n, 4)

                # check similarity between features
                latent_features = gaussians.compact_feature_field.get_features
                cos_sim = F.cosine_similarity(latent_features[src_original_indices], latent_features[tgt_original_indices], dim=2).squeeze()
                cossim_mask = (cos_sim > opt.similarity_threshold).squeeze()
                valid_mask = cossim_mask
                n = valid_mask.shape[0]
                print(f"Found matches: {n}, Reject cos: {n - cossim_mask.sum()}", end=" ")

                src_original_indices = src_original_indices[valid_mask]
                tgt_original_indices = tgt_original_indices[valid_mask]
                merge_xyz = merge_xyz[valid_mask]
                merge_alpha = merge_alpha[valid_mask]
                scaling_factors = scaling_factors[valid_mask]
                rotation_quats = rotation_quats[valid_mask]

                src_alpha = gaussians.compact_feature_field.get_opacity[src_original_indices].reshape(-1, 1, 1)
                tgt_alpha = gaussians.compact_feature_field.get_opacity[tgt_original_indices].reshape(-1, 1, 1)

                gaussians.compact_feature_field._xyz[src_original_indices] = merge_xyz
                gaussians.compact_feature_field._opacity[src_original_indices] = gaussians.compact_feature_field.inverse_opacity_activation(merge_alpha)
                gaussians.compact_feature_field._features_dc[src_original_indices] = (gaussians.compact_feature_field._features_dc[src_original_indices] * src_alpha + gaussians.compact_feature_field._features_dc[tgt_original_indices] * tgt_alpha) / (src_alpha + tgt_alpha)
                scaling = gaussians.compact_feature_field.scaling_inverse_activation(scaling_factors)
                gaussians.compact_feature_field._scaling[src_original_indices] = scaling
                gaussians.compact_feature_field._rotation[src_original_indices] = rotation_quats

                # prune target points
                prune_mask = torch.zeros(gaussians.compact_feature_field.get_xyz.shape[0], device="cuda")
                prune_mask[tgt_original_indices] = 1
                prune_mask = prune_mask.bool()
                print("pairs: ", src_original_indices.shape[0], tgt_original_indices.shape[0])
                gaussians.compact_feature_field.prune_points(prune_mask, src_original_indices)
                gaussians.compact_feature_field.merge_postfix()
                gaussians.compact_feature_field.scheduler.step()

            if iteration in save_iterations or iteration == opt.iterations:
                # save the model
                point_cloud_path = os.path.join(output_path, f"point_cloud/iteration_{scene.loaded_iter + iteration}")
                gaussians.compact_feature_field.save_ply(os.path.join(point_cloud_path, "feature_field.ply"))
                # number of gaussians
                with open(os.path.join(point_cloud_path, "num_gaussians.txt"), 'w') as f:
                    f.write("number of gaussians: " + str(gaussians.get_num_gaussians) + "\n")
                    f.write("number of feature gaussians: " + str(gaussians.compact_feature_field.get_num_gaussians) + "\n")

    return gaussians


def compress_features(opt: OptimizationParams, pipe: PipelineParams, scene: Scene, gaussians: GaussianModel):

    # Training parameters
    num_epochs = getattr(opt, "compress_epoch", 30)
    batch_size = getattr(opt, "compress_batch_size", 64)

    # hyperparameter for the autoencoder
    hidden_dims = [128, 64, 32, 16]

    # Initialize the compact feature field with an autoencoder
    gaussians.set_compact_feature_field()
    gaussians.compact_feature_field.set_autoencoder(gaussians.semantic_feature_size, hidden_dims, opt.ae_lr)

    # train the autoencoder to compress the semantic features
    with torch.no_grad():
        semantic_feature_data = gaussians.get_semantic_feature.detach().requires_grad_(False)

        data_mask = (gaussians._global_contribution > opt.contrib_threshold).squeeze()  # (N, )

        if pipe.filter_var:
            threshold = gaussians._semantic_feature_var.quantile(0.9999).item()
            var_mask = (gaussians._semantic_feature_var < threshold).squeeze()   # (N, )
            data_mask[~var_mask] = False
            print(f"Train only features with low variance than {threshold:.2f}:", data_mask.sum(), "out of", len(data_mask))

        # Filter based on variance
        semantic_feature_data = semantic_feature_data[data_mask].reshape(-1, gaussians.semantic_feature_size)

        if pipe.normalize_feature:
            semantic_feature_data = F.normalize(semantic_feature_data, p=2, dim=1)

        semantic_feature_data = semantic_feature_data.detach().cpu()
        print("semantic_feature_data: ", semantic_feature_data.shape)

    # Create a DataLoader for batching
    dataset = TensorDataset(semantic_feature_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    mse_criterion = nn.MSELoss()
    metric_criterion = nn.L1Loss()
    lambda_cossim = getattr(opt, "lambda_cossim", 0.01)
    lambda_metric = getattr(opt, "lambda_metric", 0.01)

    writer = SummaryWriter(log_dir="runs/autoencoder_compression")

    # Training loop
    print("Training the autoencoder to compress semantic features...")

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        epoch_cossim_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_metric_loss = 0.0
        epoch_scale_loss = 0.0

        for batch in dataloader:
            gaussians.compact_feature_field.autoencoder_optimizer.zero_grad()
            batch_features = batch[0].to("cuda", non_blocking=True)
            reconstruction, latent = gaussians.compact_feature_field.autoencoder(batch_features)

            mse_loss = mse_criterion(reconstruction, batch_features)
            cos_sim = F.cosine_similarity(reconstruction, batch_features, dim=1)
            cossim_loss = (1 - cos_sim).mean()

            # basic loss for training the autoencoder
            loss = mse_loss + lambda_cossim * cossim_loss
            
            # optional regularization (can be excluded)
            batch_features_norm = F.normalize(batch_features, p=2, dim=1)
            latent_norm = F.normalize(latent, p=2, dim=1)
            # Compute full pairwise cosine similarity matrices (B x B)
            input_cosine = torch.mm(batch_features_norm, batch_features_norm.t())
            latent_cosine = torch.mm(latent_norm, latent_norm.t())
            metric_loss = metric_criterion(latent_cosine, input_cosine)
            loss += lambda_metric * metric_loss

            loss.backward()
            gaussians.compact_feature_field.autoencoder_optimizer.step()

            epoch_loss += loss.item()
            epoch_cossim_loss += cossim_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_metric_loss += metric_loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_mse_loss = epoch_mse_loss / len(dataloader)
        avg_cossim_loss = epoch_cossim_loss / len(dataloader)
        avg_metric_loss = epoch_metric_loss / len(dataloader)

        if epoch % 5 == 0:
            tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_epoch_loss:.5f} (MSE: {avg_mse_loss:.5f}, Cosine: {avg_cossim_loss:.5f}, Metric: {avg_metric_loss:.5f})")
        
        # --------- Logging the Metrics ---------
        writer.add_scalar('Loss/train_total', avg_epoch_loss, epoch+1)
        writer.add_scalar('Loss/train_mse', avg_mse_loss, epoch+1)
        writer.add_scalar('Loss/train_cosine', avg_cossim_loss, epoch+1)

    writer.close()

    # After training, use the encoder to compress all features
    gaussians.compact_feature_field.set_feature(gaussians.get_semantic_feature)

    return gaussians


def set_features(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, iteration : int, save_iterations: int, skip_train : bool, skip_test : bool, output_path : str):

    # default semantic feature size is 512
    semantic_feature_size = 512
    # If you are using different semantic feature size, you can change it here
    if dataset.foundation_model == "dino":
        semantic_feature_size = 384

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
    
    ### STEP1: accumulate features from all training views
    # if the features are not lifted, lift the features first
    if not gaussians.set_feature_flag:
        gaussians.lift_feature_setup()

        accum_features = torch.zeros((gaussians.get_num_gaussians, 1, gaussians.semantic_feature_size), dtype=torch.float, device="cuda").requires_grad_(False)
        accum_features_square = torch.zeros((gaussians.get_num_gaussians, 1, gaussians.semantic_feature_size), dtype=torch.float, device="cuda").requires_grad_(False)
        accum_contribution = torch.zeros((gaussians.get_num_gaussians), dtype=torch.float, device="cuda").requires_grad_(False)

        start_time = time.time()
        with torch.no_grad():
            for view in tqdm(viewpoint_stack):
                gt_feature_map = view.semantic_feature.cuda().float()
                gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
                gt_feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(view.image_height, view.image_width), mode='bilinear', align_corners=True).squeeze(0)
                
                render_pkg = render_collect_features(view, gaussians, pipe, background, gt_feature_map)

                accum_features += render_pkg["accum_features"]
                accum_features_square += render_pkg["accum_features_square"]
                accum_contribution += render_pkg["importance_score"]
            
            elapsed_time = time.time() - start_time
            print(f"Time elapsed for feature accumulation: {elapsed_time:.4f} seconds")
            print("fps: ", len(viewpoint_stack) / elapsed_time)

            gaussians.set_semantic_feature(accum_contribution, accum_features, accum_features_square)

        del accum_features
        del accum_features_square
        del accum_contribution
        torch.cuda.empty_cache()

    ### STEP2: compress the features (Traning AutoEncoder)
    gaussians = compress_features(opt, pipe, scene, gaussians)
    torch.cuda.empty_cache()

    # if not skip_test and (len(scene.getTestCameras()) > 0):
    #     render_save_features(output_path, f"test_before_merging", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, background)
    # else:
    #     render_save_features(output_path, f"train_before_merging", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, background)

    point_cloud_path = os.path.join(output_path, f"point_cloud/iteration_{scene.loaded_iter}")
    gaussians.compact_feature_field.save_ply(os.path.join(point_cloud_path, "feature_field.ply"))

    # save gaussian data
    train_before_merging_path = os.path.join(output_path, "train_before_merging")
    makedirs(train_before_merging_path, exist_ok=True)
    with open(os.path.join(train_before_merging_path, "gaussian_data.txt"), 'w') as f:
        f.write("number of gaussians: " + str(gaussians.get_num_gaussians) + "\n")
        f.write("number of compact feature gaussians: " + str(gaussians.compact_feature_field.get_num_gaussians) + "\n")

    ### STEP3: perform adaptive sparsification
    gaussians = adaptive_sparsification(dataset, opt, pipe, scene, gaussians, save_iterations, output_path)

    with torch.no_grad():
        # save the model
        point_cloud_path = os.path.join(output_path, f"point_cloud/iteration_{scene.loaded_iter + opt.iterations}")
        gaussians.compact_feature_field.save_ply(os.path.join(point_cloud_path, "feature_field.ply"))
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), save_semantic_feature=False)

        if not skip_test and (len(scene.getTestCameras()) > 0):
            render_save_features(output_path, f"test", scene.loaded_iter + opt.iterations, scene.getTestCameras(), gaussians, pipe, background)
        elif not skip_train:
            render_save_features(output_path, f"train", scene.loaded_iter + opt.iterations, scene.getTrainCameras(), gaussians, pipe, background)

        lightfps, initialfps, importance_score = collect_fps_importance(output_path, f"train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, background)
        
        # save gaussian data
        with open(os.path.join(point_cloud_path, "gaussian_data.txt"), 'w') as f:
            f.write("number of gaussians: " + str(gaussians.get_num_gaussians) + "\n")
            f.write("number of compact feature gaussians: " + str(gaussians.compact_feature_field.get_num_gaussians) + "\n")
            f.write("light fps: " + str(lightfps) + "\n")
            f.write("initial fps: " + str(initialfps) + "\n")

        np.savez(os.path.join(point_cloud_path,"imp_score"), importance_score.flatten().cpu().detach().numpy()) 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", "-o", default="", type=str)
    args = get_combined_args(parser)

    if args.output == "":
        args.output = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)
    set_features(model.extract(args), op.extract(args), pipeline.extract(args), args.iteration, args.save_iterations, args.skip_train, args.skip_test, args.output) ###

    print("Done")
