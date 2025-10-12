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
from gaussian_renderer import render, render_feature
from gaussian_renderer import GaussianModel
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import time
# from utils.pose_utils import render_path_spiral
# from utils.clip_utils import CLIPEditor
# import yaml

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

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, edit_config, selected_frames = None):

#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
#     gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
#     saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
#     saved_hr_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_hr_feature")
#     depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
    
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
#     makedirs(feature_map_path, exist_ok=True)
#     makedirs(gt_feature_map_path, exist_ok=True)
#     makedirs(saved_feature_path, exist_ok=True)
#     makedirs(saved_hr_feature_path, exist_ok=True)
#     makedirs(depth_path, exist_ok=True) ###

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

#         if selected_frames is not None and view.image_name not in selected_frames:
#             print("skip", view.image_name)
#             continue
        
#         render_pkg = render_feature(view, gaussians, pipeline, background) 

#         gt = view.original_image[0:3, :, :]
#         gt_feature_map = view.semantic_feature.cuda() 
#         torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
#         ### depth ###
#         depth = render_pkg["depth"]
#         scale_nor = depth.max().item()
#         depth_nor = depth / scale_nor
#         depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
#         colormap = plt.get_cmap('jet')
#         depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
#         depth_colored_rgb = depth_colored[:, :, :3]
#         depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
#         output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
#         depth_image.save(output_path)
#         ##############

#         # visualize feature map
#         feature_map = render_pkg["feature_map"] 
#         hr_feature = feature_map
#         feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###

#         feature_map_vis = feature_visualize_saving(feature_map, gt_feature_map)
#         Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

#         # save feature map
#         feature_map = feature_map.cpu().numpy().astype(np.float16)
#         hr_feature = hr_feature.cpu().numpy().astype(np.float16)
#         torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
#         torch.save(torch.tensor(hr_feature).half(), os.path.join(saved_hr_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
#         torch.save(torch.tensor(gt_feature_map).half(), os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))


def render_set(model_path: str, name: str, iteration: int, views: list, gaussians: GaussianModel, pipeline: PipelineParams, background, edit_config = "no_edit", selected_frames = None):

    feature_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_vis")
    saved_hr_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
    saved_hr_light_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature_light")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    makedirs(feature_vis_path, exist_ok=True)
    makedirs(saved_hr_feature_path, exist_ok=True)
    makedirs(saved_hr_light_feature_path, exist_ok=True)

    # measure time
    lightgs_time = 0
    featuregs_time = 0
    elasped_time = []

    with torch.no_grad():
        latent_features = gaussians.compact_feature_field.normalize_features()

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if selected_frames is not None and view.image_name not in selected_frames:
                continue

            # read time
            lightgs_start = time.time()
            
            latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
            latent_feature_map = render(view, gaussians.compact_feature_field, pipeline, background, override_color=latent_features)["render"]
            
            light_feature_map = gaussians.compact_feature_field.decode_featuremap(latent_feature_map)
            torch.cuda.synchronize()

            lightgs_end = time.time()
            elasped_time.append(lightgs_end - lightgs_start)
            lightgs_time += lightgs_end - lightgs_start


            featuregs_start = time.time()            
            # original feature rendering
            render_pkg = render_feature(view, gaussians, pipeline, background)
            torch.cuda.synchronize()
            featuregs_end = time.time()
            featuregs_time += featuregs_end - featuregs_start

            feature_map = render_pkg["feature_map"]
            gt_feature_map = view.semantic_feature.cuda()

            # gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            render_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            # # save gt feature map
            # torch.save(torch.tensor(gt_feature_map).half(), os.path.join(gt_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

            C, H, W = light_feature_map.shape

            # if light_feature_map.shape != gt_feature_map.shape:
            #     light_feature_map = nn.functional.interpolate(light_feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            # if feature_map.shape != gt_feature_map.shape:
            #     feature_map = nn.functional.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
            # save feature map
            torch.save(feature_map.half(), os.path.join(saved_hr_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            torch.save(light_feature_map.half(), os.path.join(saved_hr_light_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

            if light_feature_map.shape != gt_feature_map.shape:
                # light_feature_map = nn.functional.interpolate(light_feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                # feature_map = nn.functional.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                gt_feature_map = nn.functional.interpolate(gt_feature_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

            # save pca visualization
            fmap_light = nn.functional.normalize(light_feature_map[None, ...], dim=1)
            fmap = nn.functional.normalize(feature_map[None, ...], dim=1)
            gt_fmap = nn.functional.normalize(gt_feature_map[None, ...], dim=1)

            # change nan to 0
            fmap[torch.isnan(fmap)] = 0.0
            fmap_light[torch.isnan(fmap_light)] = 0.0
            gt_fmap[torch.isnan(gt_fmap)] = 0.0

            # concatenate feature map
            concat_fmap = torch.cat([gt_fmap, fmap, fmap_light], dim=3)    # (1, C, H, 3*W)
            pca = sklearn.decomposition.PCA(3, random_state=42)
            feature_concat = concat_fmap.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy()

            pca.fit(fmap_light.permute(0, 2, 3, 1).reshape(-1, C)[::3].cpu().numpy())
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
            plt.title(f"Rendered FeatureGS")

            plt.subplot(1, 3, 3)
            plt.imshow(light_feature_vis.cpu().numpy())
            plt.axis("off")
            plt.title(f"Light FeatureGS")

            plt.suptitle(f"View {idx:05d} Feature Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(feature_vis_path, f'{idx:05d}_feature_vis_concat_sub.png'))
            plt.close()

            # save pca visualization for each feature map
            # Since torchvision expects (C, H, W) we need to permute the dimensions.
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

    lightgs_fps = len(views) / lightgs_time
    featuregs_fps = len(views) / featuregs_time
    print("LightGS FPS: ", lightgs_fps, "FeatureGS FPS: ", featuregs_fps)

    # save the fps in file
    # Save FPS results to a file
    fps_save_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fps_results.txt")
    with open(fps_save_path, "a") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"LightGS FPS: {lightgs_fps:.2f}\n")
        f.write(f"FeatureGS FPS: {featuregs_fps:.2f}\n")
        f.write("-" * 50 + "\n")
        f.write("LightGS elapsed times (seconds):\n")
        for t in elapsed_time:
            f.write(f"{t:.4f}\n")
        

def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)


def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)


###
def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background, 
                       edit_config, speedup, multi_interpolate, num_views):

    if multi_interpolate:
        name = name + "_multi_interpolate"

    # non-edit
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")

    makedirs(render_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)

    view = views[0]
    
    # create novel poses
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose) 
    if not multi_interpolate:
        poses = interpolate_matrices(render_poses[0], render_poses[-1], num_views)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        # mlp encoder
        render_pkg = render_feature(view, gaussians, pipeline, background) 

        gt_feature_map = view.semantic_feature.cuda() 
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        # visualize feature map
        feature_map = render_pkg["feature_map"]
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###

        feature_map_vis = feature_visualize_saving(feature_map, gt_feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))


def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, edit_config): 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    
    # create novel poses
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 200) 

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        rendering = torch.clamp(render_feature(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, novel_view : bool, 
                video : bool , edit_config: str, novel_video : bool, multi_interpolate : bool, num_views : int): 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 512)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config)
            #  render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, selected_frames=frames)

        if not skip_test and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, edit_config)

        # if novel_view:
        #      render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
        #                         edit_config, multi_interpolate, num_views)
        # if novel_video:
        #      render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config)


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