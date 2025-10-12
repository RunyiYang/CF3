#!/usr/bin/env python
import os
import json
import glob
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision
import matplotlib.pyplot as plt

# 추가: SLERP와 Cubic Spline 보간을 위한 SciPy import
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline

# 기존 모듈 임포트
from scene import Scene
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral
import colormaps
from clip import tokenize
#from openclip_encoder import OpenCLIPTextEncoder
from scipy.spatial.transform import Rotation

def load_text_queries(json_folder: str) -> list:
    """
    json_folder 내의 모든 frame_*.json 파일에서 category 정보를 읽어,
    중복 없이 텍스트 쿼리 리스트를 반환합니다.
    """
    json_paths = sorted(glob.glob(os.path.join(json_folder, 'frame_*.json')))
    categories_set = set()
    for js_path in json_paths:
        with open(js_path, 'r') as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            cat = obj.get("category", None)
            if cat is not None:
                categories_set.add(cat)
    text_queries = list(categories_set)
    print(f"Loaded text queries: {text_queries}")
    return text_queries

def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth

# ──────────────────────────────────────────────
# 부드러운 trajectory를 생성하는 함수
# ──────────────────────────────────────────────
def generate_smooth_trajectory(poses, num_frames, rot_weight=1.0):
    """
    poses: np.array of shape (N, 3, 4)
    num_frames: 생성할 전체 프레임 개수
    rot_weight: 회전 각도에 대한 가중치 (기본 1.0)
    """
    N = poses.shape[0]
    # 1) translation과 rotation 분리
    translations = poses[:, :3, 3]                # (N,3)
    rotations = Rotation.from_matrix(poses[:, :3, :3])  # (N)

    # 2) 각 세그먼트의 '비용' 계산: dt + rot_weight * dθ
    seg_costs = []
    for i in range(N-1):
        # translation 거리
        dt = np.linalg.norm(translations[i+1] - translations[i])
        # 회전 각도 (rad 단위)
        rel_rot = rotations[i].inv() * rotations[i+1]
        dtheta = np.linalg.norm(rel_rot.as_rotvec())
        seg_costs.append(dt + rot_weight * dtheta)

    total_cost = sum(seg_costs)
    # 3) 누적 비용 비율로 key_times 생성
    key_times = [0.0]
    cum = 0.0
    for c in seg_costs:
        cum += c
        key_times.append(cum / total_cost)
    key_times = np.array(key_times)  # shape (N,)

    # 4) SLERP, CubicSpline으로 보간
    slerp = Slerp(key_times, rotations)
    interp_times = np.linspace(0.0, 1.0, num_frames)
    interp_rots = slerp(interp_times).as_matrix()           # (num_frames,3,3)
    cs = CubicSpline(key_times, translations, axis=0)
    interp_trans = cs(interp_times)                        # (num_frames,3)

    # 5) 합쳐서 (num_frames,3,4) 배열 반환
    smooth_poses = np.zeros((num_frames, 3, 4), dtype=poses.dtype)
    for i in range(num_frames):
        smooth_poses[i, :3, :3] = interp_rots[i]
        smooth_poses[i, :, 3] = interp_trans[i]
    return smooth_poses

# ──────────────────────────────────────────────
# render_sets 함수 (기존 작업 유지, 단 카메라 view 생성 부분 수정)
# ──────────────────────────────────────────────
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, 
                novel_view: bool, video: bool, edit_config: str, novel_video: bool, multi_interpolate: bool, num_views: int,
                output_text_dir: str, mask_thresh: float , text_queries: str, save_rgb: bool = False, keyframe_num: int = 6, start_idx: int = 2):
    """
    Gaussian 모델로 전체 scene의 feature를 얻은 후, 
    json_folder 내의 모든 프레임에서 category 정보를 모아 텍스트 쿼리셋을 구성한 뒤 
    각 쿼리에 대해 heatmap 및 예측 mask를 계산하여 npy 파일로 저장합니다.
    """
    with torch.no_grad():
        # Gaussian 모델 및 Scene 초기화
        gaussians = GaussianModel(dataset.sh_degree, 512)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 출력 폴더 생성
        os.makedirs(output_text_dir, exist_ok=True)
        
        # JSON 폴더에서 텍스트 쿼리셋 구성 (모든 프레임에서 category 추출, 중복 제거)
        #text_queries = load_text_queries(json_folder)
        text_queries = [item.strip() for item in text_queries.split(",")]

        
        # CLIP 모델 준비 (텍스트 인코딩용)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)
        clip_model = upsampler.model
        
        # colormap 옵션 설정
        colormap_options = colormaps.ColormapOptions(
            colormap="turbo",
            normalize=True,
            colormap_min=-1.0,
            colormap_max=1.0,
        )

        # 각 텍스트 쿼리에 대해 처리
        for query in text_queries:
            text = tokenize(query).to(device)
            text_feat = clip_model.model.encode_text(text).squeeze().to(torch.float32)
            
            if text_feat.ndim != 1:
                text_feat = text_feat.squeeze()
            
            query_path = os.path.join(output_text_dir, f"{query}")
            render_path = os.path.join(query_path, "renders")
            os.makedirs(render_path, exist_ok=True)
            
            # views[0:50] 사용 (원래의 keyframe들)
            views = scene.getTrainCameras()
            view = views[0]
            # keyframe_indices 인자가 없으면, 기존 방식으로 10개 균등 추출
            
            
            keyframe_indices = np.linspace(start_idx, len(views)-1, keyframe_num, dtype=int)
            
            
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (view.original_image.shape[2], view.original_image.shape[1])
            
            if save_rgb:
                video_path = os.path.join(render_path, '../rgb.mp4')
            else:
                video_path = os.path.join(render_path, f'../cf3_{query}.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20, size)
            
            # 기존의 render_poses 계산 (각 keyframe의 3×4 pose)
            render_poses = []
            """for cam in views:
                pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], axis=1)
                render_poses.append(pose)
            render_poses = np.array(render_poses)  # shape: (N, 3, 4)"""
            for idx in keyframe_indices:
                cam = views[idx]
                pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], axis=1)
                render_poses.append(pose)
            render_poses = np.array(render_poses)  # shape: (4, 3, 4)

           
            # ──────────────────────────────────────────────
            # [변경 부분] 기존 인접 frame linear interpolation 대신,
            # 전체 keyframe을 부드럽게 피팅하는 trajectory 생성
            # num_views 인자를 보간할 전체 프레임 수로 사용 (예: 200)
            # ──────────────────────────────────────────────
            smooth_poses = generate_smooth_trajectory(render_poses, num_frames=num_views)
            
            latent_features = gaussians.compact_feature_field.normalize_features()

            # rendering process
            for idx, pose in enumerate(tqdm(smooth_poses, desc="Rendering progress")):
                # view의 변환 행렬 갱신
                world_view = getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)
                view.world_view_transform = torch.tensor(world_view).transpose(0, 1).cuda()
                view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                view.camera_center = view.world_view_transform.inverse()[3, :3]

                rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.).permute(1, 2, 0)

                if save_rgb:
                    rgb_np = (rendering.detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
                    video_writer.write(rgb_np)
                else:
                    latent_features = gaussians.compact_feature_field.get_normalized_features.reshape(-1, 3)
                    latent_feature_map = render(view, gaussians.compact_feature_field, pipeline, background, override_color=latent_features)["render"]
                    image_feats = gaussians.compact_feature_field.decode_featuremap(latent_feature_map)
                    
                    # (b) 이미지와 단일 텍스트 피쳐 간 유사도 계산: (H, W)
                    similarity_map = torch.einsum("chw,c->hw", 
                                                F.normalize(image_feats, dim=0),
                                                F.normalize(text_feat, dim=0))

                    # (c) 필터링: cv2의 평균 필터 적용 (평활화)
                    np_relev = similarity_map.detach().cpu().numpy()
                    scale = 30
                    kernel = np.ones((scale, scale)) / (scale**2)
                    avg_filtered = cv2.filter2D(np_relev, -1, kernel)
                    avg_filtered = torch.from_numpy(avg_filtered).to(similarity_map.device)
                    filtered_map = 0.5 * (avg_filtered + similarity_map)
                    
                    
                    heatmap_norm = (filtered_map - filtered_map.min()) / (filtered_map.max() - filtered_map.min() + 1e-9)
                    valid_composited = colormaps.apply_colormap(heatmap_norm.unsqueeze(-1), colormap_options)
                    output = filtered_map
                    output = output - torch.min(output)
                    output = output / (torch.max(output) + 1e-9)
                    output = output * (1.0 - (-1.0)) + (-1.0)
                    output = torch.clip(output, 0, 1)
                    
                    mask_pred = (output.detach().cpu().numpy() > mask_thresh).astype(np.uint8)
                    mask_pred = smooth(mask_pred)
                    mask_vis = mask_pred.astype(bool)

                    print(mask_pred.shape, rendering.shape, valid_composited.shape)
                    valid_composited[~mask_vis, :] = rendering[~mask_vis, :] * 0.5
                    
                    # blended masked-heatmap frame
                    masked_np = (valid_composited.detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
                    video_writer.write(masked_np)
        
            video_writer.release()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Gaussian 렌더링과 JSON 기반 텍스트 쿼리로 heatmap/mask 생성")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--novel_video", action="store_true")
    parser.add_argument("--edit_config", default="no editing", type=str)
    parser.add_argument("--multi_interpolate", action="store_true")
    parser.add_argument("--num_views", default=300, type=int)
    parser.add_argument("--output_text_dir", type=str, default="text_query_outputs", help="텍스트 쿼리 결과 npy 파일 저장 폴더")
    parser.add_argument("--mask_thresh", type=float, default=0.4, help="마스크 예측 threshold")
    parser.add_argument("--text_queries", type=str, help="text queries to use")
    parser.add_argument("--rgb", action="store_true",help="if set, save RGB video (rgb.mp4); otherwise save masked heatmap (masked_heatmap.mp4)")
    parser.add_argument("--keyframe_num", default=6, type=int)
    parser.add_argument("--start_idx", default=2, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.novel_view,
                args.video, args.edit_config, args.novel_video, args.multi_interpolate, args.num_views,
                args.output_text_dir, args.mask_thresh,args.text_queries, args.rgb,args.keyframe_num, args.start_idx)