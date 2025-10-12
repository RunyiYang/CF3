#!/usr/bin/env python
"""
Usage:
    python featuregs_script.py \
        --property_path /path/to/property.ply \
        --ae_path /path/to/decoder.pth \
        --featuregs_path /path/to/featuregs_point_cloud.ply \
        --output_path ./results \
        --text_queries "a can of red bull drink" \
        --threshold 0.9 \
        --gpu 0
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
from openclip_encoder import OpenCLIPTextEncoder
from torch import nn


# ----------------- Parse arguments -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run Feature3DGS visualization pipeline with CNN decoder + CLIP")
    parser.add_argument("--property_path", type=str, required=True, help="PLY file path used to define property names")
    parser.add_argument("--ae_path", type=str, required=True, help="Trained CNN decoder .pth file path")
    parser.add_argument("--featuregs_path", type=str, required=True, help="Feature3DGS point cloud PLY path")
    parser.add_argument("--output_path", type=str, required=True, help="Output folder to save results")
    parser.add_argument("--text_queries", type=str, required=True, help="Comma separated text queries")
    parser.add_argument("--feature_size", type=int, default=128, help="Feature dimension size (input to CNN decoder)")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for similarity")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use (CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


# ----------------- Utility: save PLY -----------------
def save_colored_ply(xyz: np.ndarray, colors: np.ndarray, out_path: str):
    """
    xyz:    (N,3) float32
    colors: (N,3) uint8
    """
    n = xyz.shape[0]
    dtype = [
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','u1'),('green','u1'),('blue','u1')
    ]
    vertex = np.empty(n, dtype=dtype)
    vertex['x'], vertex['y'], vertex['z'] = xyz.T
    vertex['red'], vertex['green'], vertex['blue'] = colors.T
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(out_path)
    print(f"Saved {n} points to {out_path}")


# ----------------- CNN Decoder -----------------
class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----------------- Main -----------------
def main():
    args = get_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure output folder exists
    os.makedirs(args.output_path, exist_ok=True)

    # 1) Load CNN decoder
    cnn_decoder = CNN_decoder(args.feature_size, 512).to(device)
    cnn_decoder.load_state_dict(torch.load(args.ae_path))
    cnn_decoder.eval()

    # 2) Load Feature3DGS point cloud & features
    ply = PlyData.read(args.langsplat_path)['vertex'].data
    props = ['x','y','z'] + [f'semantic_{i}' for i in range(args.feature_size)]
    arr = np.vstack([ply[p] for p in props]).T
    fgs = torch.from_numpy(arr).float().to(device)
    fgs_pc       = fgs[:, :3]
    fgs_feat_3d  = fgs[:, 3:]

    sel_gs = np.zeros(fgs_pc.shape[0], dtype=bool)

    # 3) Decode features to 512-dim
    fgs_feat_3d = fgs_feat_3d.unsqueeze(-1).unsqueeze(-1)
    with torch.no_grad():
        fgs_feat = cnn_decoder(fgs_feat_3d)
    fgs_feat = fgs_feat.squeeze(-1).squeeze(-1)

    # 4) Normalize features
    fgs_norm = F.normalize(fgs_feat, dim=1).detach().cpu().numpy()

    # 5) Prepare CLIP text encoder
    clip = OpenCLIPTextEncoder(device)

    # 6) Process each query
    for query in [q.strip() for q in args.text_queries.split(",")]:
        with torch.no_grad():
            txt_feat = clip.encode(query).squeeze().to(torch.float32)

        similarity_map = torch.einsum("nc,c->n",
            F.normalize(fgs_feat, dim=1),
            F.normalize(txt_feat, dim=0))
        sims = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-9)
        sims = sims.detach().cpu().numpy()

        # A) Feature3DGS: mark selected points
        sel_lsp = sims > args.threshold
        for lsp_idx, is_sel in enumerate(sel_lsp):
            if is_sel:
                sel_gs[lsp_idx] = True

        # Save selected GS
        orig_ply = PlyData.read(args.langsplat_path)
        orig_vert = orig_ply['vertex']
        data = orig_vert.data

        # Load property list from property PLY
        prop_ply    = PlyData.read(args.property_path)['vertex'].data
        props_list  = list(prop_ply.dtype.names)

        # Keep only properties existing in both
        all_orig_props = data.dtype.names
        keep_props     = [p for p in props_list if p in all_orig_props]

        # Apply mask
        filtered = data[sel_gs]
        new_dtype = [(p, data.dtype.fields[p][0]) for p in keep_props]
        new_data = np.empty(filtered.shape, dtype=new_dtype)
        for p in keep_props:
            new_data[p] = filtered[p]

        # Save filtered point cloud
        new_vert = PlyElement.describe(new_data, name='vertex')
        out = PlyData([new_vert], text=True)
        out.write(os.path.join(args.output_path, "point_cloud.ply"))
        print("Done")


if __name__ == "__main__":
    main()
