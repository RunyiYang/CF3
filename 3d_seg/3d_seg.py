#!/usr/bin/env python
"""
Usage:
    python script.py \
        --property_path /path/to/property.ply \
        --ae_path /path/to/autoencoder.pth \
        --cf3_path /path/to/feature_field.ply \
        --reference_path /path/to/reference.ply \
        --output_path /path/to/output \
        --text_queries "a can of red bull drink" \
        --k 3 \
        --threshold 0.9 \
        --gpu 0
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from plyfile import PlyData, PlyElement
from openclip_encoder import OpenCLIPTextEncoder
from torch import nn
import copy


# ----------------- Parse arguments -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run CF3 visualization pipeline with AE + CLIP")
    parser.add_argument("--property_path", type=str, required=True, help="PLY file path used to define property names")
    parser.add_argument("--ae_path", type=str, required=True, help="Trained Autoencoder .pth file path")
    parser.add_argument("--cf3_path", type=str, required=True, help="CF3 PLY file path with compressed features")
    parser.add_argument("--reference_path", type=str, required=True, help="Reference GS PLY file path with semantic features")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for filtered GS PLY")
    parser.add_argument("--text_queries", type=str, required=True, help="Comma separated text queries")
    parser.add_argument("--feature_size", type=int, default=512, help="Feature dimension size")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors for mapping")
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


# ----------------- Autoencoder -----------------
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size=[128,64,32,16]):
        super().__init__()
        # encoder
        enc = []
        d = input_size
        for h in latent_size:
            enc += [nn.Linear(d,h,bias=False), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d,3,bias=False))
        self.encoder = nn.Sequential(*enc)
        # decoder
        dec = []
        d = 3
        for h in reversed(latent_size):
            dec += [nn.Linear(d,h,bias=False), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d,input_size,bias=False))
        self.decoder = nn.Sequential(*dec)

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z), z


# ----------------- Main -----------------
def main():
    args = get_args()

    # Make sure output folder exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set CUDA device before torch import
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load AE
    ae = Autoencoder(args.feature_size).to(device)
    ae.load_state_dict(torch.load(args.ae_path))
    ae.eval()

    # 2) Load CF3 point cloud & decode features
    ply = PlyData.read(args.cf3_path)['vertex'].data
    arr = np.vstack([ply[p] for p in ['x','y','z','f_dc_0','f_dc_1','f_dc_2']]).T
    cf3 = torch.from_numpy(arr).float().to(device)
    cf3_pc       = cf3[:, :3]
    cf3_feat_3d  = cf3[:, 3:]
    with torch.no_grad():
        cf3_feat = ae.decoder(cf3_feat_3d)  # (N',512)

    # 3) Load reference GS point cloud & semantic features
    ply = PlyData.read(args.reference_path)['vertex'].data
    props = ['x','y','z'] + [f'semantic_{i}' for i in range(args.feature_size)]
    arr = np.vstack([ply[p] for p in props]).T
    gs = torch.from_numpy(arr).float().to(device)
    gs_pc       = gs[:, :3]
    gs_feature  = gs[:, 3:]

    # Remove NaN points
    gs_pc_np      = gs_pc.cpu().numpy()
    gs_feat_np    = gs_feature.cpu().numpy()
    sel_gs_orig = np.zeros(gs_pc_np.shape[0], dtype=bool)
    valid_mask_np = ~np.isnan(gs_pc_np).any(axis=1)
    orig_indices  = np.nonzero(valid_mask_np)[0]
    gs_pc_clean       = gs_pc_np[ valid_mask_np ]
    gs_feat_clean     = gs_feat_np[ valid_mask_np ]
    gs_pc   = torch.from_numpy(gs_pc_clean).to(device)
    gs_feature = torch.from_numpy(gs_feat_clean).to(device)

    # 4) Build GS→CF3 mapping
    cf3_norm = F.normalize(cf3_feat,   dim=1).detach().cpu().numpy()
    gs_norm  = F.normalize(gs_feature, dim=1).detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=args.k, n_jobs=-1).fit(cf3_pc.cpu().numpy())
    idxs = nbrs.kneighbors(gs_pc.cpu().numpy(), return_distance=False)
    gs_expand = torch.from_numpy(gs_norm).unsqueeze(1)
    cf3_cand  = torch.from_numpy(cf3_norm)[idxs]
    cos_sim   = (gs_expand * cf3_cand).sum(dim=2)
    best_k    = torch.argmax(cos_sim, dim=1).numpy()
    assigned  = idxs[np.arange(gs_norm.shape[0]), best_k]
    mapping = defaultdict(list)
    for gs_idx, cf3_idx in enumerate(assigned):
        mapping[int(cf3_idx)].append(gs_idx)
    result = [mapping[i] for i in range(cf3_pc.shape[0])]

    # 5) Prepare CLIP text encoder
    clip = OpenCLIPTextEncoder(device)

    # 6) Process each query
    for query in [q.strip() for q in args.text_queries.split(",")]:
        with torch.no_grad():
            txt_feat = clip.encode(query).squeeze().to(torch.float32)

        similarity_map = torch.einsum("nc,c->n",
            F.normalize(cf3_feat, dim=1),
            F.normalize(txt_feat, dim=0))
        sims = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-9)
        sims = sims.detach().cpu().numpy()

        # A) CF3 visualization
        sel_cf3 = sims > args.threshold
        colors_cf3 = np.zeros((cf3_pc.shape[0],3), dtype=np.uint8)
        colors_cf3[sel_cf3]    = [255, 0,   0]
        colors_cf3[~sel_cf3]   = [  0, 0, 255]
        save_colored_ply(
            cf3_pc.cpu().numpy(),
            colors_cf3,
            os.path.join(args.output_path, f"{query}_on_cf3_vis_{args.threshold}.ply")
        )

        # B) Reference GS visualization
        N = gs_pc.shape[0]
        colors_gs = np.zeros((N,3), dtype=np.uint8)
        sel_gs = np.zeros(N, dtype=bool)
        for cf3_idx, is_sel in enumerate(sel_cf3):
            if is_sel:
                sel_gs[mapping[cf3_idx]] = True
        colors_gs[sel_gs]    = [255, 0,   0]
        colors_gs[~sel_gs]   = [255,255,  0]
        save_colored_ply(
            gs_pc.cpu().numpy(),
            colors_gs,
            os.path.join(args.output_path, f"{query}_on_reference_vis_{args.threshold}.ply")
        )

        # Save selected GS indices
        for cf3_idx, is_sel in enumerate(sel_cf3):
            if is_sel:
                sel_gs_orig[mapping[cf3_idx]] = True

        orig_ply = PlyData.read(args.reference_path)
        orig_vert = orig_ply['vertex']
        data = orig_vert.data
        prop_ply    = PlyData.read(args.property_path)['vertex'].data
        props_list  = list(prop_ply.dtype.names)
        all_orig_props = data.dtype.names
        keep_props     = [p for p in props_list if p in all_orig_props]
        filtered = data[sel_gs_orig]
        new_dtype = [(p, data.dtype.fields[p][0]) for p in keep_props]
        new_data = np.empty(filtered.shape, dtype=new_dtype)
        for p in keep_props:
            new_data[p] = filtered[p]
        new_vert = PlyElement.describe(new_data, name='vertex')
        out = PlyData([new_vert], text=True)
        out.write(os.path.join(args.output_path, "point_cloud.ply"))
        print("Done")


if __name__ == "__main__":
    main()
