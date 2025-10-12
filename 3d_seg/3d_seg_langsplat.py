#!/usr/bin/env python
"""
Usage:
    python langsplat_script.py \
        --property_path /path/to/property.ply \
        --ae_path /path/to/autoencoder.pth \
        --langsplat_path /path/to/langsplat.ply \
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
    parser = argparse.ArgumentParser(description="Run LangSplat visualization pipeline with AE + CLIP")
    parser.add_argument("--property_path", type=str, required=True, help="PLY file path used to define property names")
    parser.add_argument("--ae_path", type=str, required=True, help="Trained Autoencoder .pth file path")
    parser.add_argument("--langsplat_path", type=str, required=True, help="LangSplat PLY file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output folder to save results")
    parser.add_argument("--text_queries", type=str, required=True, help="Comma separated text queries")
    parser.add_argument("--feature_size", type=int, default=512, help="Feature dimension size")
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
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


# ----------------- Main -----------------
def main():
    args = get_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure output folder exists
    os.makedirs(args.output_path, exist_ok=True)

    # 1) Load Autoencoder
    ae = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(device)
    ae.load_state_dict(torch.load(args.ae_path))
    ae.eval()

    # 2) Load LangSplat point cloud & decode features
    ply = PlyData.read(args.langsplat_path)['vertex'].data
    arr = np.vstack([ply[p] for p in ['x','y','z','semantic_0','semantic_1','semantic_2']]).T
    lsp = torch.from_numpy(arr).float().to(device)
    lsp_pc       = lsp[:, :3]
    lsp_feat_3d  = lsp[:, 3:]
    lsp_feat_3d  = lsp_feat_3d / (lsp_feat_3d.norm(dim=-1, keepdim=True) + 1e-9)

    sel_gs = np.zeros(lsp_pc.shape[0], dtype=bool)
    with torch.no_grad():
        lsp_feat = ae.decode(lsp_feat_3d)

    # 3) Normalize features
    lsp_norm = F.normalize(lsp_feat, dim=1).detach().cpu().numpy()

    # 4) Prepare CLIP text encoder
    clip = OpenCLIPTextEncoder(device)

    # 5) Process each query
    for query in [q.strip() for q in args.text_queries.split(",")]:
        with torch.no_grad():
            txt_feat = clip.encode(query).squeeze().to(torch.float32)

        similarity_map = torch.einsum("nc,c->n",
            F.normalize(lsp_feat, dim=1),
            F.normalize(txt_feat, dim=0))
        sims = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-9)
        sims = sims.detach().cpu().numpy()

        # A) LangSplat: red = selected, blue = non-selected
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
