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
    parser = argparse.ArgumentParser(description="Run Open-vocab 3D Segmentation visualization pipeline")
    parser.add_argument("--cf3_path", type=str, required=True, help="Path to CF3 root folder containing point_cloud/iteration_33000")
    parser.add_argument("--text_queries", type=str, required=True, help="Comma separated text queries")
    parser.add_argument("--feature_size", type=int, default=512, help="Feature dimension size")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors for mapping")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for similarity")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use (CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


# ----------------- Utility: save PLY -----------------
def save_colored_ply(xyz: np.ndarray, colors: np.ndarray, out_path: str):
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
        enc, d = [], input_size
        for h in latent_size:
            enc += [nn.Linear(d,h,bias=False), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d,3,bias=False))
        self.encoder = nn.Sequential(*enc)

        dec, d = [], 3
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive internal paths
    ae_path        = os.path.join(args.cf3_path, "point_cloud/iteration_33000/autoencoder.pth")
    cf3_ply_path   = os.path.join(args.cf3_path, "point_cloud/iteration_33000/feature_field.ply")
    reference_path = os.path.join(args.cf3_path, "point_cloud/iteration_33000/point_cloud.ply")
    output_path    = os.path.join(args.cf3_path, "3d_seg")
    os.makedirs(output_path, exist_ok=True)

    # 1) Load AE
    ae = Autoencoder(args.feature_size).to(device)
    ae.load_state_dict(torch.load(ae_path))
    ae.eval()

    # 2) Load CF3 point cloud & decode features
    ply = PlyData.read(cf3_ply_path)['vertex'].data
    arr = np.vstack([ply[p] for p in ['x','y','z','f_dc_0','f_dc_1','f_dc_2']]).T
    cf3 = torch.from_numpy(arr).float().to(device)
    cf3_pc, cf3_feat_3d = cf3[:, :3], cf3[:, 3:]
    with torch.no_grad():
        cf3_feat = ae.decoder(cf3_feat_3d)

    # 3) Load reference GS point cloud
    ply = PlyData.read(reference_path)['vertex'].data
    props = ['x','y','z'] + [f'semantic_{i}' for i in range(args.feature_size)]
    arr = np.vstack([ply[p] for p in props]).T
    gs = torch.from_numpy(arr).float().to(device)
    gs_pc, gs_feature = gs[:, :3], gs[:, 3:]


    all_props = ply.dtype.names
    exclude_keywords = ("semantic_", "contribution", "feature_var")
    keep_props = [p for p in all_props if not any(p.startswith(k) or p == k for k in exclude_keywords)]
    
    # Remove NaN
    mask = ~torch.isnan(gs_pc).any(dim=1)
    gs_pc, gs_feature = gs_pc[mask], gs_feature[mask]

    # 4) Build GS→CF3 mapping
    cf3_norm = F.normalize(cf3_feat, dim=1).cpu().numpy()
    gs_norm  = F.normalize(gs_feature, dim=1).cpu().numpy()
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

    # 5) CLIP text encoder
    clip = OpenCLIPTextEncoder(device)

    # 6) Process queries
    for query in [q.strip() for q in args.text_queries.split(",")]:
        with torch.no_grad():
            txt_feat = clip.encode(query).squeeze().float()

        sims = torch.einsum("nc,c->n", F.normalize(cf3_feat, dim=1), F.normalize(txt_feat, dim=0))
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
        sims = sims.cpu().numpy()

        # CF3 visualization
        sel_cf3 = sims > args.threshold
        colors_cf3 = np.zeros((cf3_pc.shape[0],3), dtype=np.uint8)
        colors_cf3[sel_cf3] = [255,0,0]
        colors_cf3[~sel_cf3] = [0,0,255]
        save_colored_ply(cf3_pc.cpu().numpy(), colors_cf3, os.path.join(output_path, f"{query}_on_cf3_vis_{args.threshold}.ply"))

        # Reference GS visualization
        N = gs_pc.shape[0]
        colors_gs = np.zeros((N,3), dtype=np.uint8)
        sel_gs = np.zeros(N, dtype=bool)
        for cf3_idx, is_sel in enumerate(sel_cf3):
            if is_sel:
                sel_gs[mapping[cf3_idx]] = True
        colors_gs[sel_gs] = [255,0,0]
        colors_gs[~sel_gs] = [255,255,0]
        save_colored_ply(gs_pc.cpu().numpy(), colors_gs, os.path.join(output_path, f"{query}_on_reference_vis_{args.threshold}.ply"))


        filtered = ply[sel_gs]
        new_dtype = [(p, ply.dtype.fields[p][0]) for p in keep_props]
        new_data = np.empty(filtered.shape, dtype=new_dtype)
        for p in keep_props:
            new_data[p] = filtered[p]

        new_vert = PlyElement.describe(new_data, name='vertex')
        out = PlyData([new_vert], text=False)

        save_file = os.path.join(output_path, f"{query}_3dgs.ply")
        out.write(save_file)
        print(f"[Done] Saved selected GS subset for query '{query}' → {save_file}")

if __name__ == "__main__":
    main()
