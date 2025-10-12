import os
import torch
import numpy as np
from glob import glob

def process_frame(s_file, f_file, sam_level):
    seg_map = torch.from_numpy(np.load(s_file))      
    feature_map = torch.from_numpy(np.load(f_file))      
    
    _, h, w = seg_map.shape
    
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    seg = seg_map[:, y, x].squeeze(-1).long()
    
    mask = seg != -1
    
    point_feature1 = feature_map[seg[sam_level:sam_level+1]].squeeze(0)
    
    mask = mask[sam_level:sam_level+1].reshape(1, h, w)
    
    point_feature = point_feature1.reshape(h, w, -1).permute(2, 0, 1)
    
    point_feature = point_feature * mask.to(point_feature.dtype)
    
    point_feature = point_feature.half()
    
    return point_feature

def main(input_folder, output_folder, sam_level):
    os.makedirs(output_folder, exist_ok=True)
    
    s_files = sorted(glob(os.path.join(input_folder, '*_s.npy')))
    
    for s_file in s_files:
        f_file = s_file.replace('_s.npy', '_f.npy')
        if not os.path.exists(f_file):
            print(f"Skipping {s_file}: corresponding _f.npy not found.")
            continue
        
        base_name = os.path.basename(s_file)
        frame_base = base_name.split('_s.npy')[0]
        
        point_feature = process_frame(s_file, f_file, sam_level)
        
        out_file = os.path.join(output_folder, f"{frame_base}_fmap_CxHxW.pt")
        torch.save(point_feature, out_file)
        print(f"Saved point_feature to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert segmentation and feature numpy files to point feature tensors.")
    parser.add_argument("input_folder", type=str, help="input folder (here xxx_s.npy and xxx_f.npy files are located)")
    parser.add_argument("output_folder", type=str, help="output folder (here xxx_fmap_CxHxW.pt files will be saved)")
    parser.add_argument("sam_level", type=int, help="sam_level (0, 1, 2, 3)")
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.sam_level)