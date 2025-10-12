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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.colmap_loader import rotmat2qvec
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, semantic_feature_size: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.feature_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # semantic feature attributes
        self._semantic_feature = torch.empty(0)
        self._semantic_feature_var = torch.empty(0)
        self._global_contribution = torch.empty(0)
        self.semantic_feature_size = semantic_feature_size

        self.compact_feature_field = CompactFeatureField(semantic_feature_size)
        self.set_feature_flag = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_feature, 
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._semantic_feature) = model_args 
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 

    @property
    def get_semantic_feature_var(self):
        return self._semantic_feature_var
    
    @property
    def get_num_gaussians(self):
        return self._xyz.shape[0]
    
    ########################################################

    def rewrite_semantic_feature(self, x):
        self._semantic_feature = x

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], self.semantic_feature_size, 1).float().cuda() 
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._semantic_feature.transpose(1, 2).requires_grad_(False)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def lift_feature_setup(self):

        self._xyz.requires_grad = True
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._opacity.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        self._semantic_feature.requires_grad = False
        
        print("num gaussians: ", self._semantic_feature.shape[0])

    @torch.no_grad()
    def set_semantic_feature(self, accum_contribution, accum_features, accum_features_square = None):

        semantic_weighted_average = accum_features / (accum_contribution[:, None, None] + 1e-9)
        self._semantic_feature = semantic_weighted_average.detach().clone().requires_grad_(False).float().cuda()

        if accum_features_square is not None:
            semantic_weighted_variance = accum_features_square / (accum_contribution[:, None, None] + 1e-9) - semantic_weighted_average ** 2
            self._semantic_feature_var = torch.norm(semantic_weighted_variance, dim=2, keepdim=False).detach().clone().requires_grad_(False).float().cuda() # torch.Size([num_gaussians, 1])

        self._global_contribution = accum_contribution.unsqueeze(1).detach().clone().requires_grad_(False).float().cuda()
        
    def set_compact_feature_field(self):
        
        self.compact_feature_field._xyz = nn.Parameter(self._xyz.detach().clone().requires_grad_(True))
        self.compact_feature_field._scaling = nn.Parameter(self._scaling.detach().clone().requires_grad_(True))
        self.compact_feature_field._rotation = nn.Parameter(self._rotation.detach().clone().requires_grad_(True))
        self.compact_feature_field._opacity = nn.Parameter(self._opacity.detach().clone().requires_grad_(True))
        self.compact_feature_field._semantic_feature_var = self._semantic_feature_var.detach().clone().requires_grad_(False)
        self.compact_feature_field._global_contribution = self._global_contribution.detach().clone().requires_grad_(False)

    def set_opacity(self, alpha):
        if alpha is not torch.Tensor:
            opacity = self.inverse_opacity_activation(alpha * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacity = self.inverse_opacity_activation(alpha)

        self._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(True))

    def detach_setup(self):
        self._xyz = self._xyz.detach().requires_grad_(False)
        self._features_dc = self._features_dc.detach().requires_grad_(False)
        self._features_rest = self._features_rest.detach().requires_grad_(False)
        self._opacity = self._opacity.detach().requires_grad_(False)
        self._scaling = self._scaling.detach().requires_grad_(False)
        self._rotation = self._rotation.detach().requires_grad_(False)
        self._semantic_feature = self._semantic_feature.detach().requires_grad_(False)

    def segment_setup(self, mask3d):

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        print("segment setup: ", mask3d.shape, self._xyz.shape[0])

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))

        if self._semantic_feature.shape[0] > 0:
            semantic_feature_sub = self._semantic_feature[mask3d].detach()
            self._semantic_feature = nn.Parameter(set_requires_grad(semantic_feature_sub, True))

        if self._semantic_feature_var.shape[0] > 0:
            semantic_feature_var_sub = self._semantic_feature_var[mask3d].detach()
            self._semantic_feature_var = nn.Parameter(set_requires_grad(semantic_feature_var_sub, True))

        print("after segmentation: ", xyz_sub.shape[0])

    def prune_setup(self, prune_mask3d):
        # if shape of prune_mask3d is not 1, squeeze it to 1
        if prune_mask3d.dim() != 1:
            prune_mask3d = prune_mask3d.squeeze()

        print(f"prune {(100 * prune_mask3d.sum() / self._xyz.shape[0]):.2f}% of points: {prune_mask3d.sum()} out of {self._xyz.shape[0]}")

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Extracting subsets using the mask
        mask3d = ~prune_mask3d
        
        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))

        if self._semantic_feature.shape[0] > 0:
            semantic_feature_sub = self._semantic_feature[mask3d].detach()
            self._semantic_feature = nn.Parameter(set_requires_grad(semantic_feature_sub, True))

        if self._semantic_feature_var.shape[0] > 0:
            semantic_feature_var_sub = self._semantic_feature_var[mask3d].detach()
            self._semantic_feature_var = nn.Parameter(set_requires_grad(semantic_feature_var_sub, True))

        if self._global_contribution.shape[0] > 0:
            global_contribution_sub = self._global_contribution[mask3d].detach()
            self._global_contribution = nn.Parameter(set_requires_grad(global_contribution_sub, True))

    def alpha_threshold_setup(self, alpha_threshold):

        mask3d = (self.get_opacity > alpha_threshold).squeeze()

        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        semantic_feature_sub = self._semantic_feature[mask3d].detach()

        print("alpha thresholding: ", self._xyz.shape[0], " -> ", xyz_sub.shape[0])

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))
        self._semantic_feature = nn.Parameter(set_requires_grad(semantic_feature_sub, True))

    def transform_setup(self, r, t):

        self._xyz = self._xyz.requires_grad_(True)
        self._rotation = self._rotation.requires_grad_(True)
        self._features_dc = self._features_dc.requires_grad_(False)
        self._features_rest = self._features_rest.requires_grad_(False)
        self._opacity = self._opacity.requires_grad_(True)
        self._scaling = self._scaling.requires_grad_(True)
        self._semantic_feature = self._semantic_feature.requires_grad_(False)

        def quaternion_multiply(r1, r2):
            w1, x1, y1, z1 = r1.unbind(-1)
            w2, x2, y2, z2 = r2.unbind(-1)
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack((w, x, y, z), dim=-1)

        R = build_rotation(r[None, :]).squeeze(0)

        self._xyz = torch.matmul(self._xyz.detach(), R.T) + t
        self._rotation = quaternion_multiply(r, self._rotation.detach())

    def unify_setup(self, GaussianModel):

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)
        
        # concatenate the tensors
        self._xyz = nn.Parameter(set_requires_grad(torch.cat((self._xyz, GaussianModel._xyz), dim=0), True))
        self._features_dc = nn.Parameter(set_requires_grad(torch.cat((self._features_dc, GaussianModel._features_dc), dim=0), True))
        self._features_rest = nn.Parameter(set_requires_grad(torch.cat((self._features_rest, GaussianModel._features_rest), dim=0), True))
        self._opacity = nn.Parameter(set_requires_grad(torch.cat((self._opacity, GaussianModel._opacity), dim=0), True))
        self._scaling = nn.Parameter(set_requires_grad(torch.cat((self._scaling, GaussianModel._scaling), dim=0), True))
        self._rotation = nn.Parameter(set_requires_grad(torch.cat((self._rotation, GaussianModel._rotation), dim=0), True))
        self._semantic_feature = nn.Parameter(set_requires_grad(torch.cat((self._semantic_feature, GaussianModel._semantic_feature), dim=0), True))

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, save_semantic_feature=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        l.append('contribution')

        if save_semantic_feature:
            l.append('feature_var')
            for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):  
                l.append('semantic_{}'.format(i))

        return l
    
    def save_ply(self, path, save_semantic_feature=True):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        contribution = self._global_contribution.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_semantic_feature)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if save_semantic_feature:
            feature_var = self._semantic_feature_var.detach().cpu().numpy() 
            semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, contribution, feature_var, semantic_feature), axis=1) 
            elements[:] = list(map(tuple, attributes))
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, contribution), axis=1)
            elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        print("semantic feature size: ", self.semantic_feature_size)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        if opacities.dtype != np.float64:
            print(f"[INFO] Converting opacities dtype from {opacities.dtype} to float64")
            opacities = opacities.astype(np.float64, copy=False)
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        semantic_feature = np.zeros((xyz.shape[0], self.semantic_feature_size, 1))
        semantic_feature_var = np.zeros((xyz.shape[0], 1))
        contribution = np.zeros((xyz.shape[0], 1))
        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        if count > 0:
            if self.semantic_feature_size != count:
                print("[WARNING] Loaded semantic feature size does not match the model's semantic feature size: ", count)

            semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
            semantic_feature = np.expand_dims(semantic_feature, axis=-1)

            self.set_feature_flag = True

            try :
                semantic_feature_var = np.asarray(plydata.elements[0]["feature_var"])[..., np.newaxis]
            except:
                print("[WARNING] No feature variance found in the ply file")

            try :
                contribution = np.asarray(plydata.elements[0]["contribution"])[..., np.newaxis]
            except:
                print("[WARNING] No contribution found in the ply file")
        else:
            print("[WARNING] Loading pre-trained Original Gaussian Splatting, not Feature 3DGS")

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._semantic_feature_var = torch.tensor(semantic_feature_var, dtype=torch.float, device="cuda").requires_grad_(False)
        self._global_contribution = torch.tensor(contribution, dtype=torch.float, device="cuda").requires_grad_(False)
        self.active_sh_degree = self.max_sh_degree

        compact_feature_field_path = path.replace("point_cloud.ply", "feature_field.ply")
        if os.path.exists(compact_feature_field_path):
            self.compact_feature_field.load_ply(compact_feature_field_path)
        else:
            print("No Feature field PLY is found")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "semantic_feature": new_semantic_feature} 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1,1) 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

# simple Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size=[128, 64, 32, 16]):
        super(Autoencoder, self).__init__()
        # Create encoder layers
        encoder_layers = []
        input_dim = input_size
        for i, dim in enumerate(latent_size):
            encoder_layers.append(nn.Linear(input_dim, dim, bias=False))
            encoder_layers.append(nn.ReLU())
            input_dim = dim
        encoder_layers.append(nn.Linear(input_dim, 3, bias=False))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers
        decoder_layers = []
        input_dim = 3
        for i, dim in enumerate(reversed(latent_size)):
            decoder_layers.append(nn.Linear(input_dim, dim, bias=False))
            decoder_layers.append(nn.ReLU())
            input_dim = dim
        # Final layer to output the original size
        decoder_layers.append(nn.Linear(input_dim, input_size, bias=False))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class CompactFeatureField:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, semantic_feature_size=512):
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.feature_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # semantic feature attributes
        self.semantic_feature_size = semantic_feature_size
        self.autoencoder = None
        self._semantic_feature_var = torch.empty(0)
        self._global_contribution = torch.empty(0)

        self._min_latent_semantic_feature = torch.empty(0)
        self._max_latent_semantic_feature = torch.empty(0)

        # optional mapping code from vanilla GS to 3DGS code
        # self.vanilla_to_semantic_mapping = torch.empty(0)
        # self.vanilla_indices = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args 
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return self._features_dc
    
    def normalize_features(self):
        self._min_latent_semantic_feature = self._features_dc.min(dim=0)[0].squeeze().detach().reshape(1, 1, 3)
        self._max_latent_semantic_feature = self._features_dc.max(dim=0)[0].squeeze().detach().reshape(1, 1, 3)
        self.normalized_features_dc = (self._features_dc - self._min_latent_semantic_feature) / (self._max_latent_semantic_feature - self._min_latent_semantic_feature)

    @property
    def get_normalized_features(self):

        return self.normalized_features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_semantic_feature_var(self):
        return self._semantic_feature_var
    
    @property
    def get_num_gaussians(self):
        return self._xyz.shape[0]
    
    @property
    def get_global_contribution(self):
        return self._global_contribution
    
    @torch.no_grad()
    def set_feature(self, semantic_feature):

        semantic_feature = F.normalize(semantic_feature.squeeze(1), dim=1)
        # latent vector
        features_dc = self.autoencoder.forward(semantic_feature)[1]    # (N, 3)
        print(features_dc)
        
        features_dc = features_dc.reshape(self._xyz.shape[0], 1, 3)
        
        self._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(True))

    # def init_gs_mapping(self): 
    #     self.vanilla_to_semantic_mapping = torch.arange(0, self.get_num_gaussians, device="cuda")
    #     self.vanilla_indices = torch.arange(0, self.get_num_gaussians, device="cuda")

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feature_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def decode_featuremap(self, latent_feature_map):

        # scale to original range
        h, w = latent_feature_map.shape[1], latent_feature_map.shape[2]
        
        # decode latent feature map
        latent_feature_map = latent_feature_map.permute(1, 2, 0).reshape(-1, 3)
        latent_feature_map = latent_feature_map * (self._max_latent_semantic_feature - self._min_latent_semantic_feature) + self._min_latent_semantic_feature
        feature_map = self.autoencoder.decoder(latent_feature_map)
        feature_map = feature_map.reshape(h, w, -1).permute(2, 0, 1)

        return feature_map
    
    def rescale_featuremap(self, latent_feature_map):

        # scale to original range
        h, w = latent_feature_map.shape[1], latent_feature_map.shape[2]
        min = self._min_latent_semantic_feature.reshape(3, 1, 1)
        max = self._max_latent_semantic_feature.reshape(3, 1, 1)
        latent_feature_map = latent_feature_map * (max - min) + min
        
        return latent_feature_map

    def set_autoencoder(self, semantic_feature_size: float, hidden_dims: list, lr: float):
        self.autoencoder = Autoencoder(semantic_feature_size, hidden_dims).to("cuda")
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.autoencoder_scheduler = ExponentialLR(self.autoencoder_optimizer, gamma=0.5)

        self.autoencoder.train()

    def set_opacity(self, alpha):
        if alpha is not torch.Tensor:
            opacity = self.inverse_opacity_activation(alpha * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacity = self.inverse_opacity_activation(alpha)

        self._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(True))


    def detach_setup(self):
        self._xyz = self._xyz.detach().requires_grad_(False)
        self._features_dc = self._features_dc.detach().requires_grad_(False)
        self._opacity = self._opacity.detach().requires_grad_(False)
        self._scaling = self._scaling.detach().requires_grad_(False)
        self._rotation = self._rotation.detach().requires_grad_(False)

    def segment_setup(self, mask3d):

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        print("segment setup: ", mask3d.shape, self._xyz.shape[0])

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))

        if self._semantic_feature_var.shape[0] > 0:
            semantic_feature_var_sub = self._semantic_feature_var[mask3d].detach()
            self._semantic_feature_var = nn.Parameter(set_requires_grad(semantic_feature_var_sub, True))

        if self._global_contribution.shape[0] > 0:
            global_contribution_sub = self._global_contribution[mask3d].detach()
            self._global_contribution = set_requires_grad(global_contribution_sub, False)

        print("after segmentation: ", xyz_sub.shape[0])

    def prune_setup(self, prune_mask3d):
        
        print(f"prune {(100 * prune_mask3d.sum() / self._xyz.shape[0]):.2f}% of points: {prune_mask3d.sum()} out of {self._xyz.shape[0]}")

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Extracting subsets using the mask
        mask3d = ~prune_mask3d.reshape(-1)
        
        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))

        if self._features_dc.shape[0] > 0:
            features_dc_sub = self._features_dc[mask3d].detach()
            self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))

        if self._semantic_feature_var.shape[0] > 0:
            semantic_feature_var_sub = self._semantic_feature_var[mask3d].detach()
            self._semantic_feature_var = nn.Parameter(set_requires_grad(semantic_feature_var_sub, False))

        if self._global_contribution.shape[0] > 0:
            global_contribution_sub = self._global_contribution[mask3d].detach()
            self._global_contribution = set_requires_grad(global_contribution_sub, False)

    def alpha_threshold_setup(self, alpha_threshold):

        mask3d = (self.get_opacity > alpha_threshold).squeeze()

        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        semantic_feature_sub = self._semantic_feature[mask3d].detach()

        print("alpha thresholding: ", self._xyz.shape[0], " -> ", xyz_sub.shape[0])

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, True))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, True))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, True))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, True))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, True))

    def transform_setup(self, r, t):

        self._xyz = self._xyz.requires_grad_(True)
        self._rotation = self._rotation.requires_grad_(True)
        self._features_dc = self._features_dc.requires_grad_(False)
        self._opacity = self._opacity.requires_grad_(True)
        self._scaling = self._scaling.requires_grad_(True)

        def quaternion_multiply(r1, r2):
            w1, x1, y1, z1 = r1.unbind(-1)
            w2, x2, y2, z2 = r2.unbind(-1)
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack((w, x, y, z), dim=-1)

        R = build_rotation(r[None, :]).squeeze(0)

        self._xyz = torch.matmul(self._xyz.detach(), R.T) + t
        self._rotation = quaternion_multiply(r, self._rotation.detach())

    def unify_setup(self, GaussianModel):

        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)
        
        # concatenate the tensors
        self._xyz = nn.Parameter(set_requires_grad(torch.cat((self._xyz, GaussianModel._xyz), dim=0), True))
        self._features_dc = nn.Parameter(set_requires_grad(torch.cat((self._features_dc, GaussianModel._features_dc), dim=0), True))
        self._opacity = nn.Parameter(set_requires_grad(torch.cat((self._opacity, GaussianModel._opacity), dim=0), True))
        self._scaling = nn.Parameter(set_requires_grad(torch.cat((self._scaling, GaussianModel._scaling), dim=0), True))
        self._rotation = nn.Parameter(set_requires_grad(torch.cat((self._rotation, GaussianModel._rotation), dim=0), True))


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # l.append('semantic_var')

        return l
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        semantic_feature_var = self._semantic_feature_var.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # save autoencoder in same directory as autoencoder.pth
        ae_path = os.path.join(os.path.dirname(path), "autoencoder.pth")
        torch.save(self.autoencoder.state_dict(), ae_path)

        semantic_feature_var_path = os.path.join(os.path.dirname(path), "semantic_feature_var.npy")
        np.save(semantic_feature_var_path, semantic_feature_var)

        # # save indices pytorch tensor in npy
        # indices_path = os.path.join(os.path.dirname(path), "vanilla_indices.npy")
        # np.save(indices_path, self.vanilla_indices.cpu().numpy())
        
        # # save color to semantic mapping tensor in npy
        # mapping_path = os.path.join(os.path.dirname(path), "vanilla_to_semantic_mapping.npy")
        # np.save(mapping_path, self.vanilla_to_semantic_mapping.cpu().numpy())

    def save_vis_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_min, f_dc_max = np.percentile(f_dc, [10, 90])
        f_dc = (f_dc - f_dc_min) / (f_dc_max - f_dc_min)

        semantic_feature_var = self._semantic_feature_var.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, semantic_feature_var), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # save autoencoder in same directory as autoencoder.pth
        ae_path = os.path.join(os.path.dirname(path), "autoencoder.pth")
        torch.save(self.autoencoder.state_dict(), ae_path)

        # # save indices pytorch tensor in npy
        # indices_path = os.path.join(os.path.dirname(path), "vanilla_indices.npy")
        # np.save(indices_path, self.vanilla_indices.cpu().numpy())
        
        # # save color to semantic mapping tensor in npy
        # mapping_path = os.path.join(os.path.dirname(path), "vanilla_to_semantic_mapping.npy")
        # np.save(mapping_path, self.vanilla_to_semantic_mapping.cpu().numpy())


    def load_ply(self, path):
        plydata = PlyData.read(path)

        # load autoencoder
        ae_path = path.replace("feature_field.ply", "autoencoder.pth")
        self.autoencoder = Autoencoder(self.semantic_feature_size).to("cuda")
        state_dict = torch.load(ae_path)
        print("Autoencoder INFO: from ", ae_path, state_dict.keys())
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.eval()

        # # save indices pytorch tensor in npy
        # self.vanilla_indices = None
        # indices_path = os.path.join(os.path.dirname(path), "vanilla_indices.npy")
        # if os.path.exists(indices_path):
        #     with open(indices_path, 'rb') as f:
        #         self.vanilla_indices = torch.tensor(np.load(f), dtype=torch.long, device="cuda")
        
        # # save color to semantic mapping tensor in npy
        # self.vanilla_to_semantic_mapping = None
        # mapping_path = os.path.join(os.path.dirname(path), "vanilla_to_semantic_mapping.npy")
        # if os.path.exists(mapping_path):
        #     with open(mapping_path, 'rb') as f:
        #         self.vanilla_to_semantic_mapping = torch.tensor(np.load(f), dtype=torch.long, device="cuda")

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        semantic_feature_var = np.zeros((xyz.shape[0], 1))
        if "semantic_var" in plydata.elements[0]:
            semantic_feature_var = np.asarray(plydata.elements[0]["semantic_var"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_feature_var = torch.tensor(semantic_feature_var, dtype=torch.float, device="cuda").requires_grad_(False)
        self.active_sh_degree = self.max_sh_degree

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, labels = None):
        mask = mask.bool().squeeze()

        valid_points_mask = ~mask
        print(f"prune {mask.sum() / self._xyz.shape[0]*100:.2f}% : {self._xyz.shape[0]} -> {valid_points_mask.sum()}")

        # if labels is not None:
        #     valid = (labels >= 0)
        #     print("prune: ", mask.sum(), labels.shape[0])
        #     prune_original_indices = self.vanilla_indices[mask][valid]
        #     survivor_indices = self.vanilla_indices[labels][valid]
        #     self.vanilla_to_semantic_mapping[prune_original_indices] = survivor_indices
        # self.vanilla_indices = self.vanilla_indices[valid_points_mask]

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.feature_gradient_accum = self.feature_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self._semantic_feature_var.shape[0] > 0:
            semantic_feature_var_sub = self._semantic_feature_var[valid_points_mask].detach()
            self._semantic_feature_var = semantic_feature_var_sub.requires_grad_(False)
            
        if self._global_contribution.shape[0] > 0:
            global_contribution_sub = self._global_contribution[valid_points_mask].detach()
            self._global_contribution = global_contribution_sub.requires_grad_(False)            


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "semantic_feature": new_semantic_feature} 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1,1) 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_merge_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.feature_gradient_accum[update_filter] += torch.norm(self._features_dc.grad[update_filter], dim=-1, keepdim=False) + torch.norm(self._rotation.grad[update_filter], dim=-1, keepdim=True) + torch.norm(self._scaling.grad[update_filter], dim=-1, keepdim=True) + torch.norm(self._opacity.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def merge_postfix(self):
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feature_gradient_accum = torch.zeros((self.get_features.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

