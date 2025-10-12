/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <tuple>
#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

std::tuple<torch::Tensor, torch::Tensor>
knnGraphCUDA(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);

  torch::Tensor knn_indices = torch::full({P, 4}, 0.0, int_opts);
  torch::Tensor knn_dist = torch::full({P, 4}, 0.0, float_opts);
  
  SimpleKNN::knnGraph(P, (float3*)points.contiguous().data<float>(), knn_indices.contiguous().data<int>(), knn_dist.contiguous().data<float>());

  return std::make_tuple(knn_indices, knn_dist);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
mergeKNN(const torch::Tensor& points, const torch::Tensor& cov3d, const torch::Tensor& alpha)
{
  const int P = points.size(0);

  auto int_opts = points.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);

  torch::Tensor merge_indices = torch::full({P, 1}, -1, int_opts);
  torch::Tensor merge_p = torch::full({P, 3}, 0.0, float_opts);
  torch::Tensor merge_cov3d = torch::full({P, 6}, 0.0, float_opts);
  torch::Tensor merge_alpha = torch::full({P, 1}, 0.0, float_opts);
  
  SimpleKNN::mergeKNN(P, (float3*)points.contiguous().data<float>(), cov3d.contiguous().data_ptr<float>(), alpha.contiguous().data<float>(), merge_indices.contiguous().data<int>(), (float3*)merge_p.contiguous().data<float>(), merge_cov3d.contiguous().data<float>(), merge_alpha.contiguous().data<float>());

  return std::make_tuple(merge_indices, merge_p, merge_cov3d, merge_alpha);
}
