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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
	static void knnGraph(int P, float3* points, int* knn_indices, float* knn_dist);
	static void mergeKNN(int P, float3* points, float* cov3d, float* alphas, int* merge_indices, float3* merge_p, float* merge_cov3d, float* merge_alphas);
};

#endif