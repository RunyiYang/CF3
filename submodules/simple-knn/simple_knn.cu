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

#define BOX_SIZE 1024

#include <cfloat>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}



// Structure to hold distance and index pair
struct DistIdx
{
    float dist;
    int idx;
};

template<int K>
__device__ void updateKBestDistIdx(const float3& ref, const float3& point, int point_idx, DistIdx* knn)
{
    float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
    float dist = d.x * d.x + d.y * d.y + d.z * d.z; // Squared Euclidean distance
    for (int j = 0; j < K; j++)
    {
        if (knn[j].dist > dist)
        {
            DistIdx t = knn[j];
            knn[j] = { dist, point_idx };
            dist = t.dist;
            point_idx = t.idx;
        }
    }
}

// Kernel that outputs distances and neighbor indices only
__global__ void boxKNNGraph(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, int* knn_indices, float* knn_dist)
{
    int idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 point = points[indices[idx]];
    DistIdx best[4] = { {FLT_MAX, -1}, {FLT_MAX, -1}, {FLT_MAX, -1} }; // Hardcoded K=3

    // Initial search in a small local window
    for (int i = max(0, idx - 4); i <= min(P - 1, idx + 4); i++)
    {
        if (i == idx)
            continue;
		updateKBestDistIdx<4>(point, points[indices[i]], indices[i], best);
    }

    float reject = best[3].dist;  // Worst of the 4 best distances

    // Search across all boxes
    for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
    {
        MinMax box = boxes[b];
        float dist = distBoxPoint(box, point);
        if (dist > reject || dist > best[2].dist)
            continue;

        for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
        {
            if (i == idx)
                continue;
			updateKBestDistIdx<4>(point, points[indices[i]], indices[i], best);
        }
    }

    // Store results
    int orig_idx = indices[idx];
    for (int k = 0; k < 4; k++) {
        knn_indices[orig_idx * 4 + k] = best[k].idx;
        knn_dist[orig_idx * 4 + k] = best[k].dist; // Store squared distance
    }
}

void SimpleKNN::knnGraph(int P, float3* points, int* knn_indices, float* knn_dist)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxKNNGraph << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), knn_indices, knn_dist);

	cudaFree(result);
}


// New structure to store candidate information including per-axis differences.
struct DistIdxXYZ {
    float dist; // Squared Euclidean distance (used only for candidate ordering)
    int idx;
    float dx;
    float dy;
    float dz;
};

struct Cov3D {
    float cxx;
    float cxy;
    float cxz;
    float cyy;
    float cyz;
    float czz;
};

// Helper: Compute cosine similarity between two float3 vectors.
__device__ float cosine_similarity(const float3 a, const float3 b) {
    float dot = a.x * b.x + a.y * b.y + a.z * b.z;
    float norm_a = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    float norm_b = sqrtf(b.x * b.x + b.y * b.y + b.z * b.z);
    return (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0f;
}

__device__ Cov3D addOuterProduct(const Cov3D& cov, const float3& mu)
{
    Cov3D out = cov;
    // Add µ µ^T to the 3×3
    out.cxx += mu.x * mu.x;
    out.cyy += mu.y * mu.y;
    out.czz += mu.z * mu.z;
    out.cxy += mu.x * mu.y;
    out.cxz += mu.x * mu.z;
    out.cyz += mu.y * mu.z;
    return out;
}

__device__ Cov3D subtractOuterProduct(const Cov3D& cov, const float3& mu)
{
    Cov3D out = cov;
    out.cxx -= mu.x * mu.x;
    out.cyy -= mu.y * mu.y;
    out.czz -= mu.z * mu.z;
    out.cxy -= mu.x * mu.y;
    out.cxz -= mu.x * mu.z;
    out.cyz -= mu.y * mu.z;
    return out;
}

__device__ float determinant(const Cov3D &cov) {
    return cov.cxx * (cov.cyy * cov.czz - cov.cyz * cov.cyz) -
           cov.cxy * (cov.cxy * cov.czz - cov.cxz * cov.cyz) +
           cov.cxz * (cov.cxy * cov.cyz - cov.cxz * cov.cyy);
}

__device__ Cov3D inverse(const Cov3D &cov) {
    Cov3D inv;
    float det = determinant(cov);
    inv.cxx =  (cov.cyy * cov.czz - cov.cyz * cov.cyz) / det;
    inv.cxy = -(cov.cxy * cov.czz - cov.cxz * cov.cyz) / det;
    inv.cxz =  (cov.cxy * cov.cyz - cov.cxz * cov.cyy) / det;
    inv.cyy =  (cov.cxx * cov.czz - cov.cxz * cov.cxz) / det;
    inv.cyz = -(cov.cxx * cov.cyz - cov.cxz * cov.cxy) / det;
    inv.czz =  (cov.cxx * cov.cyy - cov.cxy * cov.cxy) / det;
    return inv;
}

// __device__ float computeBhattacharyyaDistance(const Cov3D &cov0, const Cov3D &cov1, const float3 &diff) {
//     // Compute the average covariance: Sigma = (cov0 + cov1)/2.
//     Cov3D Sigma;
//     Sigma.cxx = 0.5f * (cov0.cxx + cov1.cxx);
//     Sigma.cxy = 0.5f * (cov0.cxy + cov1.cxy);
//     Sigma.cxz = 0.5f * (cov0.cxz + cov1.cxz);
//     Sigma.cyy = 0.5f * (cov0.cyy + cov1.cyy);
//     Sigma.cyz = 0.5f * (cov0.cyz + cov1.cyz);
//     Sigma.czz = 0.5f * (cov0.czz + cov1.czz);
    
//     // First term: (1/8)*diff^T*inv(Sigma)*diff
//     Cov3D invSigma = inverse(Sigma);
//     float term1 = (1.0f / 8.0f) * (
//         diff.x * (invSigma.cxx * diff.x + invSigma.cxy * diff.y + invSigma.cxz * diff.z) +
//         diff.y * (invSigma.cxy * diff.x + invSigma.cyy * diff.y + invSigma.cyz * diff.z) +
//         diff.z * (invSigma.cxz * diff.x + invSigma.cyz * diff.y + invSigma.czz * diff.z)
//     );
    
//     // Second term: 0.5 * ln( det(Sigma) / sqrt(det(cov0)*det(cov1)) )
//     float detSigma = determinant(Sigma);
//     float det0 = determinant(cov0);
//     float det1 = determinant(cov1);
//     float term2 = 0.5f * logf(detSigma / sqrtf(det0 * det1));
    
//     return term1 + term2;
// }

// Kernel that builds kNN candidates (storing dx, dy, dz) and then performs matching
// if, for a candidate neighbor, the absolute differences on x, y, and z are all below
// specified thresholds.
__global__ void boxMergeKNNKernel(
    uint32_t P,
    float3* __restrict__ points,
    uint32_t* __restrict__ indices, // Morton-sorted indices
    MinMax* __restrict__ boxes,
    float* __restrict__ cov3d,
    float* __restrict__ inv_cov3d,
    float* __restrict__ alphas,         // Input opacity per Gaussian
    int* __restrict__ merge_indices,    // Output: matched partner index per point (-1 if unmatched)
    float3* __restrict__ merge_p,       // Output: merged mean for each point
    float* __restrict__ merge_cov3d,     // Output: merged covariance for each point
    float* __restrict__ merge_alphas,     // Output: merged opacity for each point
    float merge_thresh,                   // Threshold for absolute difference in each axis
    float parallel_thresh                 // Threshold for cosine similarity of covariance vectors
) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    // Work on the point corresponding to indices[idx]
    int myIdx = indices[idx];
    float3 p = points[myIdx];
    float my_alpha = alphas[myIdx];
    int base = myIdx * 6;
    // Retrieve projected variances from cov3d
    Cov3D my_cov;
    my_cov.cxx = cov3d[base + 0];
    my_cov.cxy = cov3d[base + 1];
    my_cov.cxz = cov3d[base + 2];
    my_cov.cyy = cov3d[base + 3];
    my_cov.cyz = cov3d[base + 4];
    my_cov.czz = cov3d[base + 5];

    // Initialize outputs: unmatched (-1) and original values.
    merge_indices[myIdx] = -1;
    merge_p[myIdx] = p;
    merge_alphas[myIdx] = my_alpha;

    // Use K = 4 candidate neighbors.
    const int K = 4;
    DistIdxXYZ best[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        best[k].dist = FLT_MAX;
        best[k].idx = -1;
        best[k].dx = 0.0f;
        best[k].dy = 0.0f;
        best[k].dz = 0.0f;
    }

    // ---- First Pass: Local Window Search ----
    int localWindow = 4;
    int start = max(0, idx - localWindow);
    int end   = min((int)P - 1, idx + localWindow);
    for (int i = start; i <= end; i++) {
        if (i == idx) continue;
        int candidate = indices[i];
        float3 q = points[candidate];
        float dx = p.x - q.x;
        float dy = p.y - q.y;
        float dz = p.z - q.z;
        float d2 = dx * dx + dy * dy + dz * dz;
        // Insert candidate into best[] if closer than an existing candidate.
        for (int j = 0; j < K; j++) {
            if (d2 < best[j].dist) {
                DistIdxXYZ temp = best[j];
                best[j].dist = d2;
                best[j].idx = candidate;
                best[j].dx = dx;
                best[j].dy = dy;
                best[j].dz = dz;
                d2 = temp.dist;
            }
        }
    }

    // ---- Second Pass: Search in Local Boxes ----
    int num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    for (int b = 0; b < num_boxes; b++) {
        MinMax box = boxes[b];
        float boxDist = distBoxPoint(box, p);
        if (boxDist > best[K - 1].dist)
            continue;
        int box_start = b * BOX_SIZE;
        int box_end = min((uint32_t)P, (b + 1) * BOX_SIZE);
        for (int i = box_start; i < box_end; i++) {
            if (i == idx) continue;
            int candidate = indices[i];
            float3 q = points[candidate];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dz = p.z - q.z;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > best[K - 1].dist)
                continue;
            for (int j = 0; j < K; j++) {
                if (d2 < best[j].dist) {
                    DistIdxXYZ temp = best[j];
                    best[j].dist = d2;
                    best[j].idx = candidate;
                    best[j].dx = dx;
                    best[j].dy = dy;
                    best[j].dz = dz;
                    d2 = temp.dist;
                }
            }
        }
    }

    // ---- Matching: Use simple per-axis criterion.
    for (int k = 0; k < K; k++) {
        int neighbor = best[k].idx;
        if (neighbor < 0)
            continue;

        // Retrieve neighbor's projected covariance.
        int base_neighbor = neighbor * 6;
        Cov3D neighbor_cov;
        neighbor_cov.cxx = cov3d[base_neighbor + 0];
        neighbor_cov.cxy = cov3d[base_neighbor + 1];
        neighbor_cov.cxz = cov3d[base_neighbor + 2];
        neighbor_cov.cyy = cov3d[base_neighbor + 3];
        neighbor_cov.cyz = cov3d[base_neighbor + 4];
        neighbor_cov.czz = cov3d[base_neighbor + 5];

        float3 neighbor_p = points[neighbor];

        // Compute Mahalanobis distance squared
        float d2 = best[k].dx * (inv_cov3d[base + 1]*best[k].dx + inv_cov3d[base + 2]*best[k].dy + inv_cov3d[base + 3]*best[k].dz) +
                    best[k].dy * (inv_cov3d[base + 2]*best[k].dx + inv_cov3d[base + 4]*best[k].dy + inv_cov3d[base + 5]*best[k].dz) +
                    best[k].dz * (inv_cov3d[base + 3]*best[k].dx + inv_cov3d[base + 5]*best[k].dy + inv_cov3d[base + 6]*best[k].dz);

        
        float d2_neighbor = best[k].dx * (inv_cov3d[base_neighbor + 1]*best[k].dx + inv_cov3d[base_neighbor + 2]*best[k].dy + inv_cov3d[base_neighbor + 3]*best[k].dz) +
                            best[k].dy * (inv_cov3d[base_neighbor + 2]*best[k].dx + inv_cov3d[base_neighbor + 4]*best[k].dy + inv_cov3d[base_neighbor + 5]*best[k].dz) +
                            best[k].dz * (inv_cov3d[base_neighbor + 3]*best[k].dx + inv_cov3d[base_neighbor + 5]*best[k].dy + inv_cov3d[base_neighbor + 6]*best[k].dz);

        if ((d2 < merge_thresh) && (d2_neighbor < merge_thresh)) {

            // Try to claim a match using atomicCAS (to ensure each point is matched only once).
            int expected = -1;
            if (atomicCAS(&merge_indices[myIdx], expected, neighbor) == expected) {
                expected = -1;
                if (atomicCAS(&merge_indices[neighbor], expected, myIdx) == expected) {
                    // Weighted merge of covariances using opacities.
                    float neighbor_alpha = alphas[neighbor];
                    float total_alpha = my_alpha + neighbor_alpha;

                    float3 merged_mean;
                    merged_mean.x = (my_alpha * p.x + neighbor_alpha * neighbor_p.x) / total_alpha;
                    merged_mean.y = (my_alpha * p.y + neighbor_alpha * neighbor_p.y) / total_alpha;
                    merged_mean.z = (my_alpha * p.z + neighbor_alpha * neighbor_p.z) / total_alpha;

                    merge_p[myIdx] = merged_mean;
                    merge_p[neighbor] = merged_mean;

                    Cov3D M1 = addOuterProduct(my_cov, p);
                    Cov3D M2 = addOuterProduct(neighbor_cov, neighbor_p);
                    Cov3D sum_M;
                    sum_M.cxx = (my_alpha * M1.cxx + neighbor_alpha * M2.cxx) / total_alpha;
                    sum_M.cyy = (my_alpha * M1.cyy + neighbor_alpha * M2.cyy) / total_alpha;
                    sum_M.czz = (my_alpha * M1.czz + neighbor_alpha * M2.czz) / total_alpha;
                    sum_M.cxy = (my_alpha * M1.cxy + neighbor_alpha * M2.cxy) / total_alpha;
                    sum_M.cxz = (my_alpha * M1.cxz + neighbor_alpha * M2.cxz) / total_alpha;
                    sum_M.cyz = (my_alpha * M1.cyz + neighbor_alpha * M2.cyz) / total_alpha;

                    Cov3D merged_cov = subtractOuterProduct(sum_M, merged_mean);

                    // Store the merged covariance.
                    int base = myIdx * 6;
                    merge_cov3d[base + 0] = merged_cov.cxx;
                    merge_cov3d[base + 1] = merged_cov.cxy;
                    merge_cov3d[base + 2] = merged_cov.cxz;
                    merge_cov3d[base + 3] = merged_cov.cyy;
                    merge_cov3d[base + 4] = merged_cov.cyz;
                    merge_cov3d[base + 5] = merged_cov.czz;

                    int base_neighbor = neighbor * 6;
                    merge_cov3d[base_neighbor + 0] = merged_cov.cxx;
                    merge_cov3d[base_neighbor + 1] = merged_cov.cxy;
                    merge_cov3d[base_neighbor + 2] = merged_cov.cxz;
                    merge_cov3d[base_neighbor + 3] = merged_cov.cyy;
                    merge_cov3d[base_neighbor + 4] = merged_cov.cyz;
                    merge_cov3d[base_neighbor + 5] = merged_cov.czz;

                    // Merge opacities
                    float merged_alpha = my_alpha + neighbor_alpha * (1.0f - my_alpha);

                    // Alternative merge opacities
                    // float merged_alpha = min(my_alpha + neighbor_alpha, 0.9f);
                    // float merged_alpha = my_alpha + neighbor_alpha;
                    // if (merged_alpha > 1.0f) {
                    //     merged_alpha -= my_alpha * neighbor_alpha;
                    // }

                    merge_alphas[myIdx] = merged_alpha;
                    merge_alphas[neighbor] = merged_alpha;
                    break; // Stop after a successful match.
                } else {
                    merge_indices[myIdx] = -1; // Revert if neighbor was already matched.
                }
            }
        }
    }
}

// Each covariance matrix is stored as 6 floats in the order:
// [ cxx, cxy, cxz, cyy, cyz, czz ]
// representing the symmetric matrix:
// [ cxx, cxy, cxz ]
// [ cxy, cyy, cyz ]
// [ cxz, cyz, czz ]
__global__ void invertCovarianceKernel(const int P, const float *cov3d, float *inv_cov3d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    int base = idx * 6;
    // Read the symmetric matrix components
    float a = cov3d[base + 0]; // cxx
    float b = cov3d[base + 1]; // cxy
    float c = cov3d[base + 2]; // cxz
    float d = cov3d[base + 3]; // cyy
    float e = cov3d[base + 4]; // cyz
    float f = cov3d[base + 5]; // czz

    // Compute the determinant of the 3x3 matrix:
    // det = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d)
    float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    
    // Check for singularity (or near-singularity)
    if (fabsf(det) < 1e-6f) {
        // If the matrix is singular, set the inverse to zero (or handle as needed)
        inv_cov3d[base + 0] = 0.0f;
        inv_cov3d[base + 1] = 0.0f;
        inv_cov3d[base + 2] = 0.0f;
        inv_cov3d[base + 3] = 0.0f;
        inv_cov3d[base + 4] = 0.0f;
        inv_cov3d[base + 5] = 0.0f;
        return;
    }
    float invDet = 1.0f / det;

    // Compute the elements of the inverse matrix:
    // inv.cxx = (d*f - e*e) / det
    // inv.cxy = (c*e - b*f) / det
    // inv.cxz = (b*e - c*d) / det
    // inv.cyy = (a*f - c*c) / det
    // inv.cyz = (b*c - a*e) / det
    // inv.czz = (a*d - b*b) / det
    inv_cov3d[base + 0] = (d * f - e * e) * invDet;
    inv_cov3d[base + 1] = (c * e - b * f) * invDet;
    inv_cov3d[base + 2] = (b * e - c * d) * invDet;
    inv_cov3d[base + 3] = (a * f - c * c) * invDet;
    inv_cov3d[base + 4] = (b * c - a * e) * invDet;
    inv_cov3d[base + 5] = (a * d - b * b) * invDet;
}

// ---------------------------------------------------------------------
// Host function: Follows the structure of knnGraph. It computes the global
// bounding box (using CUB reductions), builds Morton codes and sorted indices,
// partitions points into local boxes, and launches the merge kernel.
// This version uses a simple per-axis criterion by comparing absolute differences.
void SimpleKNN::mergeKNN(
    int P,
    float3* points,
    float* cov3d,
    float* alphas,         // Input opacity for each Gaussian
    int* merge_indices,    // Output: array of P ints (matched partner index or -1)
    float3* merge_p,       // Output: array of P float3's (merged mean per point)
    float* merge_cov3d,    // Output: array of P float3's (merged covariance per point)
    float* merge_alphas    // Output: array of P floats (merged opacity per point)
) {
    // --- Compute global bounding box via CUB reductions ---
    float3* result;
    cudaMalloc(&result, sizeof(float3));
    size_t temp_storage_bytes;
    float3 init = {0, 0, 0}, minn, maxx;
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
    cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
    cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // --- Compute Morton codes for each point ---
    thrust::device_vector<uint32_t> morton(P);
    thrust::device_vector<uint32_t> morton_sorted(P);
    coord2Morton<<<(P + 255) / 256, 256>>>(P, points, minn, maxx, morton.data().get());

    // --- Build an index array and sort it by Morton codes ---
    thrust::device_vector<uint32_t> indices(P);
    thrust::sequence(indices.begin(), indices.end());
    thrust::device_vector<uint32_t> indices_sorted(P);
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    morton.data().get(), morton_sorted.data().get(),
                                    indices.data().get(), indices_sorted.data().get(), P);
    temp_storage.resize(temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes,
                                    morton.data().get(), morton_sorted.data().get(),
                                    indices.data().get(), indices_sorted.data().get(), P);

    // --- Partition points into local boxes ---
    uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    thrust::device_vector<MinMax> boxes(num_boxes);
    boxMinMax<<<num_boxes, BOX_SIZE>>>(P, points, indices_sorted.data().get(), boxes.data().get());

    // --- Preprocess: Invert each covariance matrix ---
    float* inv_cov3d;
    cudaMalloc(&inv_cov3d, P * 6 * sizeof(float));
    const int inv_threads = 256;
    const int inv_blocks = (P + inv_threads - 1) / inv_threads;
    invertCovarianceKernel<<<inv_blocks, inv_threads>>>(P, cov3d, inv_cov3d);
    cudaDeviceSynchronize(); // Ensure inversion completes (optional error checking)

    // --- Launch the merge kernel ---
    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    // Define per-axis thresholds for merging. Adjust these values as needed.
    float merge_threshold = 2.38f;
    float parallel_thresh = 0.5f;
    
    boxMergeKNNKernel<<<blocks, threads>>>(
         P,
         points,
         indices_sorted.data().get(),
         boxes.data().get(),
         cov3d,
         inv_cov3d,
         alphas,
         merge_indices,
         (float3*)merge_p,
         merge_cov3d, // Cast merge_cov3d from float* to float3*
         merge_alphas,
         merge_threshold,
         parallel_thresh
    );

    cudaFree(result);
}

