# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


try:
    import numpy as np
    import sklearn.neighbors
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    np = None
    sklearn = None
    _HAS_SKLEARN = False
import torch

from threedgrut.utils.misc import to_np


def k_nearest_neighbors(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    if _HAS_SKLEARN:
        x_np = x.cpu().numpy()
        model = sklearn.neighbors.NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
        distances, _ = model.kneighbors(x_np)
        return torch.from_numpy(distances).to(x)
    # Fallback implementation using PyTorch when sklearn/NumPy are unavailable
    chunk_size = 1024
    dists = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        dist = torch.cdist(x[start:end], x)
        k_d, _ = dist.topk(K, largest=False)
        dists.append(k_d)
    return torch.cat(dists, dim=0)


def nearest_neighbors(pts_src: torch.Tensor, k: int = 2) -> torch.Tensor:
    if _HAS_SKLEARN:
        pts_src_np = to_np(pts_src)
        kd_tree = sklearn.neighbors.KDTree(pts_src_np)
        _, neighbors = kd_tree.query(pts_src_np, k=k)
        mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]
        mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
        neighbors = neighbors[mask].reshape((neighbors.shape[0], k - 1))
        neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
        return neigh_inds

    # Fallback pure PyTorch implementation
    chunk_size = 1024
    indices = []
    for start in range(0, pts_src.shape[0], chunk_size):
        end = min(start + chunk_size, pts_src.shape[0])
        dist = torch.cdist(pts_src[start:end], pts_src)
        dist[torch.arange(end - start), start:end] = float('inf')
        _, idx = dist.topk(k - 1, largest=False)
        indices.append(idx)
    return torch.cat(indices, dim=0)


def nearest_neighbor_dist_cpuKD(pts_src, pts_target=None):
    """
    Compute the distance to the nearest neighbor, using a CPU kd-tree
    Passing one arg computes from a point set to itself,
    to args computes distance from each point in src to target
    """

    if _HAS_SKLEARN:
        pts_src_np = to_np(pts_src)

        if pts_target is None:
            on_self = True
            k = 2
            pts_target = pts_src
            pts_target_np = pts_src_np
        else:
            on_self = False
            k = 1
            pts_target_np = to_np(pts_target)

        kd_tree = sklearn.neighbors.KDTree(pts_target_np)
        _, neighbors = kd_tree.query(pts_src_np, k=k)

        if on_self:
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
            neighbors = neighbors[mask].reshape((neighbors.shape[0],))
        else:
            neighbors = neighbors[:, 0]

        neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
        dists = torch.linalg.norm(pts_src - pts_target[neigh_inds, :], dim=-1)
        return dists

    # Fallback using chunked torch.cdist
    if pts_target is None:
        pts_target = pts_src
        on_self = True
    else:
        on_self = False

    chunk_size = 1024
    dists_all = []
    for start in range(0, pts_src.shape[0], chunk_size):
        end = min(start + chunk_size, pts_src.shape[0])
        dist = torch.cdist(pts_src[start:end], pts_target)
        if on_self:
            diag_indices = torch.arange(start, end, device=pts_src.device)
            dist[torch.arange(end - start), diag_indices] = float('inf')
        min_d = dist.min(dim=1).values
        dists_all.append(min_d)
    return torch.cat(dists_all, dim=0)


def safe_normalize(vecs):
    norms = torch.linalg.norm(vecs, dim=-1)
    norms = torch.where(norms > 0.0, norms, 1.0)
    return vecs / norms[..., None]
