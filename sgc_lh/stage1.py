"""Stage 1: LLM-guided soft homogeneous weight generation (one-off).

Implements, in order:
  1.  Structural Space Partitioning        (eq:margin, eq:delta_range, eq:blurry_set)
      - k-means++ clustering with K centroids on l2-normalised H_str
      - boundary ambiguity coefficient  Delta_i = D(h_i, mu_k') - D(h_i, mu_k*)
      - boundary set  E_blurry(theta) = {e_i : Delta_i < theta}
  2.  Candidate fan-out                    (c nearest in the opposite graph)
  3.  Geometric homogeneous weight         (eq:geo_weight)
        W_ij = exp( -F_metric(h_i^s, h_j^t) ),  l2-normalised embeddings
  4.  LLM-guided semantic soft weight      (eq:llm_weight, via llm.py)
  5.  Affine combination                   (eq:affine_weight)
        W~_ij = (1-alpha) W_ij + alpha W^LLM_ij   (routed pairs only, alpha=0 elsewhere)
  6.  Top-c sparsification of W~           (needed for the sparse bridge, sec:complexity)

The LLM is invoked ONCE here, at initialisation, and never again
(bootstrapping refreshes centroids and seeds only -- see train.py).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans


def l2_normalise(H: np.ndarray) -> np.ndarray:
    """Rows to unit norm; required by eq:delta_range / eq:geo_weight."""
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (H / norms).astype(np.float32)


def structural_partitioning(H_norm: np.ndarray, K: int,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-means++ partitioning and boundary detection (eq:margin, eq:blurry_set).

    Returns (labels, centroids, delta). Delta_i = D(h_i, mu_k') - D(h_i, mu_k*)
    lies in [0, 2] on unit-norm vectors: the Euclidean distance between two
    unit vectors is sqrt(2 - 2 cos) <= 2.
    """
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    labels = km.fit_predict(H_norm)
    centroids = l2_normalise(km.cluster_centers_.astype(np.float32))

    # Euclidean distance to every centroid: ||h||^2 + ||mu||^2 - 2 h.mu = 2 - 2 h.mu
    sim = H_norm @ centroids.T                                 # (n, K)
    dist = np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))        # D in [0, 2]
    part = np.partition(dist, 1, axis=1)
    delta = part[:, 1] - part[:, 0]                            # eq:margin
    return labels.astype(np.int64), centroids, np.maximum(delta, 0.0).astype(np.float32)


def boundary_set(delta: np.ndarray, theta: float) -> np.ndarray:
    """E_blurry(theta) = {e_i : Delta_i < theta}   (eq:blurry_set)."""
    return np.where(delta < theta)[0]


def candidate_pairs(H_norm: np.ndarray, blurry_ids: np.ndarray, n_src: int, n_tgt: int,
                    c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-c nearest candidates in the opposite graph for every boundary entity.

    For each blurry source entity: its c nearest target entities (query).
    For each blurry target entity: its c nearest source entities (query).
    Query count Q = c * |E_blurry|  (Proposition: monotone query budget).

    Returns (pairs, W_geo, routed_from):
        pairs        (Q, 2) int64  -- (src_id, tgt_id)
        W_geo        (Q,)  float32 -- geometric weights exp(-distance)
        routed_from  (Q,)  int64   -- 0 = source entity triggered, 1 = target entity
    """
    Hs, Ht = H_norm[:n_src], H_norm[n_src:]
    sim_st = Hs @ Ht.T                                         # cosine (unit vectors)

    src_mask = blurry_ids < n_src
    tgt_mask = ~src_mask
    blurry_src, blurry_tgt = blurry_ids[src_mask], blurry_ids[tgt_mask] - n_src

    out_pairs, out_from = [], []
    if len(blurry_src) and n_tgt > 0:
        k = min(c, n_tgt)
        top = np.argpartition(-sim_st[blurry_src], k - 1, axis=1)[:, :k]   # (nb, c)
        for row, i in enumerate(blurry_src):
            for j in top[row]:
                out_pairs.append((int(i), int(j)))
                out_from.append(0)
    if len(blurry_tgt) and n_src > 0:
        k = min(c, n_src)
        top = np.argpartition(-sim_st[:, blurry_tgt], k - 1, axis=0)[:k, :]  # (c, nb)
        for col, j in enumerate(blurry_tgt):
            for i in top[:, col]:
                out_pairs.append((int(i), int(j)))
                out_from.append(1)

    if not out_pairs:
        z = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64), z.astype(np.float32), z

    pairs = np.asarray(out_pairs, dtype=np.int64)
    routed_from = np.asarray(out_from, dtype=np.int64)
    d = np.linalg.norm(Hs[pairs[:, 0]] - Ht[pairs[:, 1]], axis=1)
    return pairs, np.exp(-d).astype(np.float32), routed_from   # eq:geo_weight


def fuse_weights(W_geo: np.ndarray, W_llm: np.ndarray | None,
                 routed_mask: np.ndarray | None, alpha: float) -> np.ndarray:
    """W~_ij = (1-alpha) W_ij + alpha W^LLM_ij   (eq:affine_weight).

    Entries that were not routed keep the purely geometric weight
    (alpha set to 0 locally); W^LLM in (0,1) is a confidence score, not a
    calibrated probability (see manuscript discussion).
    """
    W_tilde = W_geo.copy()
    if W_llm is not None and routed_mask is not None and len(routed_mask) == len(W_geo):
        m = np.asarray(routed_mask, dtype=bool)
        if m.any():
            W_tilde[m] = (1.0 - alpha) * W_geo[m] + alpha * np.asarray(W_llm)[m]
    return np.clip(W_tilde, 1e-12, 1.0)


def sparsify_top_c(pairs: np.ndarray, W_tilde: np.ndarray, n_src: int, n_tgt: int,
                   c: int, device: str = "cpu") -> torch.Tensor:
    """Top-c sparsification of W~ into the sparse bridge (sec:complexity).

    Keep the c largest fused weights per source row (duplicates coming from
    source- and target-triggered routing keep the max). Returns A_bridge(W~)
    as a symmetric sparse (n x n) tensor (eq:bridge): zeros on the intra-graph
    diagonal blocks, W~ on the cross-graph block.
    """
    n = n_src + n_tgt
    if len(pairs) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (n, n), device=device
        ).coalesce()

    best: dict[tuple[int, int], float] = {}
    for idx in range(len(pairs)):
        key = (int(pairs[idx, 0]), int(pairs[idx, 1]))
        w = float(W_tilde[idx])
        if w > best.get(key, -1.0):
            best[key] = w

    by_row: dict[int, list[tuple[int, float]]] = {}
    for (i, j), w in best.items():
        by_row.setdefault(i, []).append((j, w))

    rows, cols, vals = [], [], []
    for i, lst in by_row.items():
        lst.sort(key=lambda x: -x[1])
        for j, w in lst[:c]:
            ii, jj = i, j + n_src
            rows += [ii, jj]; cols += [jj, ii]; vals += [w, w]  # symmetric (eq:bridge)

    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, v, (n, n), device=device).coalesce()
