"""Evaluation metrics: Hits@k and MRR (eq:metrics).

For a source entity with gold target e_t, rho(e^s) is the rank of e_t among
ALL target entities under cosine similarity of the learned embeddings.
    Hits@k = (1/|M_test|) sum 1[rho <= k]
    MRR    = (1/|M_test|) sum 1/rho

Indexing contract: `test_pairs` uses LOCAL ids (source id in [0, n_src),
target id in [0, n_tgt)), as distributed by the benchmarks. `Z` uses joint
indexing (source rows [0, n_src), target rows [n_src, n)).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def hits_and_mrr(Z: torch.Tensor, test_pairs: np.ndarray, n_src: int,
                 batch_size: int = 2048, ks: tuple[int, ...] = (1, 10)) -> dict:
    """Full-split ranking evaluation (ranking over the entire target graph)."""
    Zs = F.normalize(Z[:n_src], dim=1)
    Zt = F.normalize(Z[n_src:], dim=1)
    n_tgt = Z.shape[0] - n_src

    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    n_test = len(test_pairs)

    for s in range(0, n_test, batch_size):
        chunk = test_pairs[s:s + batch_size]
        src_idx = torch.as_tensor(chunk[:, 0], dtype=torch.long, device=Z.device)
        tgt_idx = torch.as_tensor(chunk[:, 1], dtype=torch.long, device=Z.device).clamp(0, n_tgt - 1)

        sims = Zs[src_idx] @ Zt.t()                               # (b, n_tgt)
        gold_sim = sims.gather(1, tgt_idx.unsqueeze(1))
        rank = (sims > gold_sim).sum(dim=1) + 1                   # 1-based rank of gold
        rank = rank.float().cpu().numpy()

        for k in ks:
            hits[k] += int((rank <= k).sum())
        mrr_sum += float((1.0 / rank).sum())

    return {
        "hits@1": hits.get(1, 0) / n_test,
        "hits@10": hits.get(10, 0) / n_test,
        "mrr": mrr_sum / n_test,
    }


@torch.no_grad()
def bootstrap_ci(metric_values: list[float], n_resamples: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for a per-seed metric sequence
    (paired bootstrap with 10,000 resamples in the manuscript protocol)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(metric_values, dtype=np.float64)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
