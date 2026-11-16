"""Loss functions.

  - ea_loss: margin-based ranking loss over the seed set (eq:ea_loss)
        L_EA = sum_{(e_s, e_t) in S} max(0, gamma
              + ||z_{e_s} - z_{e_t}||^2 - ||z_{e_s'} - z_{e_t'}||^2)
    sampled form: one corrupted counterpart per positive, target replaced by a
    random target entity.
  - infonce: symmetric batch InfoNCE (eq:infonce, eq:gcl_total)
        L_GCL = ( L(1->2) + L(2->1) ) / 2
    denominator runs over the mini-batch only (|B| in-batch entities,
    |B|-1 negatives per direction); cosine similarity; temperature tau.
    Proposition: the batch loss is lower-bounded by
    log(1 + (|B|-1) e^{-2/tau}) > 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ea_loss(Z: torch.Tensor, seeds: torch.Tensor, gamma: float = 1.0,
            n_tgt: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sampled margin ranking loss (eq:ea_loss).

    seeds: (|S|, 2) int tensor of positive pairs (src_idx, tgt_idx) in joint
    indexing (src row, tgt row + n_src offset already applied by caller).
    One corrupted counterpart per positive: e_t replaced by a random target.
    """
    pos = Z[seeds[:, 0]] - Z[seeds[:, 1]]
    d_pos = torch.sum(pos * pos, dim=1)

    n = Z.shape[0]
    if n_tgt is None:
        n_tgt = n
    corrupt = torch.randint(0, n_tgt, (seeds.shape[0],), generator=generator,
                            device=seeds.device) + (n - n_tgt)
    corrupt = torch.where(corrupt == seeds[:, 1],
                          (corrupt + 1 - (n - n_tgt)) % n_tgt + (n - n_tgt),
                          corrupt)                                   # avoid gold target
    neg = Z[seeds[:, 0]] - Z[corrupt]
    d_neg = torch.sum(neg * neg, dim=1)

    return F.relu(gamma + d_pos - d_neg).sum()


def batch_infonce(Z1: torch.Tensor, Z2: torch.Tensor, batch: torch.Tensor,
                  tau: float = 0.1) -> torch.Tensor:
    """Symmetric batch InfoNCE (eq:infonce, eq:gcl_total).

    Z1, Z2: (n, d') outputs of the two views; batch: (|B|,) vertex indices.
    Returns the scalar L_GCL = (L(1->2) + L(2->1)) / 2.
    """
    z1 = F.normalize(Z1[batch], dim=1)
    z2 = F.normalize(Z2[batch], dim=1)
    sim = z1 @ z2.t() / tau                                       # (|B|, |B|)
    labels = torch.arange(sim.shape[0], device=sim.device)
    l12 = F.cross_entropy(sim, labels)                            # -log softmax diag
    l21 = F.cross_entropy(sim.t(), labels)                        # swapped views
    return 0.5 * (l12 + l21)                                      # eq:gcl_total
