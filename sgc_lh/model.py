"""Enhanced E-GCN with dual-view structural perturbation and cross-graph bridge.

Implements:
  - Multi-view structural perturbation (manuscript, "Multi-View Structural
    Perturbation"):
        A~(v) = A . M(v),  M_ij(v) ~ Bernoulli(1 - p_v), symmetric masks
    (edge dropping on the intra-graph adjacency + relation masking, where
    whole relation types are dropped with probability p_v).
  - Cross-graph bridge                  (eq:bridge):
        A_bridge(W~) = [[0, W~], [W~^T, 0]]
  - Symmetric-normalised message passing (eq:message_passing):
        P(v) = D~(v)^(-1/2) A_aug(v) D~(v)^(-1/2),  H(l+1) = sigma(P(v) H(l) W(l))
    The degree matrix is recomputed from the AUGMENTED matrix (paper note:
    recomputing from the perturbed matrix alone would break the symmetry
    normalisation on the bridge entries). Propagation is non-expansive
    (Proposition: spectral non-expansiveness of the normalised operator).
  - Shared encoder weights across the two views; views are perturbed
    independently, so z_i^(1) and z_i^(2) are two stochastic views of the
    same vertex.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def perturb_view(edge_index: torch.Tensor, edge_rel: torch.Tensor, n: int,
                 n_relations: int, p_drop: float) -> torch.Tensor:
    """A~(v) = A . M(v): symmetric edge dropping + relation masking.

    Edge dropping: each edge survives with prob (1 - p_drop), applied
    symmetrically (keep (i,j) iff both (i,j) and (j,i) survive) so the view
    stays symmetric. Relation masking: each relation type is dropped entirely
    with probability p_drop. Returns the sparse perturbed adjacency A~(v).
    """
    m = edge_index.shape[1]
    edge_keep = torch.rand(m, device=edge_index.device) >= p_drop
    if n_relations > 0:
        rel_alive = torch.rand(n_relations, device=edge_index.device) >= p_drop
        edge_keep = edge_keep & rel_alive[edge_rel]

    # enforce symmetry of the kept mask: keep (i,j) iff (j,i) also kept
    key = edge_index[0].long() * n + edge_index[1].long()
    key_rev = edge_index[1].long() * n + edge_index[0].long()
    order = torch.argsort(key_rev)
    pos = torch.searchsorted(key_rev[order], key)
    pos = pos.clamp(max=m - 1)
    rev_alive = edge_keep[order][pos]
    keep = edge_keep & rev_alive & (key_rev[order][pos] == key)

    idx, _ = edge_index[:, keep], None
    vals = torch.ones(idx.shape[1], device=edge_index.device)
    return torch.sparse_coo_tensor(idx, vals, (n, n), device=edge_index.device).coalesce()


def normalise_adj(A_aug: torch.Tensor) -> torch.Tensor:
    """P = D~^(-1/2) A_aug D~^(-1/2), D~ computed FROM A_aug (eq:message_passing)."""
    deg = torch.zeros(A_aug.shape[0], device=A_aug.device)
    deg.scatter_add_(0, A_aug.indices()[0], A_aug.values())
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    idx = A_aug.indices()
    new_vals = A_aug.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, new_vals, A_aug.shape, device=A_aug.device).coalesce()


def sparse_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    idx = torch.cat([A.indices(), B.indices()], dim=1)
    vals = torch.cat([A.values(), B.values()])
    return torch.sparse_coo_tensor(idx, vals, A.shape, device=A.device).coalesce()


class DualViewEGCN(nn.Module):
    """L-layer enhanced E-GCN, weights shared across the two views."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.weights = nn.ModuleList([nn.Linear(dims[l], dims[l + 1], bias=False)
                                      for l in range(n_layers)])
        self.dropout = dropout

    def forward(self, H0: torch.Tensor, edge_index: torch.Tensor, edge_rel: torch.Tensor,
                n: int, n_relations: int, A_bridge: torch.Tensor, p_drop: float):
        """Two independently perturbed views through the shared encoder.

        A_aug(v) = A~(v) + A_bridge(W~); degrees recomputed from A_aug;
        L = len(self.weights) propagation steps per view (eq:message_passing).
        Returns (Z1, Z2), each in R^{n x d'}.
        """
        out = []
        for _view in (0, 1):                       # independent perturbations
            A_tilde = perturb_view(edge_index, edge_rel, n, n_relations, p_drop)
            A_aug = sparse_add(A_tilde, A_bridge)  # symmetric by construction
            P = normalise_adj(A_aug)
            H = H0
            for layer in self.weights:
                H = torch.sparse.mm(P, H) @ layer.weight.T
                H = F.elu(H)                   # sigma (ELU, cf. two-stage E-GCN lineage)
                if self.dropout > 0 and self.training:
                    H = F.dropout(H, self.dropout, self.training)
            out.append(H)
        return out[0], out[1]
