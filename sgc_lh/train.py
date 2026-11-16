"""End-to-end SGC-LH training pipeline.

Pipeline (Figure: four tiers):
  Tier I    Structural representation extraction + space partitioning
            (k-means, boundary detection).
  Tier II   LLM-guided soft homogeneous weighting: ONE query round on the
            boundary set induced by the INITIAL H_str; affine fusion; top-c
            sparsification into the bridge. The LLM is NOT re-queried during
            bootstrapping and W~ is NOT recomputed (explicit protocol point
            in the manuscript, a deliberate cost decision).
  Tier III  Dual-view perturbed E-GCN forward passes (shared weights).
  Tier IV   Joint optimisation  L_total = L_EA + lambda * L_GCL (eq:total_loss)
            with iterative bootstrapping every R=10 epochs: centroids are
            recomputed from the CURRENT embeddings and the seed set S expands
            under (i) mutual-NN, (ii) cosine > 0.75, (iii) one-to-one, with at
            most min(ceil(0.05|S|), 200) new pairs per round.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import SGC_LHConfig
from .data import KGPair, build_intra_adjacency
from .evaluate import hits_and_mrr
from .llm import BaseLLMScorer, make_scorer
from .losses import batch_infonce, ea_loss
from .model import DualViewEGCN
from . import stage1


def _resolve_device(cfg: SGC_LHConfig) -> str:
    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_stage1(pair: KGPair, H_str: np.ndarray, cfg: SGC_LHConfig,
               scorer: BaseLLMScorer, device: str, verbose: bool = True):
    """Tier I + Tier II. Returns the sparse bridge A_bridge(W~) and diagnostics."""
    H_norm = stage1.l2_normalise(H_str)

    labels, centroids, delta = stage1.structural_partitioning(H_norm, cfg.n_clusters, cfg.seed)
    blurry = stage1.boundary_set(delta, cfg.boundary_theta)
    routed_frac = len(blurry) / len(H_norm)
    if verbose:
        print(f"[Stage 1] |E_blurry| = {len(blurry)}/{len(H_norm)} ({routed_frac:.1%} routed, theta={cfg.boundary_theta})")

    pairs, W_geo, routed_from = stage1.candidate_pairs(
        H_norm, blurry, pair.n_src, pair.n_tgt, cfg.candidate_fanout
    )
    if verbose:
        print(f"[Stage 1] candidate pairs = {len(pairs)}  (Q = c*|E_blurry| = {cfg.candidate_fanout * len(blurry)})")

    # eq:prompt / eq:llm_weight -- one-off LLM invocation on the boundary set
    W_llm = None
    routed_mask = None
    if len(pairs) > 0:
        pair_texts = []
        for i, j in zip(pairs[:, 0].tolist(), pairs[:, 1].tolist()):
            a = pair.src_entity_texts.get(i, {"name": f"entity_{i}", "attr": ""})
            b = pair.tgt_entity_texts.get(j, {"name": f"entity_{j}", "attr": ""})
            pair_texts.append((a, b))
        t0 = time.time()
        W_llm = scorer.score_pairs(pair_texts)
        routed_mask = np.ones(len(pairs), dtype=bool)
        if verbose:
            print(f"[Stage 1] LLM scoring: {len(pairs)} queries in {time.time() - t0:.1f}s "
                  f"(mean W^LLM = {float(np.mean(W_llm)):.3f})")

    W_tilde = stage1.fuse_weights(W_geo, W_llm, routed_mask, cfg.llm_alpha)   # eq:affine_weight
    A_bridge = stage1.sparsify_top_c(pairs, W_tilde, pair.n_src, pair.n_tgt,
                                     cfg.candidate_fanout, device)            # eq:bridge (top-c)
    return A_bridge, {"routed_frac": routed_frac, "n_queries": len(pairs),
                      "pairs": pairs, "W_tilde": W_tilde}


def bootstrap_seeds(Z: np.ndarray, seeds: np.ndarray, n_src: int, n_tgt: int,
                    cfg: SGC_LHConfig) -> np.ndarray:
    """Seed expansion under the three manuscript constraints.

    (i) mutual nearest neighbour, (ii) cosine similarity > 0.75, (iii) one-to-one;
    at most min(ceil(0.05|S|), 200) new pairs per round.
    """
    Zs = Z[:n_src] / np.clip(np.linalg.norm(Z[:n_src], axis=1, keepdims=True), 1e-12, None)
    Zt = Z[n_src:] / np.clip(np.linalg.norm(Z[n_src:], axis=1, keepdims=True), 1e-12, None)

    used_src = set(seeds[:, 0].tolist())
    used_tgt = set(seeds[:, 1].tolist())

    cand = []
    for i in range(n_src):
        if i in used_src:
            continue
        j = int(np.argmax(Zs[i] @ Zt.T))
        if j in used_tgt:
            continue
        j_rev = int(np.argmax(Zs @ Zt[j]))               # mutual-NN reverse check
        if j_rev != i:                                   # (i) mutual NN
            continue
        sim = float(Zs[i] @ Zt[j])
        if sim <= cfg.bootstrap_sim_threshold:           # (ii) cosine > 0.75
            continue
        cand.append((i, j, sim))

    budget = min(math.ceil(cfg.bootstrap_frac * len(seeds)), cfg.bootstrap_max_new)
    added = []
    taken_tgt = set()
    for i, j, _s in sorted(cand, key=lambda x: -x[2]):   # (iii) one-to-one
        if len(added) >= budget:
            break
        if i in used_src or j in used_tgt or j in taken_tgt:
            continue
        added.append((i, j))
        used_src.add(i); used_tgt.add(j); taken_tgt.add(j)
    if added:
        return np.vstack([seeds, np.asarray(added, dtype=seeds.dtype)])
    return seeds


def train(pair: KGPair, cfg: SGC_LHConfig | None = None, verbose: bool = True) -> dict:
    """Full SGC-LH pipeline on a KG pair. Returns the learned embeddings + metrics."""
    cfg = cfg or SGC_LHConfig()
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if verbose:
        print(f"[setup] device={device}, seed={cfg.seed}")

    # ---- Tier I: structural representation extraction --------------------
    from .data import make_initial_embeddings
    H_str = make_initial_embeddings(pair, cfg.input_dim, seed=cfg.seed)
    H0 = torch.as_tensor(H_str, dtype=torch.float32, device=device)
    n_src, n_tgt = pair.n_src, pair.n_tgt
    n = n_src + n_tgt

    # joint indexing of seeds: (src, tgt + n_src)
    seeds = np.stack([pair.train_pairs[:, 0], pair.train_pairs[:, 1] + n_src], axis=1)
    n_val = max(1, int(cfg.val_frac_of_seeds * len(seeds)))
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(seeds))
    val_seeds, train_seeds = seeds[perm[:n_val]], seeds[perm[n_val:]]
    if verbose:
        print(f"[setup] n={n} (src {n_src} / tgt {n_tgt}), "
              f"|S|={len(train_seeds)} train / {len(val_seeds)} val seeds")

    A, edge_index, edge_rel = build_intra_adjacency(pair, device)
    n_relations = int(edge_rel.max().item()) + 1 if edge_rel.numel() else 0

    # ---- Tier II: one-off LLM-guided soft weighting -----------------------
    scorer = make_scorer(cfg.llm_backend, cfg.llm_model_name, cfg.device,
                         cfg.llm_batch_size, cfg.llm_max_prompt_chars)
    A_bridge, s1_info = run_stage1(pair, H_str, cfg, scorer, device, verbose)

    # ---- Tier III/IV: joint optimisation ---------------------------------
    model = DualViewEGCN(cfg.input_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    # H0 = H_str as TRAINABLE entity embeddings (initialised from Word2Vec /
    # TransE per the paper); weight decay applies only to the E-GCN weight
    # matrices, not to the entity embeddings.
    H0 = torch.nn.Parameter(torch.as_tensor(H_str, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(
        [{"params": model.parameters(), "weight_decay": cfg.weight_decay},
         {"params": [H0], "weight_decay": 0.0}],
        lr=cfg.lr
    )

    train_t = torch.as_tensor(train_seeds, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_seeds, dtype=torch.long, device=device)
    n_tgt_t = n_tgt

    best_val, best_state, best_H0, best_epoch, patience_left = -1.0, None, None, -1, cfg.early_stop_patience
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        # One epoch = full pass over the seed set in mini-batches of B
        # (paper: batch size B=1024 for the contrastive and alignment pairs).
        seed_perm = torch.randperm(len(train_t), device=device)
        n_steps = max(1, math.ceil(len(train_t) / cfg.batch_size))
        epoch_loss = epoch_ea = epoch_gcl = 0.0

        for step, batch_idx in enumerate(seed_perm.split(cfg.batch_size)):
            Z1, Z2 = model(H0, edge_index, edge_rel, n, n_relations, A_bridge, cfg.edge_drop_rate)
            Z_avg = 0.5 * (Z1 + Z2)

            l_ea = ea_loss(Z_avg, train_t[batch_idx], cfg.margin_gamma, n_tgt_t)

            vb = min(cfg.batch_size, n)
            vertex_batch = torch.randperm(n, device=device)[:vb]
            l_gcl = batch_infonce(Z1, Z2, vertex_batch, cfg.tau)

            loss = l_ea + cfg.loss_lambda * l_gcl
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()); epoch_ea += float(l_ea.item())
            epoch_gcl += float(l_gcl.item())
        epoch_loss /= n_steps; epoch_ea /= n_steps; epoch_gcl /= n_steps

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                Z1e, Z2e = model(H0, edge_index, edge_rel, n, n_relations, A_bridge, 0.0)
                Z_eval = 0.5 * (Z1e + Z2e)
                val_h1 = _val_hits(Z_eval, val_t, n_src)
            history.append({"epoch": epoch, "loss": epoch_loss,
                            "l_ea": epoch_ea, "l_gcl": epoch_gcl,
                            "val_hits1": val_h1})
            if verbose:
                print(f"[epoch {epoch:4d}] loss={epoch_loss:.4f} "
                      f"(L_EA={epoch_ea:.4f}, L_GCL={epoch_gcl:.4f}) val Hits@1={val_h1:.4f}")
            if val_h1 > best_val + 1e-6:
                best_val, best_epoch, patience_left = val_h1, epoch, cfg.early_stop_patience
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_H0 = H0.detach().clone()
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"[early stop] no val improvement for {cfg.early_stop_patience} evals "
                              f"(best val Hits@1={best_val:.4f} @ epoch {best_epoch})")
                    break

        # ---- iterative bootstrapping: every R epochs ----------------------
        if epoch % cfg.bootstrap_every == 0 and epoch < cfg.max_epochs:
            model.eval()
            with torch.no_grad():
                Zb1, Zb2 = model(H0, edge_index, edge_rel, n, n_relations, A_bridge, 0.0)
                Z_np = (0.5 * (Zb1 + Zb2)).cpu().numpy()
            new_seeds = bootstrap_seeds(Z_np, train_seeds, n_src, n_tgt, cfg)
            if len(new_seeds) > len(train_seeds):
                if verbose:
                    print(f"[bootstrap @ {epoch}] seeds {len(train_seeds)} -> {len(new_seeds)}")
                train_seeds = new_seeds
                train_t = torch.as_tensor(train_seeds, dtype=torch.long, device=device)
            # NOTE: centroids are refreshed in principle; since W~ and the LLM
            # are NOT recomputed (paper protocol), we refresh the clustering
            # statistics only when stage-1 diagnostics are requested.

    if best_state is not None:
        model.load_state_dict(best_state)
        H0.data.copy_(best_H0)

    model.eval()
    with torch.no_grad():
        Z1e, Z2e = model(H0, edge_index, edge_rel, n, n_relations, A_bridge, 0.0)
        Z_eval = 0.5 * (Z1e + Z2e)
    test_metrics = hits_and_mrr(Z_eval, pair.test_pairs, n_src)

    if verbose:
        print("[test] " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
        print(f"[stage1 diag] routed={s1_info['routed_frac']:.1%}, queries={s1_info['n_queries']}")

    return {"embeddings": Z_eval.cpu().numpy(), "test_metrics": test_metrics,
            "history": history, "stage1": s1_info}


def _val_hits(Z: torch.Tensor, val_pairs: torch.Tensor, n_src: int) -> float:
    Zs = F.normalize(Z[:n_src], dim=1)
    Zt = F.normalize(Z[n_src:], dim=1)
    sims = Zs[val_pairs[:, 0]] @ Zt.t()
    gold = sims.gather(1, (val_pairs[:, 1] - n_src).unsqueeze(1))
    rank = (sims > gold).sum(dim=1) + 1
    return float((rank <= 1).float().mean().item())
