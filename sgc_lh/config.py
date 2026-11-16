"""Hyperparameter configuration.

Default values follow the manuscript's core-hyperparameter table exactly:

    Stage 1 (LLM soft weighting):  K=10, theta=0.05, c=8, alpha=0.4
    Stage 2 (GCL):                 d=300, p_v=0.2, tau=0.1
    Optimisation:                  lr=1e-3, wd=1e-5, B=1024,
                                   gamma=1.0, lambda=0.1, epochs<=200
    Bootstrapping:                 every R=10 epochs, cos>0.75, mutual NN,
                                   one-to-one, <= min(ceil(0.05|S|), 200) new pairs
    Early stopping:                patience 30 epochs on validation Hits@1
"""

from dataclasses import dataclass


@dataclass
class SGC_LHConfig:
    # ---- Stage 1: LLM-guided soft homogeneous weighting -----------------
    n_clusters: int = 10          # K, number of k-means regions
    boundary_theta: float = 0.05  # theta, margin threshold (dimensionless, Delta in [0,2])
    candidate_fanout: int = 8     # c, nearest candidates per boundary entity
    llm_alpha: float = 0.4        # alpha, affine balance between W and W^LLM

    # ---- Stage 2: dual-view GCL E-GCN -----------------------------------
    input_dim: int = 300          # d, initial structural embedding dim (Word2Vec/TransE)
    hidden_dim: int = 256         # d', hidden/output dim of the enhanced E-GCN
    n_layers: int = 2             # L, number of E-GCN layers (Figure: Layer 1 / Layer 2)
    edge_drop_rate: float = 0.2   # p_v, symmetric edge-dropping rate per view
    tau: float = 0.1              # InfoNCE temperature

    # ---- Optimisation ----------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-5    # applied to the E-GCN convolution matrices
    batch_size: int = 1024        # B for contrastive and alignment pairs
    margin_gamma: float = 1.0     # gamma in the margin ranking loss
    loss_lambda: float = 0.1      # lambda in L_total = L_EA + lambda * L_GCL
    max_epochs: int = 200
    early_stop_patience: int = 30

    # ---- Iterative bootstrapping -----------------------------------------
    bootstrap_every: int = 10     # R=10 epochs
    bootstrap_sim_threshold: float = 0.75
    bootstrap_max_new: int = 200  # cap: min(ceil(0.05|S|), 200)
    bootstrap_frac: float = 0.05

    # ---- Stage-1 LLM -----------------------------------------------------
    llm_model_name: str = "Qwen/Qwen-7B"
    llm_backend: str = "auto"     # "qwen" | "heuristic" | "auto" (auto: qwen if GPU else heuristic)
    llm_batch_size: int = 16
    llm_max_prompt_chars: int = 1500

    # ---- Misc -------------------------------------------------------------
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    val_frac_of_seeds: float = 0.1  # validation split carved out of the seed set
