# SGC-LH — reference implementation

Reproduction of the method described in the manuscript *"SGC-LH: LLM-Guided
Soft Weighting with Dual-View Graph Contrastive Learning for Cross-Lingual
Knowledge Graph Entity Alignment"* (submitted to MDPI *Mathematics*).

The implementation follows the manuscript equation by equation; every module
carries the corresponding equation labels (`eq:geo_weight`, `eq:margin`,
`eq:blurry_set`, `eq:prompt`, `eq:llm_weight`, `eq:affine_weight`,
`eq:bridge`, `eq:message_passing`, `eq:infonce`, `eq:ea_loss`,
`eq:total_loss`, `eq:metrics`) so results can be cross-checked against the
paper text.

## Method summary (four tiers)

| Tier | What happens | Where |
|------|--------------|-------|
| I    | Structural space partitioning: k-means (K=10) on l2-normalised H_str, boundary ambiguity Δ_i = D(h,μ') − D(h,μ*), boundary set E_blurry(θ) | `sgc_lh/stage1.py` |
| II   | LLM-guided soft homogeneous weighting: top-c candidate fan-out, W_ij = exp(−F_metric), one-off LLM scoring of the boundary set, affine fusion W̃ = (1−α)W + αW^LLM, top-c sparsification into the symmetric cross-graph bridge | `sgc_lh/stage1.py`, `sgc_lh/llm.py` |
| III  | Dual-view enhanced E-GCN: symmetric edge dropping + relation masking per view, bridge-augmented propagation with degrees recomputed from the augmented matrix | `sgc_lh/model.py` |
| IV   | Joint optimisation L = L_EA (margin ranking, γ=1.0) + λ·L_GCL (symmetric batch InfoNCE, τ=0.1), iterative seed bootstrapping every R=10 epochs (mutual-NN, cos > 0.75, one-to-one) | `sgc_lh/losses.py`, `sgc_lh/train.py` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, only for the real Qwen backend:
pip install transformers accelerate bitsandbytes
```

## Quick start (synthetic smoke test)

```bash
python scripts/run_smoke_test.py
```

Builds two structurally isomorphic synthetic KGs (400 entities per side,
175 test pairs) and runs the full pipeline. Expected output: test
Hits@1 ≈ 0.96, Hits@10 = 1.00, MRR ≈ 0.98 (~2 s on CPU).

## Real benchmark

```bash
python scripts/run_dbp15k.py --data-root data/DBP15K_ZH_EN \
    --llm auto --embeddings data/dbp15k_zh_en_emb.npy
```

See `data/README.md` for the expected directory layout.

## LLM backends (`--llm`)

- `qwen` — Qwen-7B with 4-bit weight-only quantisation via
  transformers + bitsandbytes, Yes/No verbaliser softmax (eq:llm_weight),
  invoked exactly once on the boundary set (paper protocol). Requires the
  optional dependencies and sufficient memory (or a vLLM endpoint adapted in
  `sgc_lh/llm.py`).
- `heuristic` — a difflib-based string similarity stand-in. **This is NOT the
  paper's model**; it exists so the pipeline runs without GPU/weights. Use it
  only for mechanics testing.
- `auto` (default) — uses `qwen` when transformers is importable, else `heuristic`.

## Honest scope notes

- Default hyperparameters match the manuscript: K=10, θ=0.05, c=8, α=0.4,
  d=300, d'=256, p_v=0.2, τ=0.1, lr=1e-3, weight decay 1e-5 (GCN matrices
  only), B=1024, γ=1.0, λ=0.1, ≤200 epochs, patience 30, bootstrap every
  R=10 epochs.
- **The paper's reported numbers additionally require** the original
  Word2Vec + TransE initialisation vectors (`--embeddings`) and the real
  benchmark data; neither is redistributed here. Without them the code runs
  end-to-end but does not reproduce the exact tables.
- The paper runs the LLM through vLLM; this implementation uses
  transformers + bitsandbytes as a drop-in protocol match (same prompt
  format, same single-query round, same two-token verbaliser).

## Repository layout

```
sgc_lh/
    config.py     # all hyperparameters (SGC_LHConfig dataclass)
    data.py       # benchmark loading, joint-graph construction, synthetic pair
    stage1.py     # Tier I + II: partitioning, boundary set, W_geo, fusion, bridge
    llm.py        # LLM scoring backends (Qwen 4-bit / heuristic)
    model.py      # dual-view enhanced E-GCN with bridge (eq:bridge, eq:message_passing)
    losses.py     # margin ranking L_EA + symmetric batch InfoNCE L_GCL
    evaluate.py   # Hits@k / MRR over the full split + paired bootstrap CI
    train.py      # end-to-end pipeline, bootstrapping, early stopping
scripts/
    run_smoke_test.py   # synthetic end-to-end test (no data needed)
    run_dbp15k.py       # CLI for real benchmarks
data/
    README.md     # expected benchmark layout (folder ships empty)
```
