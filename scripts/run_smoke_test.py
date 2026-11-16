#!/usr/bin/env python3
"""End-to-end smoke test on a synthetic KG pair (CPU, no LLM weights needed).

Runs the full SGC-LH pipeline -- Stage 1 (partitioning, boundary routing,
affine fusion, top-c bridge) and Stage 2 (dual-view E-GCN, joint loss,
bootstrapping, early stopping, evaluation) -- on a small synthetic pair with
the heuristic LLM backend. Verifies plumbing, not paper-scale accuracy.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgc_lh.config import SGC_LHConfig
from sgc_lh.data import synth_pair
from sgc_lh.train import train


def main():
    cfg = SGC_LHConfig(
        input_dim=64, hidden_dim=64,
        boundary_theta=0.15,          # raise so the boundary set is non-empty on toy data
        max_epochs=150, bootstrap_every=10, early_stop_patience=6,
        batch_size=256, llm_backend="heuristic", device="cpu",
    )
    pair = synth_pair(n_per_side=400, n_alignable=350, seed=cfg.seed)

    t0 = time.time()
    result = train(pair, cfg, verbose=True)
    dt = time.time() - t0

    m = result["test_metrics"]
    print("\n=== smoke test summary ===")
    print(f"wall time        : {dt:.1f}s")
    print(f"test Hits@1      : {m['hits@1']:.4f}")
    print(f"test Hits@10     : {m['hits@10']:.4f}")
    print(f"test MRR         : {m['mrr']:.4f}")
    print(f"Stage-1 routed   : {result['stage1']['routed_frac']:.1%}")
    print(f"Stage-1 queries  : {result['stage1']['n_queries']}")
    assert m["hits@1"] > 0.2, "Hits@1 unexpectedly low -- check pipeline"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
