#!/usr/bin/env python3
"""Run SGC-LH on a real benchmark (DBP15K / SRPRS / DWY layout).

Usage:
    python scripts/run_dbp15k.py --data-root data/DBP15K_ZH_EN \
        [--llm qwen|heuristic|auto] [--embeddings path/to/emb.npy] \
        [--alpha 0.4] [--theta 0.05] [--seed 42]

Data layout expected under --data-root (see data/README.md):
    triples_1 / triples_2        (or norm/triples_0.pkl + norm/triples_1.pkl)
    ent_ids_1 / ent_ids_2        (optional id remapping)
    ent_texts_1 / ent_texts_2    (JSONL {"id","name","attr"} -- Stage-1 LLM prompts)
    train_pairs / test_pairs     (id \t id per line; 30% seeds by default)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgc_lh.config import SGC_LHConfig
from sgc_lh.data import load_kg_pair
from sgc_lh.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--llm", choices=["auto", "qwen", "heuristic"], default="auto")
    ap.add_argument("--llm-model", default="Qwen/Qwen-7B")
    ap.add_argument("--embeddings", default=None,
                    help=".npy of shape [n_src+n_tgt, 300] with Word2Vec/TransE initialisation")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--theta", type=float, default=0.05)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()

    cfg = SGC_LHConfig(
        llm_alpha=args.alpha, boundary_theta=args.theta, n_clusters=args.K,
        candidate_fanout=args.c, loss_lambda=args.lam, seed=args.seed,
        device=args.device, max_epochs=args.max_epochs, llm_backend=args.llm,
        llm_model_name=args.llm_model,
    )
    pair = load_kg_pair(args.data_root)
    result = train(pair, cfg, verbose=True)

    m = result["test_metrics"]
    print("\n=== final ===")
    print(f"Hits@1={m['hits@1']:.4f}  Hits@10={m['hits@10']:.4f}  MRR={m['mrr']:.4f}")
    print(f"Stage-1: routed {result['stage1']['routed_frac']:.1%}, "
          f"{result['stage1']['n_queries']} LLM queries")


if __name__ == "__main__":
    main()
