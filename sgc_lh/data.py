"""Dataset loading and joint-graph construction.

Supports the standard EA benchmark layout used by DBP15K / SRPRS / DWY
distributions:

    data_root/
        norm/triples_0.pkl          # source triples  (or ent_ids_1/triples_1 text)
        ...
        ent_ids_1, ent_ids_2        # entity id <-> name mappings
        train_pairs, test_pairs     # seed and test pairs (ids, one pair per line)

The loader is deliberately tolerant: if the well-known pickle layout is not
found it falls back to plain-text triple files. Entity text (name + attributes)
is required only for Stage-1 LLM scoring; benchmarks distribute it as
`ent_texts_1` / `ent_texts_2` (JSONL: one JSON object per entity id).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class KGPair:
    """A source/target KG pair plus ground-truth alignment."""
    src_triples: list          # (h, r, t) with h,t in source entity id space
    tgt_triples: list
    src_entity_texts: dict     # id -> {"name": ..., "attr": ...} (may be empty)
    tgt_entity_texts: dict
    train_pairs: np.ndarray    # (n_seed, 2) int64, source id / target id
    test_pairs: np.ndarray     # (n_test, 2) int64
    n_src: int
    n_tgt: int


def _load_triples_text(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) == 3:
                h, r, t = parts
                out.append((int(h), int(r), int(t)))
    return out


def _load_pairs(path: Path) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1])))
    return np.asarray(rows, dtype=np.int64)


def _load_texts(path: Path) -> dict:
    if not path.exists():
        return {}
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = int(obj["id"])
                texts[eid] = {"name": obj.get("name", ""), "attr": obj.get("attr", obj.get("attrs", ""))}
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return texts


def load_kg_pair(data_root: str | Path) -> KGPair:
    """Load a benchmark directory. Tries the pickle layout first, then text."""
    root = Path(data_root)

    # ---- triples ----------------------------------------------------------
    src, tgt = [], []
    pkl = root / "norm" / "triples_0.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            src = pickle.load(f)
        with open(root / "norm" / "triples_1.pkl", "rb") as f:
            tgt = pickle.load(f)
    else:
        src = _load_triples_text(root / "triples_1")
        tgt = _load_triples_text(root / "triples_2")

    n_src = max((h for h, _, _ in src), default=0) + 1 if src else 0
    n_src = max(n_src, max((t for _, _, t in src), default=-1) + 1)
    n_tgt = max((h for h, _, _ in tgt), default=0) + 1 if tgt else 0
    n_tgt = max(n_tgt, max((t for _, _, t in tgt), default=-1) + 1)

    # ---- seeds / test pairs ----------------------------------------------
    train = _load_pairs(root / "train_pairs")
    test = _load_pairs(root / "test_pairs")

    # If an ent_ids mapping exists (external id -> 0-based index), apply it.
    for ids_file, offset in (("ent_ids_1", 0), ("ent_ids_2", 1)):
        f = root / ids_file
        if f.exists():
            mapping = {}
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        mapping[int(parts[0])] = int(parts[1])
            arrs = []
            for arr, off in ((train, 0), (test, 0)):
                if arr.size:
                    remapped = arr.copy()
                    remapped[:, off] = np.array([mapping.get(int(x), int(x)) for x in arr[:, off]])
                    arrs.append(remapped)
            if arrs:
                train, test = arrs

    texts_src = _load_texts(root / "ent_texts_1")
    texts_tgt = _load_texts(root / "ent_texts_2")

    return KGPair(
        src_triples=src,
        tgt_triples=tgt,
        src_entity_texts=texts_src,
        tgt_entity_texts=texts_tgt,
        train_pairs=train,
        test_pairs=test,
        n_src=n_src,
        n_tgt=n_tgt,
    )


def make_initial_embeddings(pair: KGPair, dim: int = 300, emb_path: str | None = None,
                            seed: int = 42) -> np.ndarray:
    """H_str in R^{n x d}, d=300.

    If `emb_path` points to a .npy/.npz of shape [n_src+n_tgt, d] (pre-trained
    Word2Vec + TransE concatenation, as in the paper), it is used directly.
    Otherwise a seeded random initialisation is returned; with real benchmarks
    the paper's numbers require the distributed Word2Vec/TransE vectors.
    """
    n = pair.n_src + pair.n_tgt
    if emb_path:
        arr = np.load(emb_path)
        if arr.ndim == 2 and arr.shape == (n, dim):
            return arr.astype(np.float32)
        raise ValueError(f"embedding file shape {arr.shape} != ({n}, {dim})")

    # No pre-trained vectors: fall back to deterministic name-hash embeddings
    # (a stand-in for the paper's Word2Vec/TransE initialisation, which carries
    # the cross-lingual semantic signal that Stage-1's geometric weight W_geo
    # relies on when selecting the top-c candidates). Entities sharing the same
    # name get the same base vector + small per-entity noise, so on the
    # synthetic pair (where aligned entities share a name) W_geo ranks true
    # pairs first -- mirroring how multilingual embeddings behave on DBP15K.
    rng = np.random.default_rng(seed)
    H = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(n, dim)).astype(np.float32)
    texts = [pair.src_entity_texts, pair.tgt_entity_texts]
    n_src = pair.n_src
    for i in range(n):
        side, eid = (texts[0], i) if i < n_src else (texts[1], i - n_src)
        info = side.get(eid)
        if info and info.get("name"):
            digest = hashlib.sha256(info["name"].encode("utf-8")).digest()
            grng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            base = grng.normal(0.0, 1.0 / np.sqrt(dim), size=dim)
            H[i] = base + 0.05 * rng.normal(0.0, 1.0 / np.sqrt(dim), size=dim)
    return H.astype(np.float32)


def build_intra_adjacency(pair: KGPair, device: str = "cpu") -> torch.Tensor:
    """Global intra-graph adjacency A (block diagonal over the joint vertex set).

    A_{ij} = 1 if a triple (i, r, j) exists in either graph; relations are kept
    separately as an edge-index/relation tensor for relation masking.
    Returns (A_sparse, edge_index, edge_rel) where edge_index[2, m] uses joint
    indices and edge_rel[m] the relation ids (source relations re-based by
    +max_src_rel so the two graphs never share a relation id).
    """
    n_src, n_tgt = pair.n_src, pair.n_tgt
    n = n_src + n_tgt
    src_rel_max = max((r for _, r, _ in pair.src_triples), default=-1) + 1

    rows, cols, rels = [], [], []
    for h, r, t in pair.src_triples:
        rows += [h, t]; cols += [t, h]; rels += [r, r]          # symmetrise
    for h, r, t in pair.tgt_triples:
        h, t = h + n_src, t + n_src
        rows += [h, t]; cols += [t, h]; rels += [r + src_rel_max, r + src_rel_max]

    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    rel = torch.tensor(rels, dtype=torch.long, device=device)
    keep = idx[0] != idx[1]                                     # drop self loops
    idx, rel = idx[:, keep], rel[keep]
    vals = torch.ones(idx.shape[1], device=device)
    A = torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()
    return A, idx, rel


def synth_pair(n_per_side: int = 300, n_relations: int = 8, n_alignable: int = 250,
               density: float = 8.0, seed: int = 42) -> KGPair:
    """Synthetic KG pair for pipeline smoke tests (no real benchmark needed).

    Constructs two structurally similar random graphs whose first
    `n_alignable` entities correspond one-to-one with similar neighbourhood
    statistics, so a correct implementation attains non-trivial Hits@1.
    """
    rng = np.random.default_rng(seed)
    n_src = n_tgt = n_per_side

    def rand_graph(n: int) -> list:
        triples = []
        for _ in range(int(n * density)):
            h, t = rng.integers(0, n, size=2)
            if h != t:
                triples.append((int(h), int(rng.integers(0, n_relations)), int(t)))
        return triples

    src = rand_graph(n_src)
    # target = copy of source with a structural-noise reshuffle
    tgt = []
    for h, r, t in src:
        tgt.append((h, int(r), t))  # identical core structure
    extra = rand_graph(n_tgt)
    seen = {(h, t) for h, _, t in tgt}
    for h, r, t in extra:
        if (h, t) not in seen:
            tgt.append((h, int(r), t))
            seen.add((h, t))

    # ONE permutation applied to BOTH endpoints (a single bijection pi makes
    # the target graph isomorphic to the source graph's core; alignment is
    # src i <-> tgt pi(i)). Applying different permutations to heads and tails
    # would destroy the structural correspondence entirely.
    ids = rng.permutation(n_tgt)
    tgt = [(int(ids[h]), r, int(ids[t])) for h, r, t in tgt]
    inv = np.empty_like(ids); inv[ids] = np.arange(n_tgt)

    alignable = np.stack([np.arange(n_alignable), ids[:n_alignable]], axis=1)
    perm = rng.permutation(len(alignable))
    train = alignable[perm[: int(0.3 * len(alignable))]]
    test = alignable[perm[int(0.3 * len(alignable)):]]

    texts = {i: {"name": f"entity_{i}", "attr": "attr"} for i in range(n_src)}
    # target entity ids[i] is the counterpart of source entity i, so its name
    # carries the same semantic token (the LLM scoring signal).
    texts_t = {i: {"name": f"entity_{int(inv[i])}", "attr": "attr"} for i in range(n_tgt)}

    return KGPair(src, tgt, texts, texts_t, train.astype(np.int64),
                  test.astype(np.int64), n_src, n_tgt)
