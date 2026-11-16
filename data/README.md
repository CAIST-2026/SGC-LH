# Data layout

SGC-LH expects a benchmark directory (DBP15K / SRPRS / DWY style) laid out as:

```
data_root/
    triples_1               # source KG triples  "head \t relation \t tail"
    triples_2               # target KG triples
    ent_ids_1               # OPTIONAL: external id -> 0-based index (source)
    ent_ids_2               # OPTIONAL: external id -> 0-based index (target)
    ent_texts_1             # OPTIONAL: JSONL {"id":..., "name":..., "attr":...}
    ent_texts_2             #           (needed for Stage-1 LLM scoring)
    train_pairs             # seed alignment  "src_id \t tgt_id"  (one pair per line)
    test_pairs              # test alignment  "src_id \t tgt_id"
```

The pickle variant used by several DBP15K distributions is also accepted:

```
data_root/
    norm/triples_0.pkl      # pickled list of source triples
    norm/triples_1.pkl      # pickled list of target triples
```

## Notes

- Entity ids in `triples_*` / `*_pairs` are expected to be 0-based contiguous
  integers. If the distribution ships its own id space, put the mapping in
  `ent_ids_1` / `ent_ids_2` and the loader remaps automatically.
- `ent_texts_*` is optional but strongly recommended: Stage 1 prompts the LLM
  with `instruction || name || attributes`, so without it the LLM scoring
  degrades to uninformative placeholders.
- Initial structural embeddings `H_str` (d = 300, Word2Vec + TransE
  concatenation in the paper) are passed via `--embeddings path.npy` as a
  single `[n_src + n_tgt, 300]` float array with rows ordered
  `[source entities; target entities]`. Without it the loader falls back to
  deterministic name-hash vectors (see `sgc_lh/data.py::make_initial_embeddings`);
  this fallback reproduces the *mechanics* of the pipeline but NOT the paper's
  reported numbers, which require the original pre-trained vectors.
- This folder intentionally ships empty; download a benchmark (e.g. DBP15K V1
  from the OpenEA / JAPE releases) and unpack it here as `data/DBP15K_ZH_EN/`.
