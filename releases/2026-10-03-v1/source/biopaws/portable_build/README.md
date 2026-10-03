# Pinned CPU protein-data reconstruction

`protein_phase.py` runs the accepted v2 protein/SCOPe builders with explicit release and builder roots. It is one phase of the offline reconstruction documented in `bio2nl/portable_data/README.md`. Run the raw-only authenticated bootstrap first.

```bash
CUDA_VISIBLE_DEVICES='' python -B /path/to/biopaws/portable_build/protein_phase.py \
  --release-root /path/to/new_release \
  --builder-root /path/to/new_release/code/biopaws \
  --log-root /path/to/new_protein_logs
```

The nine serial stages normalize Swiss-Prot, cluster, search, assign graph splits, prepare SCOPe, exclude qualifying remote homologs, build both pair datasets and audit cross-dataset overlap. The accepted MMseqs version and original scientific builder bytes are pinned; this wrapper makes no new dataset acquisition or model inference. CPU thread count is 16.

Use a fresh root populated only by the bootstrap and a nonexistent log directory. The runner refuses existing work/derived protein outputs, hashes stage inputs and outputs, preserves partial failures and stops on the first failed job. `protein_phase_status.json` reports actual progress; a launched PID is not completion evidence. No GPU is used.

This phase alone does not accept the full release. Corpus, tokenizer and independent whole-release verification must pass before derived-data reconstruction. Operational sequence-pair labels are not a guarantee of structural-homology reasoning. External source licenses and the original acquisition provenance remain distinct from a new local execution.
