# bio2nl — Asymmetric Structural Transfer Between Natural Language and Biological Foundation Models

Code and analysis for the paper *Asymmetric Structural Transfer Between Natural Language and
Biological Foundation Models*.

**Question.** Prior work shows language models transfer to biology (language\,→\,biology). Is the
transfer symmetric? We systematically evaluate the reverse direction and find that structural
transfer is **bidirectional but strongly and universally asymmetric**: language\,→\,biology is
robust and scales up, whereas biology\,→\,language is weak, never beats matched-token controls, and
decays toward chance as protein models grow (reproduced across ESM-2 and ProtBERT). All central
comparisons use models with **known training data**, eliminating the pretraining-contamination
confound.

## Repository layout

```
bio2nl/
├── tier1_gpt2/        GPT-2 experiments
│   ├── 01_pilot_bio2nl.py            reverse pilot
│   ├── 02_batch_distribution.py      reverse FT, 100-seed distribution
│   ├── 03_batch_remote_source.py     remote-homology source
│   ├── 04_batch_backbone.py          backbone ablation
│   ├── 05_bidirectional_symmetry.py  both directions, one protocol
│   └── run_cpt.py                    iso-token continued pretraining
├── tier2_llama/       cross-modal / larger backbones
│   ├── eval_esm_on_nl.py             ESM/ProtBERT/BERT on NL tasks + scaling
│   └── eval_crossmodal_2x2.py        backbones on protein homology (2x2 matrix)
├── eval/
│   ├── eval_structural_nl.py         sample-efficiency FT harness
│   ├── make_dyck_task.py             hard nested-bracket task
│   └── make_longrange_task.py        synthetic long-range task
├── analysis/
│   ├── make_all_tables.py            regenerate ALL tables -> results/SUMMARY.md
│   ├── M1_cka_alignment.py           representation alignment (CKA)
│   ├── M2_learning_dynamics.py       transfer-vs-training-step trajectories
│   ├── M3_difference_heads.py        difference-head test (reverse)
│   ├── M3b_forward_difference_heads.py   difference-head test (forward)
│   ├── M3c_rigorous_diff_heads.py    rigorous, z-scored difference-head test
│   └── analyze_distribution.py       100-seed distribution stats
├── data/
│   ├── cpt_corpus/build_cpt_corpora.py   builds iso-token CPT corpora
│   └── eval_nl/                      synthetic eval tasks (Dyck, long-range)
├── results/
│   ├── SUMMARY.md                    all result tables (auto-generated)
│   └── benchmarks/*.jsonl            raw per-run records
└── paper/             LaTeX manuscript + figures
```

## Data & checkpoints

Large artifacts are **not** in this repo. The CPT corpora, Swiss-Prot source, and evaluation
datasets are hosted on HuggingFace: **[`dnagpt/bio2nl`](https://huggingface.co/datasets/dnagpt/bio2nl)**.
Protein-homology pairs use the existing [`dnagpt/biopaws`](https://huggingface.co/datasets/dnagpt/biopaws)
release. Model checkpoints (24 CPT models) can be regenerated with `run_cpt.py`.

## Reproduce

```bash
export HF_ENDPOINT=https://hf-mirror.com                    # optional mirror
python data/cpt_corpus/build_cpt_corpora.py --target_tokens 50000000
python tier1_gpt2/run_cpt.py --group protein --seed 0 --base gpt2
python tier1_gpt2/05_bidirectional_symmetry.py --n_seeds 15
python tier2_llama/eval_esm_on_nl.py --tasks paws-x --models esm2_8M,esm2_650M,protbert,bert_base
python tier2_llama/eval_crossmodal_2x2.py
python analysis/M2_learning_dynamics.py --seeds 0,1,2
python analysis/make_all_tables.py                          # -> results/SUMMARY.md
```

## Environment
Python 3, PyTorch, transformers, datasets, scikit-learn. Experiments run on a single 32 GB GPU.
