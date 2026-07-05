# bio2nl — Project Index

**Question:** *Is structural transfer between natural language and biological sequences symmetric?*

**Answer:** No. Transfer is directional and strongly asymmetric. Language→Biology is robust;
Biology→Language is weak, does not scale, and does not generalize. Shared structural
regularity does **not** imply symmetric representational transfer.

Prior work (this group): Language→Biology forward transfer (`../nlp-trans-bio.pdf`) and the
OmniGene-4 unified model (`../omnigene4_mm.pdf`). This project supplies the reverse direction
and the symmetry analysis.

---

## Directory map

```
bio2nl/
├── PROJECT_INDEX.md          # this file — start here
├── README.md                 # original design-of-record
├── EXP2_CPT_DESIGN.md        # continued-pretraining ablation design (from gemini.md+gpt.md)
│
├── tier1_gpt2/               # GPT-2 experiments (classification FT + CPT + bidirectional)
│   ├── 01_pilot_bio2nl.py            # first reverse pilot (single seed)
│   ├── 02_batch_distribution.py      # reverse FT, 100-seed distribution
│   ├── 03_batch_remote_source.py     # reverse FT with remote-homology source
│   ├── 04_batch_backbone.py          # backbone ablation (Eng/bio/bio+Eng gpt2 variants)
│   ├── 05_bidirectional_symmetry.py  # ** both directions, one protocol (Table 1) **
│   ├── run_cpt.py                    # iso-token continued pretraining (--group --base)
│   └── cpt_ckpts/                    # 24 CPT checkpoints (22 GB): gpt2/ + gpt2-medium/
│                                     #   {protein,text,shuffled,randomaa}_seed{0,1,2}
│
├── tier2_llama/              # cross-modal / larger-model backbone tests
│   ├── eval_esm_on_nl.py             # ESM/ProtBERT/BERT on NL tasks (Tables 5,6)
│   └── eval_crossmodal_2x2.py        # backbones on protein homology (Table 4 bio side)
│
├── eval/
│   ├── eval_structural_nl.py         # sample-efficiency FT harness (PAWS-X/CoLA/RTE + local tasks)
│   ├── make_dyck_task.py             # synthetic Dyck nested-bracket task (hard, non-saturating)
│   └── make_longrange_task.py        # synthetic subj-verb agreement (deprecated: saturates)
│
├── analysis/
│   ├── make_all_tables.py            # ** master: regenerates ALL tables -> results/SUMMARY.md **
│   ├── analyze_distribution.py       # 100-seed distribution stats
│   └── mechanism_probe.py            # frozen linear-probe on CPT checkpoints
│
├── data/
│   ├── cpt_corpus/                   # iso-token CPT corpora (see Data section)
│   └── eval_nl/                      # synthetic eval tasks (Dyck, longrange)
│
└── results/
    ├── SUMMARY.md                    # ** all tables, auto-generated **
    ├── benchmarks/*.jsonl            # raw result records (one row per run)
    └── logs/                         # all run logs
```

---

## Data (all under `data/`, plus source protein pairs in `../biopaws/1-data/`)

| file | size | what |
|---|---|---|
| `cpt_corpus/sprot.fasta` | 288 MB | full Swiss-Prot (575,503 proteins), CPT source |
| `cpt_corpus/cpt_protein.txt` | 101 MB | **G2** real protein, 50.0M gpt2-tokens (iso-token) |
| `cpt_corpus/cpt_shuffled.txt` | 101 MB | **G3** residue-shuffled (same AA freq/length) |
| `cpt_corpus/cpt_randomaa.txt` | 101 MB | **G4** random AA from background freq |
| `cpt_corpus/cpt_text.txt` | 224 MB | **G1** OpenWebText English, 50.0M tokens |
| `cpt_corpus/manifest.json` | — | exact token counts per corpus |
| `eval_nl/dyck_L{40,60}_t3_*.jsonl` | 8k/2k | Dyck nested-bracket well-formedness (hard task) |
| `eval_nl/longrange_d{8,16,24}_*.jsonl` | 8k/2k | subj-verb agreement (deprecated — saturates) |
| `../biopaws/1-data/protein_pair_20k_*.csv` | — | protein homology pairs (sentence1,sentence2,label) |

External (downloaded at runtime via HF): PAWS-X (en), GLUE CoLA/RTE, OpenWebText.

---

## Results → paper tables (all in `results/SUMMARY.md`)

| Table | Finding | Source jsonl |
|---|---|---|
| 1 Bidirectional symmetry | NL→Bio 0.77 vs Bio→NL 0.55 (one protocol) | `bidirectional_symmetry.jsonl` |
| 2 Reverse FT 100-seed | 0/100 beat baseline; no transfer | `tier1_distribution.jsonl` |
| 3 CPT iso-token ablation | protein>shuffled at 124M (+0.09), vanishes at 355M | `exp2_structural_nl*.jsonl` |
| 4 2×2 cross-modal | off-domain drop: language 0.08 vs bio 0.36 | `crossmodal_2x2.jsonl` + `esm_on_nl.jsonl` |
| 5 Scaling divergence | language↑ (0.58→0.91), bio→lang↓ (0.64→0.57); ESM=ProtBERT=0.569 | `esm_on_nl.jsonl` |
| 6 ESM structural-only | above chance on PAWS, collapses on CoLA/RTE | `esm_on_nl.jsonl` |
| 7 Dyck hard task | protein>shuffled does NOT hold | `exp2_structural_nl.jsonl` (task=dyck) |

Also: `tier1_backbone_*.jsonl` (backbone ablation), `tier1_dist_remote70.jsonl` (remote source),
`tier1_pilot_seed56.json` (pilot), `router/mechanism_probe.jsonl` (frozen probe).

---

## Reproduce

```bash
export HF_ENDPOINT=https://hf-mirror.com          # AutoDL mirror
# 1. build CPT corpora (needs sprot.fasta)
python data/cpt_corpus/build_cpt_corpora.py --target_tokens 50000000
# 2. continued pretraining (per group/seed/size)
python tier1_gpt2/run_cpt.py --group protein --seed 0 --base gpt2
# 3. evals
python tier1_gpt2/05_bidirectional_symmetry.py --n_seeds 15      # Table 1
python eval/eval_structural_nl.py --base gpt2 --tasks paws-x,cola,rte  # Table 3
python tier2_llama/eval_esm_on_nl.py --tasks paws-x --models esm2_8M,esm2_650M,protbert,bert_base  # Tables 5,6
python tier2_llama/eval_crossmodal_2x2.py                        # Table 4 bio side
# 4. regenerate all tables
python analysis/make_all_tables.py                               # -> results/SUMMARY.md
```

Env: AutoDL, torch 2.3.0+cu121 (vGPU-32GB). RTX 5090 migration: reinstall torch cu124/cu128
(Blackwell sm_120) — see memory note `bio2nl-5090-migration-plan`.
