---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- cross-domain-transfer
- protein
- foundation-models
- structural-transfer
pretty_name: bio2nl
---

# bio2nl — data for *Asymmetric Structural Transfer Between Natural Language and Biological Foundation Models*

Datasets accompanying the bio2nl study of directional cross-domain structural transfer.
Code: https://github.com/maris205/bio2nl

## Contents

### `cpt_corpus/` — iso-token continued-pretraining corpora
Each file is exactly **50M GPT-2 tokens**, differing only in sequence content (the iso-token
ablation). Proteins are rendered as space-separated residues `<protein> M K V ... </protein>`.

| file | content |
|---|---|
| `cpt_protein.txt` | real Swiss-Prot proteins (G2, main) |
| `cpt_shuffled.txt` | residue-shuffled proteins (G3; same AA freq/length, destroyed structure) |
| `cpt_randomaa.txt` | random amino acids from background frequency (G4) |
| `cpt_text.txt`    | OpenWebText English (G1, control) |
| `sprot.fasta`     | full Swiss-Prot source (575,503 proteins) |
| `manifest.json`   | exact token counts per corpus |

### `eval_nl/` — synthetic structural evaluation tasks
- `dyck_L{40,60}_t3_*.jsonl` — nested-bracket well-formedness (hard, non-saturating long-range structure)
- `longrange_d{8,16,24}_*.jsonl` — subject–verb agreement across distractors

## Related
Protein-homology pairs use the existing [`dnagpt/biopaws`](https://huggingface.co/datasets/dnagpt/biopaws)
release. PAWS-X, GLUE (CoLA/RTE), OpenWebText are downloaded from their standard sources.
