# bio2nl: protein-pair supervision and English transfer

This repository studies whether protein sequence-pair supervision transfers to English sentence-pair prediction. The current audited experiment uses the deterministic October 2026 data reconstruction, nine fresh pretraining runs, 27 full source fine-tuning runs, and fixed evaluation of all 27 source-selected classifiers. Older root-level experiments and paper files remain historical; the release below identifies the current code and results.

## Current release

- [Release documentation and CPU commands](releases/2026-10-03-v1/README.md).
- [Complete 190-file source snapshot](releases/2026-10-03-v1/source/) and [file manifest](releases/2026-10-03-v1/source/MANIFEST.json).
- [Protein data and the same code archive on Hugging Face](https://huggingface.co/datasets/dnagpt/bio2nl/tree/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1), pinned to dataset commit `62ced1ec44444b1879f2c473c553e6b5a9c6767f`.
- [Fixed-transfer results](releases/2026-10-03-v1/source/results/oct2/fixed_transfer/REPORT.md), [source-fit results](releases/2026-10-03-v1/source/results/oct2/source/REPORT.md), and [Source Data](releases/2026-10-03-v1/source/paper_assets/source_data/README.md).

The source capture is `b8d50d0ae61c8407ea8dcf62e47edef5969fc56c`; it is a later exact code capture, not the Git revision used during the original training. Both the `bio2nl/` and `biopaws/` source layouts are retained inside the snapshot. No model weights or protein-data payloads are added to this Git repository by this release. Eight biological payloads are available in the linked Hugging Face data archive; the source archive contains only the identities of 36 model checkpoints.

## Experiment and interpretation

EP combines English with natural protein exposure; ES uses the same English blocks and shuffled protein; EE uses twice the English exposure at the same total token budget. Each condition has three independent pretraining seeds, each followed by three source fine-tuning seeds. Statistics first average the fine-tuning repeats within each pretraining seed, then report the mean and sample SD across the three pretraining seeds.

The natural-protein condition has a modest exploratory QQP ranking gain over its shuffled control: paired AUC difference **+0.029560 ± 0.018655**, positive for all three pretraining-seed groups. Its protein-test accuracy is **50.76%**, and English balanced accuracy is **50.18%**. Weak source learning and fixed English classification limit interpretation. These results do not establish statistical significance, transferable protein-relation semantics, a general directional asymmetry, or impossibility of transfer. The QQP cohort was evaluated historically and is not a new blind test.

Protein examples are constructed directly from pinned UniProtKB/Swiss-Prot records using normalization, MMseqs2 search/clustering, split assignment and explicit pairing rules. Labels describe **operational sequence similarity**, not certified structural homology or evolutionary independence. Pair splits contain 99,818 / 20,276 / 20,854 rows; the experimental source training subset contains 8,044 rows. The budget-matched surface reference uses BPE-token statistics; raw-residue construction baselines use all 99,818 training rows and are reported separately. Equal BPE-token budgets do not guarantee equal residue or parent-record exposure.

## Reproduction scope

From the checkout root with Python 3.12:

```sh
SOURCE="$PWD/releases/2026-10-03-v1/source"
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" verify
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" report --output "$PWD/oct2_report.json"
```

Use a fresh output path outside the source snapshot. The release documentation also covers figure rebuilding and downloaded protein-input checks. These commands inspect released files and recompute aggregate statistics; they do not rerun neural predictions or training.

The original local raw reconstruction and full training completed. A complete public-download-only raw-to-training workflow has **not** been validated: historical workers retain private-reference, absolute-path and execution-evidence dependencies. Model weights, tokenizer payloads and restricted natural-language text/reversible inputs are not supplied in this release. No permanent DOI is claimed. Source-specific licenses and UniProt attribution are preserved; a code license does not relicense upstream data or omitted weights.

The earlier September matrix and other historical studies are distinct data/protocol versions. They are not relabelled as runs on the current data. Historical files and results remain in Git history and their existing locations; the separate `emergence` project is not merged into this release.
