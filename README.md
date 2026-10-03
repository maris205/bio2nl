# bio2nl: protein-pair supervision and English transfer

This repository studies whether protein sequence-pair supervision transfers to English sentence-pair prediction. The current audited experiment uses nine fresh pretraining runs, 27 full source fine-tuning runs, and fixed evaluation of all 27 source-selected classifiers. Historical root-level experiments and paper files remain unchanged; the versioned release below identifies the current source and access instructions.

## Current portable release

- [Portable reproduction instructions](releases/2026-10-03-portable-v2/source/README.md) and [exact source manifest](releases/2026-10-03-portable-v2/source/MANIFEST.json).
- [The same versioned source archive on Hugging Face](https://huggingface.co/datasets/dnagpt/bio2nl/tree/b333517eb3f03886ab92215175653a5c87b5f7cf/releases/2026-10-03-portable-v2), dataset revision `b333517eb3f03886ab92215175653a5c87b5f7cf`.
- [Pinned biological inputs and tokenizer](https://huggingface.co/datasets/dnagpt/bio2nl/tree/5c5692107582866a3984973f83a2ac27d9a1ea89/releases/2026-10-03-portable-v2), dataset revision `5c5692107582866a3984973f83a2ac27d9a1ea89`.
- [All 9 pretrained models and 27 complete source-selected classifiers](https://huggingface.co/dnagpt/bio2nl-models/tree/9fdd0794820ef4534b7f97fcd2c7a2893aba7a9d), model revision `9fdd0794820ef4534b7f97fcd2c7a2893aba7a9d`. Weights are separate downloads, approximately 15.85 GB; they are not in this Git repository.
- [Current fixed-transfer results](releases/2026-10-03-portable-v2/source/study_source/results/oct2/fixed_transfer/REPORT.md) and [Source Data](releases/2026-10-03-portable-v2/source/study_source/paper_assets/source_data/README.md).

The release retains all 190 previously published source files under `study_source/` and adds public acquisition, local input preparation, guarded source-training and standalone CPU model-loading tools. English/QQP text is obtained from fixed upstream versions and reconstructed locally under original terms. Raw or reversible NLP inputs, row predictions and private review correspondence are not redistributed in this source release.

## Experiment and interpretation

EP combines English with natural protein exposure; ES uses the same English blocks and shuffled protein; EE uses twice the English exposure at the same total token budget. Each condition has three independent pretraining seeds, each followed by three source fine-tuning seeds. Statistics first average the fine-tuning repeats within each pretraining seed, then report the mean and sample SD across the three pretraining seeds.

The natural-protein condition has a modest exploratory QQP ranking gain over its shuffled control: paired AUC difference **+0.029560 ± 0.018655**, positive for all three pretraining-seed groups. Its protein-test accuracy is **50.76%**, and English balanced accuracy is **50.18%**. Weak source learning and fixed English classification limit interpretation. These results do not establish statistical significance, transferable protein-relation semantics, a general directional asymmetry, or impossibility of transfer. The QQP cohort was evaluated historically and is not a new blind test.

Protein examples are constructed directly from pinned UniProtKB/Swiss-Prot records using normalization, MMseqs2 search/clustering, split assignment and explicit pairing rules. Labels describe **operational sequence similarity**, not certified structural homology or evolutionary independence. Pair splits contain 99,818 / 20,276 / 20,854 rows; the experimental source training subset contains 8,044 rows. The budget-matched surface reference uses BPE-token statistics; raw-residue construction baselines use all 99,818 training rows and are reported separately. Equal BPE-token budgets do not guarantee equal residue or parent-record exposure.

## Reproduction scope

Read the [versioned README](releases/2026-10-03-portable-v2/source/README.md) for installation and exact commands. CPU input preparation reproduces the 22 locked scientific payloads. The frozen training algorithms and budgets are preserved, but a new complete GPU training run and second-machine reproduction have not been demonstrated by this publication.

The current environment imports PyTorch 2.3.0+cu121 while installed package metadata reports 2.9.1. The portable compiler refuses normal GPU execution under that mismatch; an explicit CPU-validation-only protocol permanently disables its GPU paths. The exact October 2 imported binary is not established by archived package metadata. Original weights and recorded results remain unchanged.

The model readback checks all 36 remote LFS identities and physically downloads 2 prespecified weights, plus all 18 supporting files. The downloaded PT and classifier pass standalone CPU loading and an unlabeled toy-pair check. This is a technical loading check, not new benchmark evidence.

No permanent DOI or blanket model-weight license is asserted. Source-specific attribution and terms apply. Earlier releases remain at [2026-10-03-v1](releases/2026-10-03-v1/README.md); the historical September experiments and the separate `emergence` project are not relabelled as current results.
