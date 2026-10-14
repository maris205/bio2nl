# Bio2NL: transfer between English and protein pairs

This project studies direct pair-classifier transfer in both directions using official GPT-2 and a separate from-scratch shared-BPE experiment. The **October 2026 rebuilt release** below is the current data/model entrypoint. Historical experiment and manuscript files remain preserved; they do not automatically describe the current results.

## Current data and models

| Resource | Public repository | Contents |
|---|---|---|
| Task data and experiment code | [dnagpt/bio2nl-data-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10) | 34 loadable configurations; BLAST pair views, SFT sources, PAWS-X, remote and functional sources; English proxy reconstruction, protocols and results |
| Complete model bank | [dnagpt/bio2nl-models-2026-10](https://huggingface.co/dnagpt/bio2nl-models-2026-10) | 313 complete checkpoint files, approximately 141.72 GiB; per-model data/seed/format/hash records |
| Pretraining materials | [dnagpt/bio2nl-pretraining-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10) | Natural/shuffled protein corpora, actual token streams, shared tokenizer, exact English selection and reconstruction |

Use the **[data and model guide](docs/DATA_AND_MODELS.md)** for pinned revisions, model-to-data mapping, loading examples and the scope of reproduction. Machine-readable pins are in [releases/2026-10/resources.json](releases/2026-10/resources.json). The separate [pretraining guide](docs/PRETRAINING.md) explains corpus sizes, input budgets and why English text is reconstructed from its original source.

## Experiment map

- **English → protein:** 100 official GPT-2 classifiers, English PAWS-X SFT seeds 1–100; standard rebuilt protein targets and 4 positive identity bands × 3 negative strategies × 2 BLAST views.
- **Protein → English:** five ordinary V2 protein groups × 20 seeds; remote-labelled and measured functional-variant sources × 20 seeds each; English cosine-relatedness groups and PAWS-X remain distinct targets.
- **From scratch:** EP (English + protein), ES (English + shuffled protein) and EE (English + extra English), one pretraining seed per condition, followed by the same protein SFT with 20 fine-tuning seeds each.

Results are exploratory and task-dependent. We retain raw and validation-oriented accuracy, all prescribed seeds, matched initial controls where evaluated, and the same selected seed's existing confirmation score. A label flip is fixed on validation, not selected to maximize held-out accuracy. High English cosine-proxy scores are not human-paraphrase accuracy. Weak source validation, reused confirmation data, different input/training recipes and the stronger EE control remain explicit. Fine-tuning seeds are not independent pretraining runs.

## Reproduction and historical material

Current task-data verification, English reconstruction, original construction/training snapshots and per-seed results are packaged with the task-data repository. The model repository provides format-specific loaders. The pretraining companion provides a portable corpus/token reconstruction helper. Published file identities and representative downloads/loaders were verified; a complete fresh public-only GPU rerun of every experiment is not claimed.

The [October 3 release](docs/release_2026-10-03_historical.md) remains available for its original nine-pretraining-run/27-classifier protocol. Its seed hierarchy and scores are historical and must not be mixed with the October 8 EP/ES/EE experiment. Earlier `paper/` files are preserved historical manuscripts; this documentation update does not publish private revision drafts or reviewer correspondence. Source-specific licenses and attribution are recorded in each Hugging Face repository.
