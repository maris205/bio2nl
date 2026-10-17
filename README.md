# Bio2NL: transfer between English and protein pairs

This project studies direct pair-classifier transfer in both directions using official GPT-2 and a separate from-scratch shared-BPE experiment. The **October 2026 rebuilt release** below is the current data/model entrypoint. Historical experiment and manuscript files remain preserved; they do not automatically describe the current results.

## Current data and models

| Resource | Public repository | Contents |
|---|---|---|
| Task data and experiment code | [dnagpt/bio2nl-data-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d) | 46 loadable configurations: 41 with actual sequence/sentence pairs and 5 retaining English reconstruction metadata; exact FT/test inputs, protocols and results |
| Published model bank | [dnagpt/bio2nl-models-2026-10](https://huggingface.co/dnagpt/bio2nl-models-2026-10) | 313 complete checkpoint files, approximately 141.72 GiB; per-model data/seed/format/hash records |
| Pretraining materials | [dnagpt/bio2nl-pretraining-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10) | Selected English common/extra text, natural/shuffled protein corpora, protein token streams, shared tokenizer and exact token reconstruction |

Use the **[FT and test data guide](docs/FT_AND_TEST_DATA.md)** to identify the exact input for each experiment and the **[data and model guide](docs/DATA_AND_MODELS.md)** for pinned revisions, model-to-data mapping and loading examples. Machine-readable pins are in [releases/2026-10/resources.json](releases/2026-10/resources.json). The separate [pretraining guide](docs/PRETRAINING.md) explains corpus sizes, input budgets, direct English pretraining-text downloads and token reconstruction. The English relatedness evaluation pairs are also directly downloadable; their original metadata and optional reconstruction route remain available.

The October 17 task-data completion adds the exact gap-cleaned Protein-PAWS Reference (20,000 archived rows), the 99,818-row standard-reference feature-baseline training split, and the remote identity-matched validation/test subsets (1,696/1,832). All four original English relatedness bands and the expanded 2,000-pair target now provide actual sentence pairs and labels in five additional configurations, alongside the unchanged reconstruction metadata. See the [complete task index and verification tools](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17). This update publishes task data, not additional model weights: the PP pretrained model and its 20 FT checkpoints, and 20 whole-network random controls, remain local.

## Experiment map

- **English → protein:** 100 official GPT-2 classifiers, clean English PAWS-X FT seeds 1–100; standard rebuilt protein targets and 4 positive identity bands × 3 negative strategies × 2 BLAST views. The separate gap-cleaned Reference uses the retained seeds 0–19; its protein train split is archived, not used as the FT input.
- **Protein → English:** five ordinary V2 protein groups × 20 seeds; remote-labelled and measured functional-variant sources × 20 seeds each; English cosine-relatedness groups and PAWS-X remain distinct targets.
- **From scratch:** EP (English + protein), ES (English + shuffled protein) and EE (English + extra English), one pretraining seed per condition, followed by the same protein FT with 20 fine-tuning seeds each. The PP (protein-only parameter pretraining) follow-up uses the same shared tokenizer; its weights are not included in the public model bank.

FT means supervised classification fine-tuning of both the pretrained backbone and its classification head. Historical configuration names containing `sft_` are retained for compatibility.

Results are exploratory and task-dependent. We retain raw and validation-oriented accuracy, all prescribed seeds, matched initial controls where evaluated, and the same selected seed's existing confirmation score. A label flip is fixed on validation, not selected to maximize held-out accuracy. High English cosine-proxy scores are not human-paraphrase accuracy. Weak source validation, reused confirmation data, different input/training recipes and the stronger EE control remain explicit. Fine-tuning seeds are not independent pretraining runs.

## Reproduction and historical material

Current task-data verification, English reconstruction, original construction/training snapshots and per-seed results are packaged with the task-data repository. The model repository provides format-specific loaders. The pretraining companion provides a portable corpus/token reconstruction helper. Published file identities and representative downloads/loaders were verified; a complete fresh public-only GPU rerun of every experiment is not claimed.

The [October 3 release](docs/release_2026-10-03_historical.md) remains available for its original nine-pretraining-run/27-classifier protocol. Its seed hierarchy and scores are historical and must not be mixed with the October 8 EP/ES/EE experiment. Earlier `paper/` files are preserved historical manuscripts; this documentation update does not publish private revision drafts or reviewer correspondence. Source-specific licenses and attribution are recorded in each Hugging Face repository.
