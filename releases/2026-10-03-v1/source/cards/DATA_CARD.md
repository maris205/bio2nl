# Biological data candidate: deterministic raw-v2 and source subsets

Upstream: UniProt Consortium, UniProtKB/Swiss-Prot release 2026_03. See PROTEIN_ATTRIBUTION.md for attribution, CC-BY-4.0 source terms, other-rights caveats and transformation details. No NLP text is included.

The actual separate data package has eight payloads: protein corpus train/validation/test; operational protein pair train/validation/test; the exact experimental protein source train/validation JSONL. Raw pair rows are 99,818 / 20,276 / 20,854 (140,948 total). Experimental training selects whole balanced blocks for 8,044 rows; validation is all 20,276 rows. The data have real composition/k-mer predictive cues and operational cluster isolation, not proven absence of all evolutionary overlap.

Original raw manifest: c65ad951dced3285856f8edf49618e3d6b5329fc4ad006520dab748d35b407bf. Prepared manifest: 2ea23ffd2793ed2142156f2942cc4bfb77767a0c3e87ba668a77c34a3a16c010. The original raw gate remains training-disabled; a separate subsequent source-training acceptance authorized the full run. This data package does not change either gate or authorize a new run.

File identities and original relative paths are in metadata/protein_assets.json and the data package manifest. All eight payloads were hash-verified against those current identities after copy-on-write/fallback copying; original files were not edited. The exact source subset strings were checked for the canonical 20-letter amino-acid alphabet. This packaging check is not a new scientific acceptance audit.

Natural-language resources are local-input-only acquisition recipes, never part of this data payload. No raw SCOPe sequence, synthetic task data, old full private archive, tokenizer or model is included. Current model/publication conclusions refer to this new version, while historical matrices remain separately identified.

HF dataset destination and revision: not published / unset. Attribution review is recorded; no claim of an unrestricted grant for every possible downstream right is made.
