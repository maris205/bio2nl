# Source Data for the 2026-10-03 revision candidate

This directory contains aggregate/model-level metrics from the frozen 2026-10-02 experiments, plus separately identified construction diagnostics. It contains no natural-language examples or reversible input token arrays. `export_manifest.json` records the actual input and output SHA-256 values; `export_source_data.py` verifies the final-report bindings before export and does not load or score a model.

- `SourceData.xlsx`: ten sheets (dictionary plus nine complete CSV tables).
- `transfer_per_job_metrics.csv`: all 54 neural held-out scoring cells, including all three PT and three FT seeds for both domains.
- `transfer_nested_metrics.csv`: FT means within PT, then mean and sample SD across the three PT seeds; also retains within-PT FT SD.
- `transfer_paired_contrasts.csv`: primary EP−ES and contextual EP−EE/ES−EE effects for every reported metric and both roles.
- `transfer_reference_metrics.csv`: source-selected nine-feature BPE reference and constant0/constant1 in both roles. The reference has no seed SD. Infinite constant-prediction CE has an explicit reason in `transfer_summary.json`.
- `source_epoch_metrics.csv`: all fixed-checkpoint training/validation evaluations for all five source epochs; online dropout training CE is separately labelled.
- `source_per_job_metrics.csv`: validation-CE-selected (`best`) and final-epoch (`final`) source fits. Only the selected complete classifier is used for transfer.
- `source_nested_metrics.csv`, `source_paired_contrasts.csv`: complete source fit aggregation; validation includes model-selection bias.
- `raw_residue_baselines_99818_train.csv`: raw-residue construction probes fitted on 99,818 train rows, separate from the 8,044-row neural and BPE-reference budget. Alignment identity partly defines source labels; its discrimination is a construction diagnostic, not independent learned competence.
- `source_summary.json`, `transfer_summary.json`, `design.json`: machine-readable figure and methods inputs. The source-only summary's disabled held-out flags describe that earlier stage; a separate protocol authorized the later transfer run.

Metrics are fractions in CSV/JSON/XLSX (accuracy 0–1), with plots converting specified axes to percentages; cross-entropy is in nats. EP=English+natural protein; ES=the same English+shuffled protein; EE=double English exposure at the same total token budget. Each of three PT seeds has three FT repeats; there are three independent pretraining replicates, not nine. All split identities are fixed. Error bars represent sample SD across PT means, not confidence intervals. No hypothesis tests are added here.

The English target is the historically evaluated 39,893-pair QQP cohort, freshly rebuilt from the same raw source and frozen membership rules. The source test contains 20,854 pairs from raw-v2. These are exploratory comparisons, not a new blind confirmation.

To regenerate the plots, run each `../figures/gen_fig*.py` with Python, NumPy and Matplotlib. They read only this directory and write vector PDFs and PNG previews beside the scripts. The workbook/CSV export requires OpenPyXL and the exact frozen local reports specified by `export_source_data.py --workspace ... --output ...`.
