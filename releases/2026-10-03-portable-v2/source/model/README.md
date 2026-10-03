---
language:
- en
tags:
- biology
- protein
- gpt2
- reproducibility
- research
---

# bio2nl: the 2026-10-02 model collection

This package describes **9 freshly pretrained GPT-2-style models and 27 complete
source-validation-selected protein-pair classifiers**. It includes the exact
shared 32,000-entry BPE tokenizer, all configuration/provenance identities and an
offline CPU loader. Weights are the original files, not converted or retrained.
The model repository is `dnagpt/bio2nl-models`. Pin an immutable repository
revision when downloading; the accompanying GitHub release records the verified commit.

## Scope and scientific interpretation

EP pretraining mixes English and natural protein, ES mixes the same English and
residue-shuffled protein, and EE uses English plus additional English. Each run
uses 16,777,216 input tokens and 1,024 updates. Each condition has three
pretraining seeds. Each pretrained model has three source fine-tuning seeds.
Fine-tuning uses 8,044 protein sequence-pair training examples for five epochs;
minimum source validation cross-entropy selects the saved epoch, ties earlier.
Every selected checkpoint includes the fine-tuned backbone and its original
two-logit classifier head. No English pair labels trained or selected these heads.

These are exploratory research models. Protein labels measure the experiment's
operational sequence similarity, not verified structural or remote homology.
The current EP source-test accuracy is about 50.8%; its English transfer results
do not demonstrate reliable fixed-head sentence-similarity classification. A
modest exploratory ranking difference is reported in the accompanying study.
The CPU examples here are software checks, not new scientific evaluations.

Source and result release:
[GitHub immutable source](https://github.com/maris205/bio2nl/tree/0d91df978ab5ac8604a9216bb5a6036605aea910/releases/2026-10-03-v1/source)
and [Hugging Face protein/code release](https://huggingface.co/datasets/dnagpt/bio2nl/tree/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1).

## File identities and model identifiers

The trusted `model_catalog.json` SHA-256 is:

```
5f6694f206084ad51dcc27c1485c4b4d000ffe5023310482aab367d0815df067
```

The catalog binds every file's size/SHA, complete tensor-state digest, backbone
and head digests, full config, tokenizer, training protocol and prepared-data
manifest. Classifier entries also bind the source-selected epoch and the exact
pretraining parent. These hashes are independently projected from the frozen
training/source-selection records; a digest merely declared by a checkpoint is
not trusted.

| Identifier | Artifact |
|---|---|
| `EP-pt0` | Pretrained causal LM, condition EP, pretraining seed 0 |
| `EP-pt0-ft0` | Complete source-selected EP classifier, pretraining seed 0, fine-tuning seed 0 |
| `ES-pt1-ft2` | Complete source-selected ES classifier, pretraining seed 1, fine-tuning seed 2 |

Conditions are `EP`, `ES`, `EE`; both seed indices range from 0 to 2. All 36
weights total **15,845,506,290 bytes**. One checkpoint is about 440 MB. A
classifier is self-contained: its parent pretraining weight need not be downloaded
for inference. `upload_mapping.json` is a maintainer inventory with relative
original paths; it is not used by the public loader.

## Download only the selected model

Use the immutable model-repository commit recorded by the publication receipt.
Download `inference.py`, `portable_core.py`, `model_catalog.json`, `config.json`,
`tokenizer.json`, `example_pairs.json`, and one checkpoint, for example
`checkpoints/source/EP/pt0/ft0/best.pt`. Preserve this relative layout. The
`--checkpoint` option supports a differently located weight while keeping all
digest and provenance checks.

For example, after installing `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dnagpt/bio2nl-models",
    revision="<immutable model-repository commit from the publication receipt>",
    local_dir="bio2nl-models",
    allow_patterns=["inference.py", "portable_core.py", "model_catalog.json",
                    "config.json", "tokenizer.json", "example_pairs.json",
                    "checkpoints/source/EP/pt0/ft0/best.pt"],
)
```

## Offline verification and inference

The CPU checks used Python 3.12.3 and imported PyTorch 2.3.0+cu121.
The local distribution metadata incorrectly reported torch 2.9.1; see
`ENVIRONMENT_NOTE.json`. The requirements describe the observed import versions,
not a separately verified clean installation or a proven historical training binary.
Run in the directory containing the downloaded package. No network, credentials,
GPU, training acceptance, historical workspace path or benchmark examples are
required by the inference commands.

```bash
python inference.py verify \
  --catalog model_catalog.json \
  --catalog-sha256 5f6694f206084ad51dcc27c1485c4b4d000ffe5023310482aab367d0815df067 \
  --artifact-root . --model-id EP-pt0-ft0

python inference.py predict \
  --catalog model_catalog.json \
  --catalog-sha256 5f6694f206084ad51dcc27c1485c4b4d000ffe5023310482aab367d0815df067 \
  --artifact-root . --model-id EP-pt0-ft0 --pairs example_pairs.json
```

Input is a JSON list of objects with exactly `sentence1` and `sentence2`. The
program refuses labels, empty endpoints, unknown/special content tokens, failed
text roundtrips and hash/provenance mismatches. It independently caps each
endpoint at 255 content tokens, emits `a + SEP + b + EOS`, right-pads with ID 0,
and pools the last attention-mask-valid position. Output reports truncation,
both log probabilities, probability for fixed label 1, the label-1 minus label-0
margin, and uncalibrated argmax predictions (ties label 0). No polarity reversal,
threshold adjustment, target calibration or pooling substitution occurs.

`verify` also supports pretraining IDs such as `EP-pt0`; pair classification
requires a `-ft` classifier. Programmatic `load_model` returns a standard
`GPT2LMHeadModel` for a pretraining ID, or the complete `TransferClassifier`.

The portable example uses **CPU FP32**. The study's held-out scoring used FP32
weights with CUDA BF16 autocast. These regimes need not produce bitwise-equal
scores. `cpu_replay_audit.json` checks the portable implementation against the
original frozen implementation under the same CPU FP32 regime, using one source
training protein pair and two handwritten English pairs. It does not recompute
the study's benchmark tables.

## Validation and maintenance

`portable_core.py` is a byte-identical copy of the original frozen model module
(SHA-256 `b2362edf8850fe20fbd23cc958a71f92ac2a187de4316f53f8750d3bca857592`).
`catalog_build_audit.json` records independent file/state/config/finiteness
verification of all 36 original checkpoints. `test_inference.py` covers file and
catalog tampering, wrong parent/job/epoch/config, missing or malformed heads,
nonfinite states, tokenizer tampering, path escapes and endpoint truncation.

```bash
python -m unittest discover -s . -p test_inference.py -v
```

`build_catalog.py` and `validate_cpu_replay.py` are maintainer-only provenance
tools that take an explicit original workspace root. They are not needed by
users loading this public package. No weight is copied by the catalog builder.

## Data attribution and licensing status

The protein inputs derive from UniProtKB/Swiss-Prot with sequence-source
attribution in the accompanying protein release. English pretraining uses the
documented OpenWebText selection. The experiment-generated shared BPE vocabulary
was trained on 16 MiB each of English and protein training text; it is reused from
the historical rebuilt v2 tokenizer. The tokenizer is not an independently
trained model for every later holdout. Corpus membership and exposure limitations
remain those described in the study and data card.

**No model-weight license has been assigned in this release.** No license grant
is inferred from training-data licenses or code provenance. Accordingly, this
card deliberately omits the Hugging Face `license` metadata field. Public
availability alone does not establish unrestricted redistribution or commercial
use rights. Any later license assignment must be a separate documented change.
