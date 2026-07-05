# bio2nl — All Result Tables (auto-generated)

## Table 1 — Bidirectional symmetry (GPT-2, identical protocol)
direction  transfer(flip)     n
NL->Bio    0.680 ± 0.100      15
Bio->NL    0.538 ± 0.011      15
(NL->Bio majority≈0.50 protein; Bio->NL majority=0.545 PAWS-X)

## Table 2 — Reverse (Bio->NL) FT, 100-seed distribution on PAWS-X
protein in-domain : 0.991 ± 0.006
transfer -> PAWS  : 0.543 ± 0.008  (majority 0.5465)
untrained baseline: 0.544 ± 0.008
seeds beating baseline by >0.05: 0/100  (n=100)

## Table 3 — Iso-token CPT ablation, PAWS-X sample-efficiency (3 seeds)
  [124M]  (protein−shuffled isolates STRUCTURE; protein−text isolates BIOLOGY-vs-tokens)
    frac0.05: base=0.604 text=0.698 protein=0.645 shuffled=0.559  Δ(p−s)=+0.087 Δ(p−t)=-0.053
    frac0.1: base=0.777 text=0.802 protein=0.789 shuffled=0.675  Δ(p−s)=+0.114 Δ(p−t)=-0.012
    frac1.0: base=0.909 text=0.910 protein=0.902 shuffled=0.900  Δ(p−s)=+0.002 Δ(p−t)=-0.008
  [355M]  (protein−shuffled isolates STRUCTURE; protein−text isolates BIOLOGY-vs-tokens)
    frac0.05: base=0.826 text=0.839 protein=0.832 shuffled=0.831  Δ(p−s)=+0.001 Δ(p−t)=-0.008
    frac0.1: base=0.878 text=0.879 protein=0.884 shuffled=0.884  Δ(p−s)=+0.001 Δ(p−t)=+0.005
    frac1.0: base=0.930 text=0.932 protein=0.929 shuffled=0.930  Δ(p−s)=-0.000 Δ(p−t)=-0.002

## Table 4 — 2x2 cross-modal transfer matrix (acc, 3-seed mean)
backbone    pretrain  ->protein   ->PAWS-X
esm2_8M     protein   0.999       0.641
bert_small  English   0.656       0.732
bert_tiny   English   0.636       0.583
  off-domain drop: language-model(bio) +0.076 | bio-model(NL) +0.358
(majority: protein 0.50, PAWS-X 0.545)

## Table 5 — Scaling on PAWS-X: language scales UP, bio->language DECAYS (3 seeds)
  ESM-2 ladder (protein):
    esm2_8M        8M  0.641 ± 0.052
    esm2_35M      35M  0.685 ± 0.005
    esm2_150M    150M  0.627 ± 0.056
    esm2_650M    650M  0.569 ± 0.000
  Cross-family large protein LMs (universality):
    esm2_650M    650M  0.569 ± 0.000
    protbert     420M  0.569 ± 0.000
  BERT ladder (English):
    bert_tiny      4M  0.583 ± 0.003
    bert_small    11M  0.732 ± 0.018
    bert_base    110M  0.911 ± 0.008

## Table 6 — ESM-2 is STRUCTURAL-only (above chance on PAWS, collapses on CoLA/RTE)
  paws-x   esm2_8M=0.641  bert_small=0.732
  cola     esm2_8M=0.000  bert_small=-0.014
  rte      esm2_8M=0.528  bert_small=0.593

## Table 7 — Hard structural task (Dyck): protein>shuffled does NOT hold (3 seeds)
  gpt2        dyck_L40_t3 f1.0: protein=0.601 shuffled=0.685 text=0.761  Δ(p−s)=-0.085
  gpt2        dyck_L60_t3 f1.0: protein=0.549 shuffled=0.541 text=0.598  Δ(p−s)=+0.008
  gpt2-medium dyck_L40_t3 f1.0: protein=0.689 shuffled=0.681 text=0.759  Δ(p−s)=+0.008
  gpt2-medium dyck_L60_t3 f1.0: protein=0.531 shuffled=0.535 text=0.635  Δ(p−s)=-0.005

## Table 8 — Mechanism (M1 CKA / M2 dynamics / M3 difference-heads)
  M2 learning dynamics (transfer start->end during training):
    NL->Bio: 0.630 -> 0.710
    Bio->NL: 0.533 -> 0.535
  M3c rigorous difference-head z-scores (diff-head is pretraining artifact, not induced):
    base         NL      max=0.0162 z=3.08 head L0H5
    base         protein max=0.00375 z=1.96 head L0H11
    paws_tuned   NL      max=0.01667 z=2.83 head L0H5
    paws_tuned   protein max=0.00396 z=2.18 head L0H11
    protein_cpt  NL      max=0.01618 z=2.88 head L0H5
    protein_cpt  protein max=0.0043 z=2.12 head L7H7
