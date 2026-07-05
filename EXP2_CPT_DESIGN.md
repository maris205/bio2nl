# Exp 2 — Iso-Token Continued-Pretraining Ablation (bio → NL structural reasoning)

Synthesis of `gemini.md` (The Evolutionary Prior) + `gpt.md` (Bio-CPT). This is the
**real** reverse-transfer experiment — and it differs fundamentally from Exp 01–04.

## Why Exp 01–04 could not have succeeded (and what changes)

| | Exp 01–04 (done) | Exp 2 (this design) |
|---|---|---|
| Bio-learning vehicle | **classification fine-tuning** (add a `score` head, train on protein pairs) | **continued pretraining** (next-token LM on bio sequences, keep the LM head) |
| Effect on the model | fully re-specializes the body to amino-acid classification; **destroys the NL space** | *nudges* the LM's representations while preserving language generation |
| NL after bio | nothing NL-shaped survives → transfer undefined | NL ability retained → can measure whether structure *improved* |
| Result | majority-class collapse, no transfer (robust across 100 seeds, 3 backbones, 2 source difficulties) | **open question — the actual thesis** |

So Exp 01–04 is not a failed thesis; it's the **negative control** proving that *destroying* language kills transfer. Exp 2 asks the real question: does *adding* biology (without destroying language) *improve* NL structural reasoning?

## Core hypothesis
Biological sequences encode evolutionarily-compressed structural constraints (long-range
dependency, low-entropy topology, strict motif grammar). Continued-pretraining a language
model on them may improve NL tasks requiring **structural discrimination** — paraphrase,
word-order sensitivity, grammatical acceptability, long-range dependency — *even if it does
not lower NL perplexity*. The expected win is in **sample-efficiency on structural tasks**,
not fluency.

## The iso-token ablation (the whole paper rests on the controls)

Start from stock **GPT-2 small (124M)**. Same tokenizer for all groups (proteins as
space-separated residues: `<protein> M K V K ... </protein>` so BPE doesn't make giant chunks).
**Every group sees the SAME token budget, SAME steps, SAME optimizer/batch** (iso-token).

| Group | CPT data | Purpose |
|---|---|---|
| **G0** GPT2-base | none | baseline |
| **G1** Text-CPT | natural text (OpenWebText) | "extra training" control — the key baseline |
| **G2** Protein-CPT | real Swiss-Prot/UniRef proteins | **main test** |
| **G3** Shuffled-Protein | each protein's residues shuffled | kills structure, keeps AA freq/length — *the anti-regularization-illusion control* |
| **G4** Random-AA | AA sampled from background freq | kills all sequence form |
| **G5** Markov-AA | k-mer Markov over AAs | keeps LOCAL stats, kills long-range structure |
| **G6** DNA-CPT | real DNA | is it proteins specifically, or life-sequences generally? |
| **G7** Mix (90/10) | 90% text + 10% protein | the realistic LLM-recipe question |

**Decision logic (what makes it publishable):**
```
G2 (Protein) > G1 (Text)          -> biology beats "just more tokens"
G2 > G3 (Shuffled)                -> the gain is STRUCTURE, not AA frequency
G2 > G4 (Random), G2 > G5 (Markov)-> gain needs LONG-RANGE structure, not local k-mers
```
If only `G2 > G0` but `G2 ≈ G1`, it's just the generic CPT effect — not a story.

## Evaluation — structural NL, three protocols (from gpt.md)
Do NOT lead with GLUE average. Lead with **structure-sensitive** tasks:
- **Paraphrase / equivalence:** PAWS-X, MRPC, QQP (↔ homology)
- **Word-order sensitivity:** PAWS, HANS (↔ mutation / order)
- **Grammatical acceptability:** CoLA (↔ sequence legality)
- **NLI / consistency:** RTE, MNLI, ANLI (↔ structural consistency)
- **Long-range dependency:** LAMBADA / synthetic (↔ protein long-range contacts)

Three measurement modes, in order of robustness:
- **A. Zero-shot LL scoring** — compare `P(yes)` vs `P(no)` (clean but GPT-2 instruction-weak, noisy).
- **B. Frozen linear probe** — freeze the CPT'd model, probe `[EOS]` hidden state with a linear classifier. Tests whether NL structure is *more linearly separable* after bio-CPT.
- **C. Sample-efficiency fine-tune** — fine-tune on 1% / 5% / 10% / 100% of each task. **The most robust; the expected headline: bio-CPT wins most in the low-resource regime.**

Headline metric:
```
Structural-NLP Score = mean normalized(PAWS-X, HANS, CoLA, RTE, LAMBADA, adversarial)
Bio Structural Gain   = Score(G2) - max(Score(G1), Score(G3), Score(G4))
```

## Bio-side sanity (Exp 5 in gpt.md)
Confirm the CPT'd model actually absorbed biology: protein homology, remote homology,
solubility — reuse BioPAWS. (Otherwise a null NL result is uninterpretable.)

## Scaling (gemini.md stage 3)
Repeat the ablation at **124M → 355M → (stretch) 1B**. The dream figure: G3/G4 (shuffled/random)
plateau while G2 (real bio) keeps gaining or shows a phase transition vs G1. This is where the
"bigger models" step comes in — do it AFTER the 124M ablation shows a trend.

## Mechanism (both docs; needed for a top venue)
- **Representation geometry:** t-SNE/UMAP + linear-probe margin + CKA, on paraphrase/entailment/grammaticality. Does bio-CPT make NL structure more separable, mainly in mid/high layers?
- **Universal difference heads:** does bio-CPT *induce* attention heads that flag NL perturbations (subject-object swap) the way they flag protein mutation sites? (Mirror of the forward paper's "difference operators.")
- **Manifold alignment:** does an ungrammatical sentence and an unfoldable mutant protein cause the same-direction latent perturbation?

## Minimal viable first cut (what to run now)
Groups **G0, G1, G2, G3, G4** at 124M, **50M CPT tokens, ctx 512, 3 seeds**, matched steps.
Eval: PAWS-X, HANS, CoLA, RTE (+ probe + sample-efficiency). If a trend appears → scale up.

## Expected outcomes (gpt.md)
1. Best: `G2 > G1 > G0 > G3/G4` → "evolutionary structure enhances NL structural reasoning."
2. Likely: no zero-shot gain, but **G2 wins in low-resource fine-tuning** → "structural sample-efficiency prior." Still a strong paper.
3. Null: `G2 ≈ G1` on NL (only BioPAWS improves) → small BioGPT2 paper, not the big claim.
4. Negative: `G2 ≈ G3/G4` → gain was token noise / regularization, not biology.
