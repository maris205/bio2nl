# Cover Letter — Nature Communications

Dear Editors,

We submit our manuscript, **"Asymmetric Structural Transfer Between Natural Language and
Biological Foundation Models,"** for consideration at *Nature Communications*.

**The question.** Cross-domain transfer—reusing representations learned in one domain to solve
tasks in another—is among the most consequential and least understood properties of foundation
models. A striking instance has been reported between human language and biological sequences:
models trained only on natural-language structure acquire, without biological pretraining, a
latent ability to discriminate homologous proteins. This has invited the interpretation that
abstract linguistic structure is a universal prior for biology. Yet a basic question has gone
unasked: **is this transfer symmetric?** If language and biology truly share a structural
substrate, transfer should operate in both directions. We show that it does not.

**What we find.** Through a systematic bidirectional study—fine-tuning across 100 seeds,
iso-token continued-pretraining with shuffled and random controls, model scaling, two
independent biological foundation-model families (ESM-2, ProtBERT), an adversarial
nested-structure task, and mechanistic analysis—we establish that structural transfer is
**bidirectional but strongly and universally asymmetric**. Language→biology transfer is robust
and strengthens with scale; biology→language transfer is weak, never exceeds matched-token
controls, and *decays* toward chance as protein models grow, with two independent model families
converging on identical chance-level behavior. In an architecture-matched analysis, a language
model retains four times more competence when moved to biology than a biological model retains
when moved to language.

**Why it matters, and why it is robust.** Our central comparisons use models whose training
corpora are fully documented (GPT-2, BERT, ESM-2, ProtBERT), which eliminates the
pretraining-contamination confound that clouds studies relying on proprietary large models—a
recurring and legitimate concern in this literature. Mechanistically, we rule out two intuitive
explanations (static representational alignment; an induced difference-detecting attention
circuit—a candidate we test rigorously and find does not survive), and localize the asymmetry to
learnability: biological discrimination emerges as a by-product of learning language structure,
but not the converse.

**Contribution.** The contribution is conceptual: the existence of shared structural regularities
between natural language and biological sequences does *not* imply symmetric representational
transfer, revealing an intrinsic directionality in cross-domain foundation-model learning. This
reframes a much-discussed phenomenon and offers a controlled, contamination-free methodology for
studying cross-domain transfer. We believe the breadth of the question—spanning machine learning,
computational biology, and the foundations of representation learning—fits the cross-disciplinary
readership of *Nature Communications*.

The manuscript has not been published elsewhere and is not under consideration by another journal.
All data and code are publicly available. We have no competing interests to declare.

Thank you for your consideration.

Sincerely,
Liang Wang
School of Artificial Intelligence and Automation, Huazhong University of Science and Technology
