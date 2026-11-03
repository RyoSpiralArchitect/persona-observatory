# Research history and preserved scope

This repository begins with the available completed evidence for Experiments 8
and 9. Earlier exploratory work motivated these studies, but a complete release
of Experiments 1–7 is not part of this initial repository. Their results are not
reconstructed from conversation or presented as newly verified experiments.

## Research question

Can generated responses, read through a declared observation procedure, be
encoded as a useful concept of a model's behavior under a constraint or
orientation? The operational test here is forecasting new observable response
codes. A forecast's usefulness does not identify a unique hidden mechanism,
recover the actual reasoning trajectory, or establish a persistent personality.

The preserved studies use `HuggingFaceTB/SmolLM2-360M-Instruct`, revision
`a10cc1512eabd3dde888204e902eca88bddb4951`. They generated separate fresh chats
with authored English questions. Their predictors use observed response codes
and known design conditions rather than hidden states or token logits as
features. Ordinary text generation still samples from the model's token
distribution.

## Experiment 8: allocate observations across topics

**Executed study:** 96 fit outputs, 336 support outputs and 144 target outputs;
576 generations in total. The 36 questions were divided into 8 fit, 16 support
and 12 target questions. Each phase shared four designed evidence structures.

The study compared observing a preassigned single topic twice (S2), observing
the same topic eight times (S8), and observing four topics twice each (D8).
Support forecasts mixed 80% of a common reference Q with 20% observed frequency.

On new target questions, D8 improved categorical Brier over S2 and S8 in both
observer branches. The pooled-frequency predictor G nevertheless had the best
categorical Brier and NLL among the ten fixed candidates. The strict event of
speaker-attributed explicit evidence limitation did not show the same uniform
benefit. The study also separated the empirical comparisons from the algebraic
Brier benefit of averaging predictions.

**Preserved source:** the embedded `persona_topic_budget_pilot/REPORT_JA.md`,
`PROTOCOL.md`, raw data, predictions and final audit inside
`archives/experiment_09_20261103.zip` →
`persona_shrinkage_pilot/provenance/source_v8.zip`.

## Experiment 9: globally centered shrinkage

The next study used the same balanced development pool for all predictors:
28 old questions and 336 outputs. This included the preceding 144 target
outputs, explicitly reassigned to development. New evaluation used 24 other
questions × 6 familiar instruction settings × 2 sampled outputs = 288 outputs.
Question blocks, including every setting and replicate of each question, were
held together in seven-fold development cross-validation.

Five observed outcomes were split into direct-answer eligibility D and the
conditional composition of support/reject/withhold/mixed judgments. F used a
common shrinkage strength; H could select separate strengths for the two parts.
G, H_gate, H_content, H_evidence and Hswap were fixed comparators. Strengths and
every target forecast were locked before new output generation.

| Recorded comparison | Five-category Brier difference, A/B mean |
|---|---:|
| H − G | −0.001445498586 |
| H − F | +0.001560511900 |

The co-primary criterion required both differences to be negative and was
**not met**. Each contrast had the same direction in both observer branches.
H_gate was best among the seven fixed candidates in categorical Brier and NLL
in each branch. H_gate−G was not an independently designated primary contrast;
its rank motivates a future test without replacing the H criterion.

H−G's NLL difference decomposed into −0.007023809161 from D and
+0.026873264255 from conditional composition, for a net worsening of
+0.019849455093. That exact NLL decomposition is not a decomposition of Brier
and is not evidence for two neural or psychological modules.

Source: [the preserved report](../research/experiment_09/REPORT_JA.md),
[methods](../research/experiment_09/METHODS_JA.md),
[primary criterion](../research/experiment_09/results/primary_criterion.json)
and [NLL table](../research/experiment_09/results/nll_decomposition.csv).

## Measurement and scope

A/B used the same foundation model and rubric in separate fresh contexts. They
are two measurements of the same subject outputs, not independent human raters
or independent subject-model replications. In Experiment 9, the five-category
codes agreed for 284/288 outputs; all six underlying codes agreed for 254/288.
Those agreement rates do not establish semantic ground truth.

The scores concern the judgments expressed by the model. Correctness of those
judgments against the scenario evidence is a different target. Direct-answer
eligibility is also different from assertiveness, verbosity, or consciousness.
These are small authored benchmarks with familiar evidence structures. Reported
differences are descriptive; no population significance claim is made.

## Further experiments

A motivated next comparison fixes judgment composition to the pooled
distribution and varies which information predicts D: global only, evidence
structure only, instruction setting only, or their combination. It needs a
declared selection rule and fresh evaluation questions. Reusing the current
targets as development must be recorded explicitly.

The portable repository runner enables separate new runs on CPU, Apple MPS or
CUDA. It does not silently replace the original frozen runtime or make a new
backend run a bit-for-bit replication. Every additional model revision, backend,
precision setting or sampling policy needs its own recorded run identity.

The original exact archives stay fixed. Improvements to portable code, further
experiments and amended interpretations belong in new files and new runs.
