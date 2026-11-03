# Experiment 9 — globally centered shrinkage of observable answer profiles

This is a local, prospective, authored-benchmark pilot. A source/forecast lock is
an audit trail, not public preregistration or an independent timestamp certificate.
The question is whether a compact output-derived profile forecasts new questions
better than a pooled baseline, and whether allowing separate shrinkage for
direct-answer eligibility and judgment composition improves that forecast.

## Continuity and data

Experiment 8 was already completed. Its section 8 proposed G-centered shrinkage
and the separation of answer eligibility from content. This new study implements
that proposal. We intentionally use some former v8 test data for development;
none of them is represented as an unseen v9 test.

The exact v8 source archive SHA-256 is
0538ed94a04a2fa5f51195c56b66a4eabeb2ae56c0aa1cddb12787d0db8b77c8.
Use 192 v8 support outputs (all 16 questions × six settings × first two support
seeds) and all 144 former v8 target outputs. Exclude fit and excess anchor repeats.
This yields 336 balanced development observations, 28 questions, 14 records per
setting × stratum cell. All algorithms have access to the same observations.
Original observer A and B labels are retained, unchanged and separately.

The new target is 24 authored questions, six in each of limited, confounded,
supportive, contradictory evidence strata. Each has all six familiar settings
(neutral/cautious/confident × two existing instruction wordings) and two fresh
question-specific random streams: 288 target outputs. Each stream is paired across
settings. All 24 topics differ from all 36 v8 topics; evidence structures remain
deliberately similar. This tests new-question forecasting with familiar wording
and authored evidence types, not arbitrary semantic transfer or unseen wording.

## Subject and observations

Use HuggingFaceTB/SmolLM2-360M-Instruct, revision
a10cc1512eabd3dde888204e902eca88bddb4951, with the six verified original files.
The original runtime.py and rubric.md are copied byte-identically. CPU float32,
eager attention, four Torch threads and one interop thread, batch eight,
temperature 0.8, top_p 0.95, at most 96 generated tokens, fresh chat per output.
Prompt template and six evaluation policies are unchanged. No development
completions are inserted into target prompts. The predictive observer uses
visible responses only; no hidden states, embeddings or logits are its features.

Five categories describe observable text: support, reject, withhold, mixed,
no_direct. D means a usable direct judgment by the requested answerer under the
frozen role-relative rubric; it does not mean subjective agency or authentic
personality. Strict E additionally requires that speaker to state an explicit
evidence/knowledge limit. D and E are not interchangeable.

Two fresh target observer agents use the unchanged rubric in separate contexts,
with shuffled, blinded full outputs. They are replicate uses of the same
foundation model and rubric, not independent human raters or independent models.
They see neither conditions, prior results, predictions nor each other's labels.
They read each unique full question/completion triple, lock all labels, and no
post-outcome adjudication is performed. Keep all outputs, including no_direct
and length stops; report disagreement directly.

## Parameterization and fairness

For category k, global g_k=(N_k+0.5)/(N+2.5). Let g_D=1-g_no_direct.
For a local cell with n total, n_D direct, and n_k direct-category counts:

    r(rho_D) = (n_D + rho_D*n*g_D) / (n + rho_D*n)
    h_k(rho_C) = (n_k + rho_C*n*g_k) / (n_D + rho_C*n*g_D)
    p_no_direct = 1-r; p_k = r*h_k

At infinity use r=g_D and h_k=g_k/g_D explicitly. Empty cells fall back to G.
Use the grid 0.25, 1, 4, infinity. For the flat model,

    p_k(F,rho) = (n_k + rho*n*g_k) / (n + rho*n).

H at rho_D=rho_C equals F exactly. The conditional prior mass is rho_C*n*g_D,
not rho_C*n. This identity is important: factorization alone cannot improve
a probability distribution. Only allowing different strengths changes it.
Global probabilities are estimated from the same training pool, so these are
globally centered empirical shrinkage forecasts, not a posterior from an
independently supplied prior. Positive smoothing avoids zero-probability NLL.

Seven algorithms:

| Algorithm | Pooling and allowed adaptation |
|---|---|
| G | One global five-category vector |
| F | Setting × evidence stratum, one shared rho |
| H_gate | Setting × stratum, rho_C=infinity |
| H_content | Setting × stratum, rho_D=infinity |
| H | Setting × stratum, both rho coordinates free |
| H_evidence | Evidence-stratum-only pooling, both coordinates free |
| Hswap | H vectors exchanged between two wordings within family and stratum |

H_evidence uses n=84 finally and n=72 within CV, compared with H n=14/12.
H versus H_evidence therefore compares information and estimation variance
together; it is not a pure causal isolation of instruction effects. Hswap is a
fixed finished-profile permutation, not a re-fit and not a hidden-state intervention.

## Selection before target generation

Seven folds each hold out four whole questions, one per stratum, and every
setting and seed for those questions. Folds 0..3 are old support topic_index;
folds 4..6 are former target topic_index+4. Each validation fold has 48 outputs,
training 288. G and every local count are recomputed from training only.

Choose each restricted algorithm in its own candidate set using mean held-out
five-category Brier over all records and both observers equally. Counts remain
observer-specific; one hyperparameter choice per algorithm is shared by A/B.
Use a numerical tie tolerance of 1e-12. F prefers stronger rho. Factorized
algorithms prefer fewer finite coordinates, then stronger gate rho, then stronger
composition rho. This favors global prediction under exact ties. Keep global,
diagonal or coincident selected forecasts; no forcing diversity or expanding the
grid after examining development or target outcomes.

H's candidate set contains F, H_gate, H_content and G, so its minimum CV Brier
cannot exceed theirs. This mathematical nesting is not evidence of new-target
success. All selected development scores remain development quantities.

## Outcomes and fixed comparisons

Primary loss is five-category Brier, sum of five squared errors (range 0..2),
lower is better. Also report natural-log categorical NLL. D receives binary
Brier (range 0..1) and NLL. Always score all targets, not only direct responses.

Co-primary contrasts are H-G and H-F. Descriptive support for both co-primary
claims requires negative A/B-mean Brier differences for both; also report each
observer's sign, equality, and whether all four branch/comparison signs agree.
This rule is a descriptive pilot criterion, not a significance claim.
Required controls: F-G, H-H_gate, H-H_content, H-H_evidence, H-Hswap,
H_evidence-G. These eight contrasts are reported for both categorical and D
losses, both Brier and NLL, separately for A/B and their mean. The sign convention
is first minus second, so negative favors the first.

NLL decomposes exactly on each output:

    five_category_NLL = D_NLL + D * conditional_composition_NLL.

Report the latter contribution averaged over all targets, plus the conditional
mean among D=1 with its denominator. When averaging observers, average complete
contributions, not unweighted conditional stage means.

Retain strict E as an auxiliary distinct event. From development labels separately
for each observer, ell=(M_E+0.5)/(N_D+1). Use p_E(G)=p_D(G)*ell and
p_E(H)=p_D(H)*ell. Only G/H receive this auxiliary Brier/NLL comparison; there is
no additional selection. E0 predicts zero, Brier only, to show rare-event scale.
This tests gate adaptation with pooled explicit-limit propensity, not learned
condition-specific limit expression. Verify E implies D and p_E<=p_D.

Report 24 question, six setting, four stratum, and three family block summaries,
including sign counts with tolerance 1e-12. No iid p-values or population
confidence intervals are claimed: the authored bank is neither a random sample
of language tasks nor a population of people. A/B observations are correlated
measurement branches, not independent repetitions of the subject experiment.

## Ordering, failure handling and audit

1. Derive immutable balanced development inputs and author new question bank.
2. Write all model, observer, forecast, scoring and independent audit code.
3. Run invented-count mathematical tests and tokenizer-only preflight. No new
   subject forward pass or target generation is permitted yet.
4. Independent design review; freeze all listed sources, inputs and target plan.
5. Perform declared CV and write final profiles, complete target forecasts,
   identity certificate and FORECAST_NOTE.md; lock before any target attempt.
6. Independently recompute all forecasts and selection from raw old labels.
7. Generate once, preserving attempt marker, source snapshot, all outputs and
   failure information. No silent replacement/resume after a failed attempt.
8. Two fresh blinded observers label and lock; then score once.
9. Independent final audit re-derives scores, paired contrasts and chronology.

Any necessary post-lock repair must be transparently recorded, preserve original
files and locks, and cannot alter targets or select a more favorable method.
If a target attempt fails, preserve it and report the limitation before any
replacement design. Do not silently turn a failed prospective run into a new one.

## Interpretation boundary

A successful profile is a compact, portable description of observable response
regularities within this model/task/prompt distribution. It does not identify a
unique internal state, recover a hidden reasoning trajectory, establish stable
personhood, or prove the semantic accuracy of generated self-reports. No internal
state comparison is performed in this stage. Failure against G is informative:
it bounds the predictive value of these condition-specific distinctions under
this data budget and selection procedure.
