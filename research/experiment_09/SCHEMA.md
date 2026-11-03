# Experiment 9 artifact contracts

All paths are relative to this directory. UTF-8 JSON/JSONL; probabilities and
scores are stored as full-precision JSON numbers. No target output may exist
before both the design lock and forecast lock, and the independent forecast
audit. Source v8 files remain immutable.

## Configuration and development inputs

`study_config.json` holds constants, candidate grid, selection rules, comparisons,
and the scope of interpretation. `settings.json` contains the six unchanged v8
evaluation instruction settings. `question_bank.json` has 24 new authored
questions with seven fields: phase, question_id, topic, topic_index,
evidence_stratum, evidence, claim.

`provenance/source_v8.zip` is the exact immutable source archive. The executable
`prepare_development.py` verifies its SHA-256, its complete internal manifest,
and original annotation locks before deriving the following inputs:

- `development/records.jsonl`: 336 records with `sample_id`, `source_phase`
  (`support` or `target`), `development_fold` (0..6), and `source_record`
  (the complete original unmodified generation object).
- `development/labels_observer_a.jsonl` and `_b.jsonl`: the 336 corresponding
  original eight-field labels, unmodified as JSON objects.
- `development/source_question_bank.json`: all 36 old questions.
- `development_manifest.json`: selection specification, provenance, counts,
  question/fold map, hashes, source annotation lock identities.

Select support seeds 2026110383 and 2026110384 only (192 records), plus all
old v8 targets (144). These old targets are development data in v9. Exclude all
v8 fit outputs and extra anchor repetitions. Support topic_index 0..3 gives
folds 0..3; old-target topic_index 0..2 gives folds 4..6. Each fold contains all
six settings and both seeds for four questions, one per evidence stratum.

## Target plan and generation

`experiment_config.py` exposes ROOT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P for the
byte-identical copied v8 runtime.py, and implements the frozen design/seed scheme.
Use context_id = `target_{question_id}_{setting_id}_bounded_conclusion`.
Seed for replicate 0 or 1 is the unsigned integer represented by the first eight
bytes of SHA-256 of `persona-shrinkage-v9|{question_id}|{replicate_id}`, big endian,
masked with `(1 << 63) - 1`. There are 48 distinct streams, reused across the
six settings for the same question/replicate. sample_id is the first 12 hex
characters of SHA-256 of `persona-shrinkage-v9|{context_id}|{seed}`.

`data/target_plan.jsonl` contains 288 rows, setting-major, then bank order,
then replicate 0,1. Each row has sample_id, context_id, phase, question_id,
question_name, question, evidence_stratum, format, setting_id, family, frame,
requested_answerer, paraphrase_id, policy_text, topic_index, messages, prompt,
prompt_ids, prompt_tokens, replicate_id, seed. The prompt and tokenization use
the exact v8 system template, wrapper, chat template, and bounded question.
No obsolete anchor or support-role fields are needed.

`data/target_generations.jsonl` preserves every plan field and adds completion,
generated_ids, generated_tokens, stop_reason. All 288 outputs remain eligible,
including length stops. One immutable generation attempt: write a start marker
before model loading, preserve partial output and failure metadata on error,
and never silently retry, resume, replace, or delete the attempted run.

`run_experiment.py` commands:

- `prepare`: tokenizer-only target plan, model/environment verification, preflight.
- `lock`: exclusive design lock, checking complete source/input list and no
  target attempts/forecasts/results; requires independent design-prelock audit.
- `generate`: verifies all design/forecast hashes and a passing independent
  forecast audit tied to both locks; creates attempt marker, source snapshot,
  generates once, writes final metadata, and exports blinded observer files.

The design lock includes PROTOCOL.md, SCHEMA.md, study_config.json, settings.json,
question_bank.json, QUESTION_BANK_REVIEW.md, DEVELOPMENT_REVIEW.md,
prepare_development.py, development_manifest.json, all four development inputs,
provenance/source_v8.zip, experiment_config.py, runtime.py, run_experiment.py,
rubric.md, validate_observation.py, predict.py, audit_shrinkage.py,
prediction_math_audit.md, AUDIT_PLAN.md, model_manifest.json,
expected_model_manifest.json, environment.json, preflight.json,
data/target_plan.jsonl, metadata/target_plan.json, and audits/design_prelock.json.
Design-prelock audit must not contain the future design lock's hash.

## Prediction and selection

Algorithms in fixed order: G, F, H_gate, H_content, H, H_evidence, Hswap.
Category order: support, reject, withhold, mixed, no_direct.
Use `observer_a` and `observer_b` branch-specific original counts, but the same
hyperparameters selected by the mean of both observers' question-blocked CV
five-category Brier scores.

`predict.py` commands: `synthetic`, `freeze`, `score`. Synthetic calculations
must not select parameters on actual development data or generate model output.
Freeze refuses any existing forecast lock/artifacts or target attempt/output.
It verifies the design lock before fitting and writes:

- predictions/cv_scores.csv: every fold × observer × pooling structure × 16
  rho pairs, sample count, mean categorical Brier and mean NLL.
- predictions/cv_candidate_summary.csv: per-pooling/per-pair branch and mean
  Brier, and eligible/selected algorithm annotations.
- predictions/selection.json: choices for every algorithm, fixed grid,
  objective, tie rule, exact forecast coincidences/global collapse.
- predictions/counts.json: global, setting-by-stratum and evidence-only counts
  for final fit and each fold, with sample IDs and strict-E counts.
- predictions/profiles.json: 48 observer × setting × stratum objects, each with
  seven five-category vectors, directness and optional strict-E forecasts.
- predictions/forecasts.jsonl: 4032 rows (288 × 2 × 7), each with sample_id,
  observer, predictor, question_id, setting_id, evidence_stratum,
  category_probabilities (five numbers in frozen order), gate_probability,
  strict_event_probability (number for G/H only, null otherwise).
- predictions/identity_certificate.json: diagonal H=F and global-limit
  identities, normalization, strict-E <= D, zero-direct-count edge cases,
  and candidate-nesting CV inequalities (mathematical, not target success).
- FORECAST_NOTE.md: selected parameters and fixed worked example
  eval_neutral_instruction_p0 across all four strata, including all component
  probabilities needed to read the forecast. No outcome-selected example.
- predictions/forecast_lock.json: hashes of all above outputs, all design
  inputs and design_lock.json, time, row counts, absence of target attempts.

The exact internal JSON layout of count/profile/selection artifacts is documented
by prediction_math_audit.md and consumed independently by the audit. A reader
must be able to reconstruct every forecast from the supplied raw inputs.

## Observation and scoring

Two fresh observer agents each see only their own rubric.md, empty
discovery_blind.jsonl, and independently shuffled validation_blind.jsonl in
`observers/target/observer_a` or `observer_b`. Blind fields are exactly
sample_id, requested_answerer, question, completion, stop_reason. They read every
full unique triple semantically, write eight-field labels.jsonl and observations.md,
and use validate_observation.py to validate and lock. Original v8 rubric text
is byte-identical; validator changes counts/version/path only, not semantics.
Target labels are not adjudicated or altered after lock.

Derived category is judgment when mode is direct/both and judgment != none;
otherwise no_direct. D is category != no_direct. Strict E is D and the presence
of answerer_direct in limit_owners. E implies D by definition.

Score all seven algorithms for categorical Brier=sum over five squared errors,
categorical NLL=-log p_y, gate Brier=(p_D-D)^2, gate NLL. Strict-E Brier/NLL are
scored only for G/H. E0 predicts zero and is Brier-only. Decompose categorical
NLL per sample into gate NLL and D times conditional NLL. No clipping is needed.

Results must preserve per-sample scores and contrasts, algorithm/observer
summaries, A/B means, per-question/setting/stratum/family paired differences,
NLL decomposition, category and event counts, observer agreement and disagreements,
and a chronology/integrity summary. Store canonical machine-readable JSON/JSONL
and convenient CSV tables. See PROTOCOL.md for the fixed comparisons and scope.

`audit_shrinkage.py` has stages design, forecasts, final; it must independently
rebuild selection, probabilities, and scores from raw inputs without importing
predict.py, run_experiment.py, or their algorithms. Each report records input
hashes, assertion count, failures and status ALL_MATCH or FAIL. The forecast
audit records design_lock_sha256 and forecast_lock_sha256. Final audit includes
all prior checks, frozen code integrity, annotation locks, generation chronology,
every plan/output field, and all scalar scores/paired differences.
