# Attribution calibration after Experiment 10

Status: **packet prepared; human judgments pending**. This is calibration using
existing public outputs. It is not fresh evaluation and will not replace the
original A/B labels, frozen rubric, predictions, or failed primary result.

The selection plan is frozen before human judgments. It includes all 26 distinct
exact triples from the 28 D-disagreement records and 26 agreed controls, split
equally by agreed D. Controls prefer less represented setting/evidence cells;
this is a diagnostic sample, not a population accuracy estimate. The 52 unique
cases are independently ordered into eight six-case batches and one four-case
batch. Source labels, sample IDs, conditions and selection groups are withheld
from the participant packet. Public source material remains searchable, so this
is instruction-level masking, not secure secrecy or study-history naivety.

The human reads each full question and completion and supplies one or more
admissible modes (`direct`, `reported`, `both`, `none`), exact evidence quotes,
and a reason. Multiple candidates remain unresolved, with a required explanation.
The focused attribution form is separate from the original eight-field rubric.
It does not create new judgment/stance/limit labels. D is derived as a set:
`direct`/`both` imply true and `reported`/`none` imply false.

The volunteer's pseudonym, declared actor type, full-text reading, prior exposure,
and assistance are recorded per batch. Human identity and semantic accuracy cannot
be verified by a schema. AI-assisted or model submissions are labeled separately;
they do not satisfy an unassisted human calibration gate. The investigator knows
the old labels and must not recommend answers while collecting judgments.

## Local preparation and collection

```bash
.venv/bin/python scripts/human_calibration.py prepare \
  --archive research/experiment_10_mps_20261103/evidence_20261103.zip \
  --plan research/attribution_calibration_20261103/selection_plan.json \
  --output .local/human-calibration/20261103-001

.venv/bin/python scripts/human_calibration.py template \
  --participant .local/human-calibration/20261103-001/participant \
  --batch batch_01

.venv/bin/python scripts/human_calibration.py submit \
  --participant .local/human-calibration/20261103-001/participant \
  --submission .local/human-calibration/submission.json \
  --records .local/human-calibration/20261103-001/submissions
```

Use a new output directory for another preparation. The participant reads only
`participant/`, never `sealed/`. The preparation binds case text, order, original
rubric, and guide with hashes. Keep the packet lock hash with every submission.
All packet instances, chat receipts and human submissions stay ignored under
`.local/` while collection is underway. The tracked plan and collection code
contain no human judgments or identifying declarations.

For chat collection, retain the exact human message before transcribing it into
the structured form. Add a separate receipt binding that message and the JSON
with hashes. Do not invent missing quotes, reasons, uncertainty decisions, or
declarations. Ask for missing fields; preserve partial or invalid attempts.
Record presentation errors and their corrections separately. The original
message stays unchanged.

Raw submission bytes are retained before validation, including failures. A valid
batch receives a separate acceptance record keyed by packet, pseudonym and batch;
a second submission cannot overwrite it. A correction requires a separately
declared amendment, preserving the first accepted record. No automatic amendment
or semantic adjudication is implemented.

## Closure and next gate

Do not reveal source labels during annotation. Compare after all 52 cases are
received and locked, or after an explicit partial-stop declaration. Report
coverage, set-valued ambiguity, D-stable versus D-ambiguous cases, and exposure
before any agreement numbers. Missing or unresolved cases stay visible. Do not
turn a partial batch or majority vote into a gold standard.

Then review ambiguous constructions and author a separately versioned rubric.
If attribution cannot be resolved defensibly, retain set-valued/abstaining
measurement or stop the point-label study. A revised rubric requires its own
calibration and lock before fresh observations; agreement with old labels alone
is not a success criterion. Only then select new predictors on explicitly
designated development data and freeze a fresh evaluation.

See [the staged next-study plan](../../docs/next_study_after_exp10.md).
