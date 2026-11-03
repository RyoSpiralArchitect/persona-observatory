# Adding an experiment

Use a new config, question bank, run directory, and research record. Preserve the
completed archives and their manifests. Before generation, write down the question,
the role of every dataset, the observer rubric, and the comparisons to be evaluated.

For a prospective prediction study, export sample IDs with `pobs design`, fit and
select predictors using development data only, and pass the complete forecasts to
`pobs prepare --forecasts`. The portable runner checks the prediction grid and
locks its bytes; it does not audit the external fitting procedure or question
novelty. Keep that code and provenance alongside the study plan.

Run the same prepared design once. Preserve failed attempts and length-capped
outputs. Export each observer's blind workspace separately. Semantic reading must
remain an actual observation task; do not replace it with word-list heuristics
while describing it as an independent semantic observer. Freeze complete labels
and notes before comparing methods.

Keep evidence separate from interpretation. Report every prespecified primary
comparison, metric disagreements, and scope limitations. A and B may be two
measurements of the same generated outputs; they do not double the subject sample
size. A candidate selected after viewing test ranks needs a new evaluation.

## Code changes

Create a branch, implement the specific change, and add tests for meaningful risks:
paired RNG behavior, immutable boundaries, measurement attribution, numerical
identities, or data leakage. Run `python -m unittest discover -s tests -v` and
`python scripts/verify_archive.py`. Run a tiny local CPU smoke when changing the
inference path. Record actual MPS/CUDA results before claiming hardware validation.

The `validation/` engineering smokes do not constitute new held-out scientific
evidence. Their purpose, environment, and retained outputs make that distinction
reviewable.

Original contributions are licensed under Apache-2.0. Do not commit credentials,
model weights, environments, or unreviewed personal data. Include a source and
license note for any third-party material added later.
