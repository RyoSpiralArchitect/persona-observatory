# Experiment 9 — completed, preserved study

Apart from this navigation README, these files are byte-identical excerpts from the completed study. They make the
main findings browsable without expanding the full archive. This directory is
historical evidence, not an editable template for a new run.

- [REPORT_JA.md](REPORT_JA.md): full Japanese findings and limitations.
- [METHODS_JA.md](METHODS_JA.md): observation targets, shrinkage equations and design.
- [CONCEPT_NOTES_JA.md](CONCEPT_NOTES_JA.md): relationship to the original concept-coding question.
- [PROTOCOL.md](PROTOCOL.md): specification frozen before new subject outputs.
- [SCHEMA.md](SCHEMA.md) and [rubric.md](rubric.md): data and observation definitions.
- [Interactive report](persona_shrinkage_report_20261103.html): download and open
  locally to explore all 288 new completions, both observer branches and profiles.
- [Result figure](figures/persona_shrinkage_summary_20261103.png): four-panel summary.
- [Score summary](results/score_summary.csv), [paired comparisons](results/paired_summary.csv)
  and [NLL decomposition](results/nll_decomposition.csv): compact exact aggregates.
- [Primary criterion](results/primary_criterion.json) and
  [observer agreement](results/agreement_summary.json): the recorded outcome and measurement agreement.
- [Final arithmetic audit](audits/final.json): the original independent recomputation record.

The full raw data, all predictions, semantic labels, source snapshots, code and
the preceding Experiment 8 archive are in
[`../../archives/experiment_09_20261103.zip`](../../archives/experiment_09_20261103.zip).
Run `python scripts/verify_archive.py` from the repository root to check all
copies against that archive.

## Result to retain

H improved five-category Brier over G by 0.001445, but was worse than F by
0.001561 (A/B means, with both branch directions agreeing). The predefined
co-primary criterion was not met. H_gate ranked best among the seven fixed
candidates in categorical Brier and NLL; its rank is a descriptive result,
not a replacement for the failed H co-primary criterion.

The same foundation model and rubric were used in fresh observer contexts A/B.
The new subject output count is 288, not 576 independent generations. Old
Experiment 8 targets were explicitly moved into development; the evaluation
used 24 new questions. Hidden states, persistent persona and actual reasoning
trajectories were not measured in this stage.
