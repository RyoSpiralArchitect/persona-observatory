# Research and implementation boundaries

- Preserve `archives/experiment_09_20261103.zip` and all manifest-checked copies.
  New experiments and corrections belong in separately named records. Do not
  edit historical locks, predictions, labels, or outcomes to make a check pass.
- The v9 co-primary criterion failed. H_gate's best rank is descriptive. Keep
  that distinction in summaries and future plans.
- New portable runs use `src/persona_observatory`; archived runtime files stay
  byte-identical. This portable CPU/MPS/CUDA implementation is not a claim of
  cross-device token identity or reproduction of the original sampled outputs.
- Freeze development roles, predictors, rubric, and primary comparisons before
  generating new evaluation outputs. Never relabel old test data as fresh.
- Observer exports are blinded, semantic labels are manually supplied, and A/B
  remain separate measurements. Do not use keyword annotation as a substitute.
- A failed or completed generation attempt is not overwritten or silently
  resumed. Use another run directory and preserve the previous attempt.
- Validate relevant changes with the stdlib suite and archive checker. No model
  download is necessary for core CI. Use a tiny engineering smoke for inference
  changes; do not launch a full scientific study merely to test packaging.
- Keep weights, environments, local runs, and credentials out of git. Apply
  Apache-2.0 to original project work; keep external licenses distinct.
