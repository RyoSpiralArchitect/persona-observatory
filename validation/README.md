# Implementation verification — 2026-11-03

These are engineering checks of the new portable implementation. They are not
new held-out evidence for a persona hypothesis and do not replace Experiment 9.

| Check | Observed result |
|---|---|
| Editable installation of the lightweight package | Passed |
| Core suite without Torch | 62 tests discovered; 58 passed and 4 tensor tests skipped |
| Suite with Torch 2.15.0+cpu / Transformers 4.57.6 | All 62 tests passed |
| Independent run-boundary review | Seven originally failing cases fixed and passed; tests retained |
| Archive verifier | Original v9 archive, embedded v8 archive and 19 browsable copies matched |
| Backend real-model smoke | Two four-token outputs; CPU float32, batch2, eager + KV cache |
| CLI real-model smoke | Four sixteen-token outputs; prepare, one generation, verify and observer export |
| Synthetic prediction/observation/scoring CLI test | Hand-calculated uniform probability losses matched; second scoring refused |
| Wheel build and packaged rubric/license inspection | Passed |
| Apple MPS / furnace CUDA generation | Not executed on those devices |
| GitHub Actions | Workflow supplied; no remote run claimed in this record |

The unit suite covers complete forecast grids, numeric score identities, explicit
device errors, per-sample CPU random streams, EOS versus length caps, unchanged
input/model/source boundaries, no retry, blind exports, exact observation schema,
duplicate consistency, and immutable locks. Semantic observation is not replaced
by string heuristics.

`backend_smoke_20261103/` contains the direct-backend check and its retained outputs.
`cli_smoke_20261103/` is a complete prepared/generated run plus an unfilled observer
workspace. Its `labels.template.jsonl` is a template; no semantic labels were
assigned to those smoke outputs. `cli_smoke_commands_20261103.jsonl` records command
results. All examples reuse known questions and use separate engineering seeds.

The recorded absolute model paths identify where these checks ran. On another
machine, use the documented CLI with that machine's model directory and a new run
name; do not rerun into these completed evidence directories.

`verify-run` checks generation integrity. Later score files are separately bound
by `score_lock.json`; this version does not supply a separate full re-audit command
for completed score artifacts. The historical v9 arithmetic audit remains in its
unchanged archive.
