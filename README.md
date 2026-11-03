# Persona Observatory

**Output-based persona research, with preserved evidence and local experiments on CPU, Apple MPS, and NVIDIA CUDA.**

[日本語の詳しい説明](README_JA.md) · [Apache-2.0](LICENSE) ·
[Research history](docs/research_history.md) · [Next experiment proposal (日本語)](docs/next_experiment.md)

[Inference adapters: PyTorch and native MLX](docs/adapters.md)

Persona Observatory asks whether profiles built from observable model responses
help forecast responses to new questions. Semantic observers read complete
outputs; separate code fits or evaluates probability forecasts. Hidden states
are not predictor features. A useful forecast would not establish a persistent
personality or recover the model's internal reasoning.

## Current evidence

Experiment 9 used 336 development outputs and 288 fresh evaluation outputs from
24 authored questions, six familiar instruction settings, and two samples per
question and setting. Its co-primary criterion **failed**: H did not beat both
the pooled baseline G and the common-shrinkage predictor F on five-category
Brier loss.

| Frozen predictor | Five-category Brier | Five-category NLL |
|---|---:|---:|
| G: pooled distribution | 0.658847 | 1.194051 |
| F: common shrinkage | 0.655841 | 1.208360 |
| H: separately selected gate/composition shrinkage | 0.657402 | 1.213900 |
| H_gate: conditional gate, pooled composition | **0.651359** | **1.181487** |

Lower is better. Values average two observer branches over the **same** 288
generations. H_gate ranked best among seven frozen candidates; this is a
descriptive finding, not a successful prespecified primary comparison. The
observers were separate contexts of the same foundation model, not independent
human raters or twice as many subject generations.

- [Experiment 9 report (日本語)](research/experiment_09/REPORT_JA.md)
- [Original protocol](research/experiment_09/PROTOCOL.md)
- [Preserved archive, embedded Experiment 8, and integrity checks](archives/README.md)
- [Engineering validation and its limits](validation/README.md)

The original ZIP and its browsable copies remain byte-identical. Archive
verification establishes byte and membership integrity. The original arithmetic
audit is a separate preserved record; neither check validates semantic labels.

## Start with the core checks

Python 3.11 or later is required; Python 3.12 is recommended. No model, Torch,
GPU, or network inference service is needed for these checks.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_archive.py
pobs --help
```

Tensor tests skip when Torch is absent. GitHub Actions runs these core checks
on Ubuntu and macOS; that does not validate MPS or CUDA inference.

## Run local inference

Follow the device-specific setup before installing the inference extra:
[CPU walkthrough (日本語)](README_JA.md#最初の動作確認),
[Mac / MPS](docs/mac.md), or [Linux / CUDA](docs/furnace.md).
Then use a separate run directory:

```bash
pobs doctor --device cpu --dtype float32
pobs fetch-model --config configs/smoke_cpu.json --destination models/smollm2-360m-instruct
pobs prepare --config configs/smoke_cpu.json --run-dir runs/cpu-smoke-001 --model-path models/smollm2-360m-instruct
pobs generate --run-dir runs/cpu-smoke-001
pobs verify-run --run-dir runs/cpu-smoke-001
pobs export-observer --run-dir runs/cpu-smoke-001 --observer observer_a
```

The smoke reuses two known questions, two settings, and one sample, capped at 16
new tokens. `configs/replay_v9_*.json` reuses known Experiment 9 questions. These
are engineering checks, not fresh scientific holdouts. Models, environments,
and local runs are excluded from Git.

The default adapter uses Transformers with `trust_remote_code=False`, CPU
float32 sampling, and a separate CPU random stream for each question/replicate,
paired across settings. The selected device performs model forward passes.
Equal seeds do not guarantee equal text across devices, numerical stacks, or the
archived runtime. The optional native MLX adapter uses the same model files and
tokenizer with its separately recorded NumPy sampler; see [adapter setup](docs/adapters.md).

## A prospective study

1. Declare dataset roles, predictors, observer rubric, and primary comparisons.
2. Use `pobs design` to export sample IDs. Fit/select predictors using development
   data only, then save every target forecast before generation.
3. Run `pobs prepare --forecasts predictions.jsonl` with `purpose=prospective`.
   This freezes inputs, model identity, prompt tokens, seeds, source, and forecasts.
4. Generate once. Failed and completed attempts cannot be silently resumed or
   overwritten; retain them and use a separately documented new run if needed.
5. Export blinded observer workspaces. Supply semantic labels by actual reading,
   validate and lock each observer separately, then score the locked forecasts.

The CLI does **not** fit predictors or establish that evaluation questions are
novel. External fitting, development provenance, and semantic observation need
their own review. A/B remain separate measurements of the same outputs.

See [measurement](docs/measurement.md), [configuration](docs/configuration.md),
and [contribution boundaries](CONTRIBUTING.md). The
[next experiment proposal](docs/next_experiment.md) compares global,
evidence-only, setting-only, and joint forecasts of direct judgment D, holding
judgment composition pooled. That initial proposal remains a historical design
note. Its [96-question MPS implementation](research/experiment_10_mps_20261103/README.md)
has completed 1,152 generations with forecasts and comparisons fixed beforehand.
Both agent observations are locked and scores are complete: the primary
Joint−Evidence D Brier difference is **+0.000749**, and all four advance gates
failed. Setting ranked best as a secondary finding. Initial environment context
exposed both annotators to historical study information, so they were not
study-history-naive; current condition metadata and forecasts were withheld.
These are two measurements of the same outputs, not human raters or twice as
many subject generations. See the [report and evidence](research/experiment_10_mps_20261103/REPORT_JA.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/persona_observatory/` | Portable runner, observation validation, scoring |
| `configs/`, `examples/`, `manifests/` | Engineering/replay inputs and pinned model identity |
| `archives/`, `research/experiment_09/` | Immutable evidence and browsable copies |
| `docs/` | Setup, measurement, research history, next study |
| `validation/`, `tests/` | Engineering receipts and core checks |
| `models/`, `runs/`, `.local/` | Ignored local working data |

Original project code and authored research materials are licensed under
Apache-2.0. External model weights and libraries retain their own licenses and
are not bundled. See [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md).
