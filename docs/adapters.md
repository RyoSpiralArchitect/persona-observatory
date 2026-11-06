# Explicit inference adapters

`runtime.engine` selects `transformers` (the backward-compatible default) or
`mlx`. The study, prompt, model manifest, run locks, observer exports and score
formats are shared. The inference implementation and sampler are recorded in
each run. Switching engines or numerical environments requires a new run.

| Engine | Devices | Forward pass | Sampling | Batch size |
|---|---|---|---|---|
| `transformers` | `cpu`, `mps`, `cuda`, `cuda:N`, `auto` | PyTorch, eager attention | CPU float32, Torch multinomial, per-sample RNG | Explicit positive integer |
| `mlx` | `metal`, `cpu`, `auto` | Native MLX-LM | CPU NumPy float32 nucleus, float64 probability normalization/draw, per-sample PCG64 | 1 |

MLX Metal and PyTorch MPS are different engines using the Apple GPU. Equal
seeds do not imply the same random draws or generated tokens across engines.
The MLX adapter currently accepts unquantized `model_type=llama` weights,
including the project's pinned SmolLM2-360M-Instruct. It reads the original
safetensors directly, with no weight conversion or quantization. Custom model
Python files, auto-map code, nested architectures and quantized configurations
are rejected before MLX-LM loads the model. Unsupported devices never silently
fall back. Explicit MPS runs reject `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Separate environments

The existing inference extra pins Transformers 4.57.6. The MLX extra pins
MLX/MLX-LM 0.32.2 and Transformers 5.13.0. Install them in **separate** environments;
combining the two extras produces an intentional dependency conflict.
Core installation and tests still need neither engine.

```bash
python3.12 -m venv .local/venvs/mps
.local/venvs/mps/bin/python -m pip install torch==2.15.0 -e '.[inference]'
.local/venvs/mps/bin/pobs doctor --engine transformers --device mps --dtype float32

python3.12 -m venv .local/venvs/mlx
.local/venvs/mlx/bin/python -m pip install -e '.[mlx]'
.local/venvs/mlx/bin/pobs doctor --engine mlx --device metal --dtype float32
```

Fetch the six original model files once, into a dedicated ignored directory:

```bash
.local/venvs/mps/bin/pobs fetch-model --config configs/smoke_mps.json --destination models/smollm2-360m-instruct
```

Use `configs/smoke_mlx.json` with the MLX environment, or `configs/smoke_mps.json`
with the MPS environment, following the existing `prepare`, `generate`, and
`verify-run` workflow. Always choose a new run directory. The smoke uses only
known questions and caps each of four outputs at 16 tokens.

The MLX `cpu_threads` setting must remain 4, the historical schema default; the
adapter does not apply it and records `cpu_threads=null`. MLX currently requires
`batch_size=1`. These constraints are validated rather than silently ignored.
Transformers 4 and 5 are both explicitly asked for a list of chat-template token
IDs, then checked against tokenization of the rendered prompt.

Backend preflight now checks the selected engine, sampler, numerical-library
versions, and MPS fallback flag before each generation attempt. MLX preflight
imports MLX-LM to detect library incompatibilities before loading model weights.
Archived runs remain readable with the default Transformers engine; they are
never rewritten or resumed with this adapter.

Upstream interfaces: [MLX-LM loader](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/utils.py),
[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html).
