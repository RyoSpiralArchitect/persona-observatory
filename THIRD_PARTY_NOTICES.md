# External models, dependencies, and references

The Apache-2.0 license at the repository root covers original project material.
External resources retain their own licenses. This repository does not bundle
pretrained weights or installed third-party libraries.

| Resource | How it is used | Upstream source |
|---|---|---|
| SmolLM2-360M-Instruct | Optional local subject model; six file hashes and a revision are recorded | [Pinned model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md) |
| Llama-3.2-3B-Instruct | Optional local subject model; eight file hashes and a revision are recorded. Gated upstream: access requires accepting the Meta Llama 3.2 Community License under the downloader's own account; weights are not redistributed by this repository | [Pinned model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/0cb88a4f764b7a12671c53f0838cd831a0843b95/README.md) |
| PyTorch | Optional inference on CPU/MPS/CUDA | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | Optional local tokenizer and causal model loader | [Transformers license](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE) |
| MLX / MLX-LM | Optional native Apple GPU forward adapter; not bundled | [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm) |
| Hugging Face Hub / safetensors | Optional inference dependencies | [Hub](https://github.com/huggingface/huggingface_hub), [safetensors](https://github.com/huggingface/safetensors) |
| Apache License 2.0 | License text for original project material | [Canonical text](https://www.apache.org/licenses/LICENSE-2.0.txt) |

The historical archive includes generated text, authored prompts, code, metadata,
and research reports from this project. Referenced research papers are cited by
link; their complete text is not redistributed. Model outputs and observation
notes are research records and are not statements endorsed by model providers.

Changing the subject model requires recording its exact source and following
that model's applicable license and access conditions. The project license does
not grant rights to external model weights or third-party material.
