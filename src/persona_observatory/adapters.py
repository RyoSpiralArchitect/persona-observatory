"""Explicit inference adapters; imports remain lazy for dependency-free core CI."""
from __future__ import annotations

from .backend import validate_runtime


def inspect_backend(device="auto", dtype="auto", engine="transformers"):
    validate_runtime({"engine": engine, "device": device, "dtype": dtype})
    if engine == "mlx":
        from .mlx_backend import inspect_backend as inspect
    else:
        from .backend import inspect_backend as inspect
    return inspect(device, dtype)


def generate_records(plan, model_path, runtime, on_record):
    runtime = validate_runtime(runtime)
    if runtime["engine"] == "mlx":
        from .mlx_backend import generate_records as generate
    else:
        from .backend import generate_records as generate
    return generate(plan, model_path, runtime, on_record)
