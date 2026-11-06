"""Bind a dual-model prospective study into one freeze manifest.

Usage:
  python3 scripts/freeze_study.py --study-dir research/experiment_10 \
      --fit fits/smollm2-v2-20261106 --fit fits/llama32-v1-YYYYMMDD \
      --question-bank question_bank.json

Writes study_freeze.json exclusively into the study dir, hashing: protocol,
reviewed question bank, evaluation design, fold map, dev pools, both fit
manifests (which bind their own inputs), both predictions files, and the eval
configs. Both models' forecasts must exist before either evaluation run is
prepared; this manifest is the artifact a later audit compares run inputs
against. Enforcement of "no eval generation before freeze" remains procedural:
the manifest exists to be checked, not to physically block prepare.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "src"))
from persona_observatory import __version__  # noqa: E402
from persona_observatory.storage import read_json, require, sha_file, write_json_exclusive  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True, type=Path)
    ap.add_argument("--fit", action="append", required=True, type=Path,
                    help="Fit attempt dir, relative to study dir (repeat per model)")
    ap.add_argument("--question-bank", required=True, type=Path,
                    help="Reviewed question bank, relative to study dir")
    ap.add_argument("--design", default=Path("design_eval.jsonl"), type=Path)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--dev-pool", action="append", type=Path,
                    help="Extra dev pool path, relative to study dir (repeatable); default: development/dev_pool.jsonl if present")
    ap.add_argument("--fold-map", type=Path, default=Path("development/fold_map.json"))
    args = ap.parse_args()

    sd = args.study_dir.resolve()
    items = {}
    def add(name, rel):
        p = sd / rel if not rel.is_absolute() else rel
        if not p.is_file():
            raise SystemExit("Missing freeze input: " + str(p))
        items[name] = {"path": str(p.resolve().relative_to(REPO)) if p.is_relative_to(REPO) else str(p),
                       "sha256": sha_file(p)}

    add("protocol", Path("PROTOCOL.md"))
    add("question_bank", args.question_bank)
    add("design", args.design)
    pools = args.dev_pool or ([Path("development/dev_pool.jsonl")] if (sd/"development/dev_pool.jsonl").is_file() else [])
    require(pools, "No development pool given or found")
    for i, rel in enumerate(pools):
        add(f"dev_pool_{i}", rel)
    fm = sd / args.fold_map
    if fm.is_file():
        add("fold_map", args.fold_map)
    fits = []
    for i, rel in enumerate(args.fit):
        add(f"fit_{i}_manifest", rel / "manifest.json")
        m = read_json(sd / rel / "manifest.json")
        add(f"fit_{i}_predictions", rel / "predictions.jsonl")
        fits.append({"dir": str(rel), "outputs_sha256": m["outputs_sha256"],
                     "inputs_sha256": m["inputs_sha256"]})
    for cfg in args.configs:
        add("config_" + Path(cfg).stem, Path(cfg))

    manifest = {
        "schema_version": "pobs-study-freeze-v1",
        "package_version": __version__,
        "items": items,
        "fits": fits,
        "note": ("Binds protocol, reviewed questions, design, development provenance and all "
                 "model forecast sets into one hash manifest. Both evaluation runs must reference "
                 "this freeze; any change invalidates the comparison. Not retroactive preregistration."),
    }
    write_json_exclusive(sd / "study_freeze.json", manifest)
    print(json.dumps({"status": "STUDY_FROZEN", "items": len(items), "fits": len(fits)}, indent=2))


if __name__ == "__main__":
    main()
