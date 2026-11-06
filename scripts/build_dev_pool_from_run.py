"""Build the experiment-11 development pool from the Llama development run.

Joins a completed development-pool generation run with the two locked
observer label sets. Fold assignment reuses the experiment-10 question-blocked
fold map (the question set is identical), and the question bank is copied
verbatim. Inputs are hash-bound into provenance.json; nothing here mutates
the run directory or the locked observer artifacts.

Usage:
  python3 scripts/build_llama_dev_pool.py --generations .local/llama32-devpool-001/generations.jsonl \
      --labels-a .local/observation/observer_a/labels.jsonl \
      --labels-b .local/observation/observer_b/labels.jsonl \
      [--output-dir research/experiment_11/development]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from persona_observatory.observations import derived_outcome  # noqa: E402
from persona_observatory.storage import require  # noqa: E402


def sha_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, type=Path)
    ap.add_argument("--source-role", required=True)
    ap.add_argument("--labels-a", required=True, type=Path)
    ap.add_argument("--labels-b", required=True, type=Path)
    ap.add_argument("--fold-map", type=Path,
                    default=REPO / "research" / "experiment_10" / "development" / "fold_map.json")
    ap.add_argument("--question-bank", type=Path,
                    default=REPO / "research" / "experiment_10" / "development" / "dev_question_bank.json")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "research" / "experiment_11" / "development")
    args = ap.parse_args()

    gens = rows(args.generations)
    labels = {
        "a": {r["sample_id"]: r for r in rows(args.labels_a)},
        "b": {r["sample_id"]: r for r in rows(args.labels_b)},
    }
    fold_map = json.loads(args.fold_map.read_text())
    fold_of = {q: int(f) for f, qs in fold_map.items() for q in qs}

    require(len(gens) == len({g["sample_id"] for g in gens}), "Duplicate generation sample_ids")
    for obs, lab in labels.items():
        if len(lab) != len(gens):
            raise SystemExit(f"observer {obs}: expected {len(gens)} labels, got {len(lab)}")
        missing = {g["sample_id"] for g in gens} - set(lab)
        if missing:
            raise SystemExit(f"observer {obs}: unlabeled samples {sorted(missing)[:3]}")

    pool = []
    for g in gens:
        for obs in ("a", "b"):
            out = derived_outcome(labels[obs][g["sample_id"]])
            pool.append({
                "sample_id": g["sample_id"],
                "source_role": args.source_role,
                "source_member": args.generations.name if hasattr(args.generations,"name") else str(args.generations),
                "question_id": g["question_id"],
                "topic": g["topic"],
                "evidence_stratum": g["evidence_stratum"],
                "setting_id": g["setting_id"],
                "replicate_id": g["replicate_id"],
                "fold": fold_of[g["question_id"]],
                "observer": "observer_" + obs,
                "direct_event": out["direct_event"],
                "category": out["category"],
            })

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "dev_pool.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in pool), encoding="utf-8")
    (out_dir / "fold_map.json").write_bytes(args.fold_map.read_bytes())
    (out_dir / "dev_question_bank.json").write_bytes(args.question_bank.read_bytes())
    provenance = {
        "generations": str(args.generations),
        "generations_sha256": sha_path(args.generations),
        "labels_a": str(args.labels_a),
        "labels_a_sha256": sha_path(args.labels_a),
        "labels_b": str(args.labels_b),
        "labels_b_sha256": sha_path(args.labels_b),
        "fold_map_sha256": sha_path(args.fold_map),
        "question_bank_sha256": sha_path(args.question_bank),
        "rows": len(pool),
        "unique_samples": len({r["sample_id"] for r in pool}),
        "questions": len({r["question_id"] for r in pool}),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
