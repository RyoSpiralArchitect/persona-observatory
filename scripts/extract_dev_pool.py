"""Build the experiment-10 development pool from the preserved archive.

Read-only: the ZIP is opened for reading and never modified. Output rows carry
the archived semantic labels of both observer branches plus provenance (archive
member, source role, question and setting identifiers). The proposed fold map
follows validation/experiment_10_feasibility_20261103.json: v9 development_fold
0..6 is kept for the 28 old development questions, and each former-evaluation
question takes fold 7 + topic_index.

Usage: python3 scripts/extract_dev_pool.py [output_dir]
Default output: research/experiment_10/development/
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from persona_observatory.observations import derived_outcome  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archives" / "experiment_09_20261103.zip"
PREFIX = "persona_shrinkage_pilot/"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research" / "experiment_10" / "development"

MEMBERS = {
    "dev_records": "development/records.jsonl",
    "dev_labels_a": "development/labels_observer_a.jsonl",
    "dev_labels_b": "development/labels_observer_b.jsonl",
    "dev_bank": "development/source_question_bank.json",
    "eval_gens": "data/target_generations.jsonl",
    "eval_labels_a": "observers/target/observer_a/labels.jsonl",
    "eval_labels_b": "observers/target/observer_b/labels.jsonl",
    "eval_bank": "question_bank.json",
}
CLAIM_TEMPLATE = "What can be concluded about whether {claim} from the stated evidence?"


def outcome(label):
    """Canonical rubric outcome: 'direct' and 'both' both count as direct."""
    result = derived_outcome(label)
    return result["direct_event"], result["category"]


def main():
    zf = zipfile.ZipFile(ARCHIVE)
    raw, sha = {}, {}
    for key, member in MEMBERS.items():
        blob = zf.read(PREFIX + member)
        sha[member] = hashlib.sha256(blob).hexdigest()
        raw[key] = [json.loads(line) for line in blob.splitlines()] if member.endswith(".jsonl") else json.loads(blob)

    questions = {q["question_id"]: q for q in raw["dev_bank"] + raw["eval_bank"]}
    # Records carry the authoritative fold for old development; N-questions use topic_index + 7.
    rec_meta = {}
    for rec in raw["dev_records"]:
        if rec["sample_id"] in rec_meta:
            raise SystemExit("Duplicate development sample_id: " + rec["sample_id"])
        rec_meta[rec["sample_id"]] = (rec["source_record"], rec["development_fold"])
    eval_meta = {}
    for g in raw["eval_gens"]:
        if g["sample_id"] in eval_meta:
            raise SystemExit("Duplicate evaluation sample_id: " + g["sample_id"])
        eval_meta[g["sample_id"]] = g
    n_folds = {q["question_id"]: q["topic_index"] + 7 for q in raw["eval_bank"]}

    labels = {"a": {}, "b": {}}
    for obs in ("a", "b"):
        for row in raw[f"dev_labels_{obs}"] + raw[f"eval_labels_{obs}"]:
            if row["sample_id"] in labels[obs]:
                raise SystemExit(f"Duplicate label for {row['sample_id']} in observer {obs}")
            labels[obs][row["sample_id"]] = row

    pool, seen_q = [], set()
    def emit(sample_id, meta, source, member, fold):
        qid = meta["question_id"]
        seen_q.add(qid)
        base = {"sample_id": sample_id, "source_role": source, "source_member": PREFIX + member,
                "question_id": qid, "topic": meta.get("question_name", meta.get("topic", qid)),
                "evidence_stratum": meta["evidence_stratum"], "setting_id": meta["setting_id"],
                "replicate_id": meta.get("replicate_id"), "fold": fold}
        for obs in ("a", "b"):
            label = labels[obs].get(sample_id)
            if label is None:
                raise SystemExit(f"Missing {obs} label for {sample_id}")
            direct, category = outcome(label)
            pool.append({**base, "observer": "observer_" + obs, "direct_event": direct, "category": category})

    for rec in raw["dev_records"]:
        meta, fold = rec_meta[rec["sample_id"]]
        emit(rec["sample_id"], meta, "v9_development", MEMBERS["dev_records"], fold)
    for g in raw["eval_gens"]:
        emit(g["sample_id"], g, "v9_former_evaluation", MEMBERS["eval_gens"], n_folds[g["question_id"]])

    # Checks mirror the verified feasibility record; raises survive -O.
    if len({r["sample_id"] for r in pool}) != 624:
        raise SystemExit("expected 624 generations")
    if len(pool) != 1248:
        raise SystemExit("expected 624 x 2 observer rows")
    if len(seen_q) != 52:
        raise SystemExit("expected 52 questions")
    strata = {}
    for qid in seen_q:
        strata[questions[qid]["evidence_stratum"]] = strata.get(questions[qid]["evidence_stratum"], 0) + 1
    if strata != {"limited": 13, "contradictory": 13, "supportive": 13, "confounded": 13}:
        raise SystemExit("unexpected stratum balance: " + str(strata))
    q_fold = {}
    for qid in seen_q:
        if qid in n_folds:
            q_fold[qid] = n_folds[qid]
        else:
            fs = {fm for m, fm in rec_meta.values() if m["question_id"] == qid}
            if len(fs) != 1:
                raise SystemExit(f"inconsistent folds for {qid}")
            q_fold[qid] = fs.pop()
    fold_map = {str(f): sorted(qid for qid in seen_q if q_fold[qid] == f) for f in range(13)}
    if not all(len(v) == 4 for v in fold_map.values()):
        raise SystemExit("fold map is not 13 folds of 4 questions")

    dev_bank = [{"question_id": q["question_id"], "topic": q["topic"],
                 "evidence_stratum": q["evidence_stratum"],
                 "question": q["evidence"] + "\n\n" + CLAIM_TEMPLATE.format(claim=q["claim"])}
                for q in raw["dev_bank"] + raw["eval_bank"] if q["question_id"] in seen_q]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool))
    (OUT / "dev_question_bank.json").write_text(json.dumps(dev_bank, ensure_ascii=False, indent=2) + "\n")
    (OUT / "fold_map.json").write_text(json.dumps(fold_map, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "archive": "archives/experiment_09_20261103.zip",
        "member_sha256": sha,
        "generations": 624, "observer_rows": len(pool), "questions": len(seen_q),
        "questions_by_stratum": strata,
        "note": "Read-only extraction of v9 development plus former evaluation rows with archived A/B labels. Development role is new; labels and generations are unchanged historical records. Corrected 2026-11-06: outcomes now derive through observations.derived_outcome, so archived 'both' modes count as direct (13 label rows affected); an earlier draft treated them as no_direct."}, indent=2) + "\n")
    print(f"Wrote {len(pool)} rows / {len(seen_q)} questions / {len(dev_bank)}-question bank to {OUT}")


if __name__ == "__main__":
    main()
