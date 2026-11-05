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


def read_jsonl(zf, member):
    data = zf.read(PREFIX + member)
    return [json.loads(line) for line in data.splitlines()], hashlib.sha256(data).hexdigest()


def outcome(label):
    direct = label["answerer_answer_mode"] == "direct"
    return direct, (label["answerer_judgment"] if direct else "no_direct")


def main():
    zf = zipfile.ZipFile(ARCHIVE)
    raw, sha = {}, {}
    for key, member in MEMBERS.items():
        blob = zf.read(PREFIX + member)
        sha[member] = hashlib.sha256(blob).hexdigest()
        raw[key] = [json.loads(line) for line in blob.splitlines()] if member.endswith(".jsonl") else json.loads(blob)

    questions = {q["question_id"]: q for q in raw["dev_bank"] + raw["eval_bank"]}
    folds = {qid: q.get("development_fold") for qid, q in questions.items()}
    # Records carry the authoritative fold for old development; N-questions use topic_index + 7.
    rec_meta = {}
    for rec in raw["dev_records"]:
        rec_meta[rec["sample_id"]] = (rec["source_record"], rec["development_fold"])
    eval_meta = {g["sample_id"]: g for g in raw["eval_gens"]}
    n_folds = {q["question_id"]: q["topic_index"] + 7 for q in raw["eval_bank"]}

    labels = {"a": {}, "b": {}}
    for obs in ("a", "b"):
        for row in raw[f"dev_labels_{obs}"]:
            labels[obs][row["sample_id"]] = row
        for row in raw[f"eval_labels_{obs}"]:
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
            pool.append({**base, "observer": obs, "direct_event": direct, "category": category})

    for rec in raw["dev_records"]:
        meta, fold = rec_meta[rec["sample_id"]]
        emit(rec["sample_id"], meta, "v9_development", MEMBERS["dev_records"], fold)
    for g in raw["eval_gens"]:
        emit(g["sample_id"], g, "v9_former_evaluation", MEMBERS["eval_gens"], n_folds[g["question_id"]])

    # Checks mirror the verified feasibility record.
    assert len({r["sample_id"] for r in pool}) == 624, "expected 624 generations"
    assert len(pool) == 1248, "expected 624 x 2 observer rows"
    assert len(seen_q) == 52, "expected 52 questions"
    strata = {}
    for qid in seen_q:
        strata[questions[qid]["evidence_stratum"]] = strata.get(questions[qid]["evidence_stratum"], 0) + 1
    assert strata == {"limited": 13, "contradictory": 13, "supportive": 13, "confounded": 13}, strata
    q_fold = {}
    for qid in seen_q:
        if qid in n_folds:
            q_fold[qid] = n_folds[qid]
        else:
            fs = {fm for m, fm in rec_meta.values() if m["question_id"] == qid}
            assert len(fs) == 1, f"inconsistent folds for {qid}"
            q_fold[qid] = fs.pop()
    fold_map = {str(f): sorted(qid for qid in seen_q if q_fold[qid] == f) for f in range(13)}
    assert all(len(v) == 4 for v in fold_map.values()), fold_map

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
        "note": "Read-only extraction of v9 development plus former evaluation rows with archived A/B labels. Development role is new; labels and generations are unchanged historical records."}, indent=2) + "\n")
    print(f"Wrote {len(pool)} rows / {len(seen_q)} questions / {len(dev_bank)}-question bank to {OUT}")


if __name__ == "__main__":
    main()
