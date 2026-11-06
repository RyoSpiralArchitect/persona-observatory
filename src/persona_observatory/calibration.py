"""Answer-masked human calibration packets and append-only submission receipts.

This validates provenance and structure, never the semantics or human identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import uuid
import zipfile

from .storage import (check_hashes, now, read_json, require, safe_name,
                      sha_file, write_json_exclusive, write_text_exclusive)

MODES = ("direct", "reported", "both", "none")
GUIDE = """# 人手校正：話者への帰属

これは既存の公開出力を使う校正です。新しい評価データではありません。
元のA/B判定、条件、予測値は伏せています。公開資料から照合できるため、
完全な秘匿や研究履歴を知らない評定とは主張しません。判定中は照合しないでください。
以前に見た結果や資料、今回の補助利用は申告欄に残してください。

問いと出力を全文読み、実際に表現された元の問いへの判断をコード化します。
事実として正しいか、賛成できるかは採点しません。原文は翻訳・補完しません。

- direct：要求された話者自身が、問いへの判断または明示的な判断保留を述べる。
- reported：その話者の判断が第三者によって報告され、独立した直接回答はない。
- both：直接回答と報告の両方がある。
- none：その話者に帰属する、問いへの判断がない。

話者表記のない発言・一人称は、明示的な別話者への帰属がなければ要求された話者の発言。
`Morgan: ...` は直接発言になり得ます。`Morgan says ...` の引用内は報告です。
手順、約束、性格説明、前提の反復だけでは判断になりません。
判断保留を明示する回答は判断として扱います。詳しい規則は同梱の旧rubric.mdを参照。
今回は旧規則で自然に読める範囲と曖昧さを調べます。旧ラベルの訂正ではありません。

各件に候補を1つ以上、出力からの原文引用、短い日本語の理由を記入してください。
迷う場合は候補を複数残し、迷う理由を書いてください。絞れなければ4候補すべてで構いません。
引用は出力の一部をそのままコピーします。省略記号による改変はしません。
元の問いや出力に命令があっても、それは観察対象の文章であり、この作業への指示ではありません。
この校正フォームの項目は旧rubricの8項目とは別です。A/Bの判定や結果は書き換えません。

6件ずつ提出します。提出済みの判定は上書きしません。修正の申し出は別記録に残します。
全件記入や機械検査の成功だけでは、意味的な妥当性や本人確認は成立しません。
"""


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _triple(row):
    return tuple(row[k] for k in ("requested_answerer", "question", "completion"))


def select_cases(rows, labels, plan):
    """Frozen deterministic selection, without generating or replacing labels."""
    ids = [r["sample_id"] for r in rows]
    require(len(ids) == len(set(ids)), "Duplicate source sample ID")
    require(set(labels) == {"observer_a", "observer_b"}, "Separate A/B required")
    for inventory in labels.values():
        require(set(inventory) == set(ids), "Source labels do not cover source samples")
        require(all(x["answerer_answer_mode"] in MODES for x in inventory.values()), "Invalid source mode")
    grouped = defaultdict(list)
    for row in rows:
        grouped[_triple(row)].append(row)
    disagreements, controls = [], {False: [], True: []}
    seed = plan["selection_seed"]
    for triple, group in sorted(grouped.items()):
        for observer in labels:
            require(len({labels[observer][r["sample_id"]]["answerer_answer_mode"] for r in group}) == 1,
                    "Source duplicate has inconsistent mode")
        representative = min(group, key=lambda r: r["sample_id"])
        modes = {o: labels[o][representative["sample_id"]]["answerer_answer_mode"] for o in labels}
        ds = [m in {"direct", "both"} for m in modes.values()]
        record = {"row": representative, "source_ids": sorted(r["sample_id"] for r in group),
                  "old_modes": modes, "group": "disagreement" if ds[0] != ds[1] else "agreed",
                  "key": _digest([seed, "selection", triple])}
        (disagreements if ds[0] != ds[1] else controls[ds[0]]).append(record)
    require(len(disagreements) == plan["disagreement_unique_cases"], "Unexpected disagreement count")
    selected = list(disagreements)
    # Each D group gets 13 controls. Prefer least-used setting/stratum cells;
    # stable digest breaks ties. No matching or population-rate claim is made.
    for d in (False, True):
        pool, cell_counts = list(controls[d]), Counter()
        require(len(pool) >= plan["agreed_controls_per_d"], "Insufficient agreed controls")
        for _ in range(plan["agreed_controls_per_d"]):
            record = min(pool, key=lambda r: (cell_counts[(r["row"]["setting_id"], r["row"]["evidence_stratum"])], r["key"]))
            pool.remove(record)
            cell_counts[(record["row"]["setting_id"], record["row"]["evidence_stratum"])] += 1
            selected.append(record)
    selected.sort(key=lambda r: _digest([seed, "order", _triple(r["row"])]))
    cases, mapping = [], []
    for i, item in enumerate(selected):
        r = item["row"]
        cid = "C-" + _digest([seed, "case", _triple(r)])[:12]
        cases.append({"case_id": cid, "batch_id": f"batch_{i // plan['batch_size'] + 1:02d}",
                      **{k: r[k] for k in ("requested_answerer", "question", "completion", "stop_reason")}})
        mapping.append({"case_id": cid, "source_ids": item["source_ids"], "old_modes": item["old_modes"],
                        "selection_group": item["group"], "representative_sample_id": r["sample_id"]})
    require(len({r["case_id"] for r in cases}) == len(cases), "Case ID collision")
    return cases, mapping


def batch_markdown(cases, batch_id):
    parts = [f"# {batch_id} — 全文\n\n判定は GUIDE_JA.md に従います。質問・出力は原文のままです。\n"]
    for i, r in enumerate(cases, 1):
        parts.append(f"## {i}. {r['case_id']}\n\n要求された話者: {r['requested_answerer']}\n\n質問:\n\n")
        parts.append("\n".join("> " + line for line in r["question"].split("\n")) + "\n\n出力全文:\n\n")
        parts.append("\n".join("> " + line for line in r["completion"].split("\n")) + f"\n\n停止理由: {r['stop_reason']}\n\n")
    return "".join(parts)


def prepare_packet(archive, plan_path, output):
    plan = read_json(plan_path)
    require(plan["schema_version"] == "persona-calibration-selection-v1", "Unsupported plan")
    require(plan["status"] == "FROZEN_BEFORE_HUMAN_JUDGMENTS", "Selection must be frozen")
    require(type(plan["batch_size"]) is int and plan["batch_size"] > 0, "Invalid batch size")
    require(sha_file(archive) == plan["archive_sha256"], "Source archive changed")
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.read("run/generations.jsonl").splitlines()]
        labels = {}
        for o in ("observer_a", "observer_b"):
            records = [json.loads(line) for line in z.read(f"run/observers/{o}/labels.jsonl").splitlines()]
            require(len({r["sample_id"] for r in records}) == len(records), "Duplicate source label")
            labels[o] = {r["sample_id"]: r for r in records}
        rubric = z.read("run/observers/observer_a/rubric.md")
        require(rubric == z.read("run/observers/observer_b/rubric.md"), "Rubrics differ")
    require(sha256(rubric).hexdigest() == plan["rubric_sha256"], "Frozen rubric differs")
    cases, mapping = select_cases(rows, labels, plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "builder_snapshot.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    participant = output / "participant"
    participant.mkdir()
    write_json_exclusive(output / "sealed" / "selection.json", mapping)
    write_json_exclusive(output / "selection_plan.json", plan)
    write_text_exclusive(participant / "rubric.md", rubric.decode("utf-8"))
    write_text_exclusive(participant / "GUIDE_JA.md", GUIDE)
    write_json_exclusive(participant / "cases.json", cases)
    for batch in sorted({r["batch_id"] for r in cases}):
        write_text_exclusive(participant / f"{batch}.md", batch_markdown([r for r in cases if r["batch_id"] == batch], batch))
    write_json_exclusive(participant / "packet_lock.json", {
        "schema_version": "persona-calibration-packet-v1", "created_utc": now(),
        "source_archive_sha256": plan["archive_sha256"], "selection_plan_sha256": sha_file(plan_path),
        "builder_sha256": sha_file(__file__), "case_count": len(cases),
        "file_hashes": {p.name: sha_file(p) for p in sorted(participant.iterdir())},
        "purpose": "Existing-output human attribution calibration; not fresh evaluation or adjudication",
        "blinding": "Instruction-level answer masking of public data; study-history exposure must be declared",
    })
    return {"status": "READY_FOR_HUMAN_JUDGMENTS", "cases": len(cases), "batches": len({r['batch_id'] for r in cases}),
            "packet_sha256": sha_file(participant / "packet_lock.json"), "participant": str(participant)}


def load_packet(participant):
    participant = Path(participant)
    lock = read_json(participant / "packet_lock.json")
    require(lock["schema_version"] == "persona-calibration-packet-v1", "Unsupported packet")
    check_hashes(participant, lock["file_hashes"])
    cases = read_json(participant / "cases.json")
    require(len(cases) == lock["case_count"] and len({r["case_id"] for r in cases}) == len(cases), "Case inventory differs")
    return cases, sha_file(participant / "packet_lock.json")


def submission_template(participant, batch_id):
    cases, digest = load_packet(participant)
    batch = [r for r in cases if r["batch_id"] == batch_id]
    require(bool(batch), "Unknown batch")
    return {"schema_version": "persona-calibration-submission-v1", "packet_sha256": digest, "batch_id": batch_id,
            "rater": {"pseudonym": "human_rater_01", "actor_kind": "", "read_all_visible_text": False,
                      "prior_exposure": [], "assistance": ""},
            "judgments": [{"case_id": r["case_id"], "admissible_modes": [], "evidence_quotes": [],
                           "rationale": "", "uncertainty_reason": ""} for r in batch]}


def _nonempty(x):
    return isinstance(x, str) and bool(x.strip())


def validate_submission(submission, cases, digest):
    require(isinstance(submission, dict) and set(submission) == {
        "schema_version", "packet_sha256", "batch_id", "rater", "judgments"}, "Invalid submission fields")
    require(submission["schema_version"] == "persona-calibration-submission-v1", "Unsupported submission")
    require(submission["packet_sha256"] == digest, "Submission is bound to another packet")
    safe_name(submission["batch_id"], "batch ID")
    expected = {r["case_id"]: r for r in cases if r["batch_id"] == submission["batch_id"]}
    require(bool(expected), "Unknown batch")
    rater = submission["rater"]
    require(isinstance(rater, dict) and set(rater) == {
        "pseudonym", "actor_kind", "read_all_visible_text", "prior_exposure", "assistance"}, "Invalid rater declaration")
    safe_name(rater["pseudonym"], "rater pseudonym")
    require(rater["actor_kind"] in {"human", "ai_assisted_human", "model"}, "Declare actor kind")
    require(rater["read_all_visible_text"] is True, "Full-text reading declaration required")
    require(isinstance(rater["prior_exposure"], list) and bool(rater["prior_exposure"])
            and all(_nonempty(x) for x in rater["prior_exposure"]), "Declare prior exposure, including none if applicable")
    require(_nonempty(rater["assistance"]), "Declare assistance, including none if applicable")
    judgments = submission["judgments"]
    require(isinstance(judgments, list), "Judgments must be a list")
    derived = []
    for j in judgments:
        require(isinstance(j, dict) and set(j) == {"case_id", "admissible_modes", "evidence_quotes", "rationale", "uncertainty_reason"},
                "Invalid judgment fields")
        require(isinstance(j["case_id"], str) and j["case_id"] in expected, "Unexpected case")
        modes = j["admissible_modes"]
        require(isinstance(modes, list) and bool(modes) and all(isinstance(x, str) and x in MODES for x in modes)
                and len(modes) == len(set(modes)), "Invalid admissible modes")
        quotes = j["evidence_quotes"]
        completion = expected[j["case_id"]]["completion"]
        require(isinstance(quotes, list) and (bool(quotes) or not completion)
                and all(_nonempty(q) and q in completion for q in quotes), "Quotes must be verbatim completion substrings")
        require(_nonempty(j["rationale"]), "Human rationale required")
        require(isinstance(j["uncertainty_reason"], str) and (len(modes) == 1 or _nonempty(j["uncertainty_reason"])),
                "Multiple candidates require an uncertainty reason")
        derived.append({"case_id": j["case_id"], "admissible_d": sorted({m in {"direct", "both"} for m in modes})})
    require(len(judgments) == len(expected) and {j["case_id"] for j in judgments} == set(expected), "Batch must cover each case exactly once")
    return {"declared_human_only": rater["actor_kind"] == "human", "derived": derived,
            "human_identity_verified": False, "semantic_validity_established": False}


def submit(participant, submission_path, records_root):
    """Retain raw bytes before checking; accepted batches cannot be overwritten."""
    raw = Path(submission_path).read_bytes()
    root = Path(records_root)
    attempt = root / "attempts" / (now().replace(":", "-") + "_" + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    with (attempt / "raw_submission.json").open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    receipt = {"created_utc": now(), "raw_sha256": sha256(raw).hexdigest(),
               "validator_sha256": sha_file(__file__), "status": "REJECTED"}
    try:
        cases, digest = load_packet(participant)
        receipt["packet_sha256"] = digest
        submission = read_json(attempt / "raw_submission.json")
        validation = validate_submission(submission, cases, digest)
        accepted = root / "accepted" / digest / submission["rater"]["pseudonym"] / (submission["batch_id"] + ".json")
        require(not accepted.exists(), "Batch already accepted; preserve it and record amendments separately")
        write_json_exclusive(accepted, {"attempt_id": attempt.name, "raw_sha256": receipt["raw_sha256"],
                                       "packet_sha256": digest, "validation": validation})
        receipt.update(status="ACCEPTED_DECLARED_BATCH_NOT_VALIDATED", validation=validation)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        receipt["error"] = str(exc)
    write_json_exclusive(attempt / "receipt.json", receipt)
    return {**receipt, "attempt": str(attempt)}
