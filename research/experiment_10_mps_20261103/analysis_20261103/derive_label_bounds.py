"""Post-hoc D bounds tied across identical visible triples as well as A/B.

The initial diagnostics also retain wider, per-record bounds. These additional
bounds respect the frozen identical-triple consistency rule. No labels change.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

base = Path(__file__).resolve().parent
study = base.parent
diagnostics = json.loads((base / "diagnostics.json").read_text())
with zipfile.ZipFile(study / "evidence_20261103.zip") as archive:
    records = [json.loads(line) for line in archive.read("run/generations.jsonl").splitlines()]
    scored = json.loads(archive.read("run/scores.json"))["per_row"]
scores = {(r["sample_id"], r["observer"], r["predictor"]): r for r in scored}
observers = ("observer_a", "observer_b")
groups = defaultdict(list)
agreed = []
for r in records:
    sid = r["sample_id"]
    y = [scores[(sid, o, "Global")]["direct_event"] for o in observers]
    if y[0] == y[1]:
        agreed.append((sid, int(y[0])))
    else:
        groups[(r["requested_answerer"], r["question"], r["completion"])].append(sid)
comparisons = []
for original in diagnostics["sensitivity"]:
    left, right = original["left"], original["right"]
    def difference(sid, y):
        return math.fsum((scores[(sid, o, left)]["gate_probability"] - y) ** 2
                         - (scores[(sid, o, right)]["gate_probability"] - y) ** 2 for o in observers) / 2
    fixed = math.fsum(difference(sid, y) for sid, y in agreed)
    endpoints = [[math.fsum(difference(sid, y) for sid in ids) for y in (0, 1)] for ids in groups.values()]
    interval = [(fixed + math.fsum(choose(v) for v in endpoints)) / len(records) for choose in (min, max)]
    outer = original["common_D_on_disagreements_range"]
    if not outer[0] - 1e-12 <= interval[0] <= interval[1] <= outer[1] + 1e-12:
        raise ValueError("Tied bounds fall outside the per-record bounds")
    comparisons.append({"left":left, "right":right, "recorded_delta":original["recorded_delta"],
                        "triple_consistent_common_D_range":interval, "per_record_outer_range":outer})
result = {"status":"ADDITIONAL_POST_HOC_IDENTICAL_TRIPLE_SENSITIVITY",
          "diagnostics_sha256":hashlib.sha256((base/"diagnostics.json").read_bytes()).hexdigest(),
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "disagreement_records":sum(map(len,groups.values())),"distinct_disagreement_triples":len(groups),
          "comparisons":comparisons,
          "scope":"All agreed D labels assumed correct. Each ambiguous triple has one hypothetical D shared across observers and exact duplicates. This is an algebraic envelope, not a confidence interval or adjudication."}
with (base/"label_sensitivity.json").open("x") as handle:
    json.dump(result,handle,indent=2)
    handle.write("\n")
print(json.dumps(result,indent=2))
