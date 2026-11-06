#!/usr/bin/env python3
"""Compute per-family five-category composition JS divergences for E13.

Records the scores input, eval design, and this script by SHA-256 in the
output so the E13 separation measurement is reproducible in-repo.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from persona_observatory.storage import sha_file, write_json_exclusive

CATS = ["support", "reject", "withhold", "mixed", "no_direct"]
PAIRS = [("lexswap_cautious", "cautious"), ("lexswap_confident", "confident"),
         ("gloss_cautious", "cautious"), ("gloss_cautious", "confident"),
         ("role_alex", "neutral"), ("cautious", "confident")]


def jsd(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    def k(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return (k(p, m) + k(q, m)) / 2


def family_of(setting_id):
    return setting_id[len("eval_"):].rsplit("_instruction_p", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--design", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    scores = json.loads(args.scores.read_text())
    plan = {r["sample_id"]: r for r in map(json.loads, args.design.read_text().splitlines())}
    out = {}
    for obs in ("observer_a", "observer_b"):
        cnt = defaultdict(Counter)
        for r in scores["per_row"]:
            if r["observer"] == obs and r["predictor"] == "global":
                cnt[family_of(plan[r["sample_id"]]["setting_id"])][r["category"]] += 1
        dist = {f: [c[k] / sum(c.values()) for k in CATS] for f, c in cnt.items()}
        out[obs] = {
            "composition": {f: dict(zip(CATS, dist[f])) for f in dist},
            "js_divergence": {f"{a}:{b}": jsd(dist[a], dist[b]) for a, b in PAIRS},
        }
    write_json_exclusive(args.output, {
        "schema_version": "pobs-jsd-report-v1",
        "provenance": {"scores_sha256": sha_file(args.scores),
                       "design_sha256": sha_file(args.design),
                       "script_sha256": sha_file(Path(__file__).resolve())},
        "categories": CATS, "pairs": [list(p) for p in PAIRS], "results": out})
    print(json.dumps({o: {k: round(v, 3) for k, v in d["js_divergence"].items()}
                      for o, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
