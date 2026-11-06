#!/usr/bin/env python3
"""Score human calibration answers against locked model observer labels."""
import argparse, json, math
from collections import Counter
from pathlib import Path

FIELDS = [("mode", "answerer_answer_mode"), ("judgment", "answerer_judgment"), ("stance", "answerer_stance")]

def kappa(items, key_field, human_field):
    agree = sum(1 for h, k in items if h[human_field] == k[key_field])
    n = len(items)
    pe = sum((sum(1 for h, _ in items if h[human_field] == v) / n) *
             (sum(1 for _, k in items if k[key_field] == v) / n)
             for v in {h[human_field] for h, _ in items} | {k[key_field] for _, k in items})
    po = agree / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--key", required=True, type=Path)
    args = ap.parse_args()
    human = {r["item"]: r for r in map(json.loads, args.answers.read_text().splitlines())}
    key = {r["item"]: r for r in map(json.loads, args.key.read_text())}
    items = [(human[i], key[i]) for i in sorted(human) if i in key]
    print(f"n={len(items)}")
    for hf, kf in FIELDS:
        po, kap = kappa(items, kf, hf)
        print(f"  {hf}: agreement {po:.3f}, Cohen's kappa {kap:.3f}")

if __name__ == "__main__":
    main()
