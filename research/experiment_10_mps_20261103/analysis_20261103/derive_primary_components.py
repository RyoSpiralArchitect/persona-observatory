"""Additional post-hoc algebra: distinguish setting main effects from interactions.

Run with the analysis directory as cwd. Existing outputs are never overwritten.
"""
import hashlib
import json
import math
from pathlib import Path

path = Path(__file__).resolve().parent
raw = (path / "diagnostics.json").read_bytes()
diagnostics = json.loads(raw)
mean = lambda values: math.fsum(values) / len(values)
branches = {}
for observer in ("observer_a", "observer_b"):
    cells = [c for c in diagnostics["cells"] if c["observer"] == observer]
    g = mean([c["development_d_rate"] for c in cells])
    h = mean([c["evaluation_d_rate"] for c in cells])
    settings = sorted({c["setting"] for c in cells})
    dev = {s: mean([c["development_d_rate"] for c in cells if c["setting"] == s]) for s in settings}
    evaluation = {s: mean([c["evaluation_d_rate"] for c in cells if c["setting"] == s]) for s in settings}
    cost = mean([(0.5 * (dev[s] - g)) ** 2 for s in settings])
    alignment = 2 * mean([0.5 * (dev[s] - g) * (evaluation[s] - h) for s in settings])
    interaction = diagnostics["joint_setting_components"][observer]["joint_minus_setting"]["interaction"]
    primary = mean([c["brier"]["Joint"] - c["brier"]["Evidence"] for c in cells])
    reconstructed = cost - alignment + interaction["brier_increment"]
    if abs(primary - reconstructed) > 1e-12:
        raise ValueError("Primary decomposition does not match the frozen Brier comparison")
    branches[observer] = {"setting_main": {"squared_change_cost": cost,
                        "twice_outcome_alignment": alignment, "brier_increment": cost - alignment},
                        "interaction": interaction, "primary_reconstructed": reconstructed}
result = {"status": "ADDITIONAL_POST_HOC_ALGEBRA_NO_REFIT",
          "diagnostics_sha256": hashlib.sha256(raw).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "identity": "Joint - Evidence Brier = setting-main increment + interaction increment, on the balanced grid with all local shrinkage rho=1.",
          "branches": branches,
          "ab_mean": {k: mean([b[k]["brier_increment"] for b in branches.values()])
                      for k in ("setting_main", "interaction")},
          "scope": "Algebraic accounting after the outcome was known. No causal effect, new predictor, or replacement primary test."}
with (path / "primary_components.json").open("x") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps(result, indent=2))
