"""Draft probability-safe shrinkage components; not a frozen scientific fit.

No CLI, inference, label generation, or real-data model selection is performed.
The Experiment 10 predictor module remains unchanged.
"""
from collections import Counter, defaultdict
import math

from .evaluation import CATEGORY_ORDER
from .storage import require

WEIGHTS = (0.0, 0.25, 0.5, 1.0)
CANDIDATES = tuple((e, i) for e in WEIGHTS for i in WEIGHTS if i <= e)


def _logit(p):
    return math.log(p) - math.log1p(-p)


def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def fit_components(rows, outcomes, *, cell_prior=16.0):
    """Fit supplied development counts only; callers must freeze data roles.

    Setting has fixed rho=1. Evidence uses Jeffreys-smoothed marginal rates.
    Cell rates borrow `cell_prior` observations from their Setting baseline.
    These are statistical components, not causal effects or the old ANOVA.
    """
    require(bool(rows), "Development rows required")
    require(type(cell_prior) in (int, float) and math.isfinite(cell_prior) and cell_prior > 0,
            "Cell prior must be finite and positive")
    ids = [r['sample_id'] for r in rows]
    require(len(ids) == len(set(ids)), "Duplicate development sample")
    require(all(s in outcomes and outcomes[s] in CATEGORY_ORDER for s in ids), "Invalid development outcome")
    groups = {name: defaultdict(lambda: [0, 0]) for name in ('setting', 'evidence', 'cell')}
    counts = Counter(outcomes[s] for s in ids)
    direct = len(rows) - counts['no_direct']
    global_d = (direct + 0.5) / (len(rows) + 1)
    for row in rows:
        s, e = row['setting_id'], row['evidence_stratum']
        require(isinstance(s, str) and bool(s) and isinstance(e, str) and bool(e), "Invalid design field")
        d = outcomes[row['sample_id']] != 'no_direct'
        for name, key in (('setting', s), ('evidence', e), ('cell', (s, e))):
            groups[name][key][0] += 1
            groups[name][key][1] += d
    settings = {s: (d + n * global_d) / (n * 2) for s, (n, d) in groups['setting'].items()}
    evidence = {e: _logit((d + 0.5) / (n + 1)) - _logit(global_d)
                for e, (n, d) in groups['evidence'].items()}
    cells = []
    for (s, e), (n, d) in sorted(groups['cell'].items()):
        p = (d + cell_prior * settings[s]) / (n + cell_prior)
        cells.append({'setting_id': s, 'evidence_stratum': e, 'n': n, 'd': d,
                      'gate': p, 'logit_residual': _logit(p) - _logit(settings[s]) - evidence[e]})
    return {'schema_version': 'persona-component-profile-draft-v1', 'status': 'ENGINEERING_CANDIDATE',
            'n': len(rows), 'global_d': global_d, 'setting_rho': 1.0, 'cell_prior': cell_prior,
            'setting_gates': settings, 'evidence_logit_shifts': evidence, 'cells': cells,
            'composition': [(counts[k] + 0.5) / (direct + 2) for k in CATEGORY_ORDER[:-1]]}


def predict_components(profile, row, *, evidence_weight, interaction_weight):
    """Return pooled-composition category probabilities, or Setting at (0, 0).

    0 <= interaction <= evidence <= 1. Unseen cells have no residual; unseen
    evidence has zero marginal shift; unseen settings fall back to global.
    """
    require(all(type(w) in (int, float) and math.isfinite(w) and 0 <= w <= 1
                for w in (evidence_weight, interaction_weight))
            and interaction_weight <= evidence_weight, "Require 0 <= interaction <= evidence <= 1")
    s, e = row['setting_id'], row['evidence_stratum']
    base = profile['setting_gates'].get(s, profile['global_d'])
    shift = profile['evidence_logit_shifts'].get(e, 0.0)
    cell = next((c for c in profile['cells'] if c['setting_id'] == s and c['evidence_stratum'] == e), None)
    residual = cell['logit_residual'] if cell is not None else 0.0
    if evidence_weight == 0 and interaction_weight == 0:
        gate = base
    elif evidence_weight == interaction_weight == 1 and cell is not None:
        gate = cell['gate']
    else:
        gate = _sigmoid(_logit(base) + evidence_weight * shift + interaction_weight * residual)
    return [gate * c for c in profile['composition']] + [1 - gate]
