import math
import unittest

from persona_observatory.evaluation import CATEGORY_ORDER as CAT_ORDER
from persona_observatory.forecasting import (
    PREDICTORS, RHO_CANDIDATES, base_rates, build_forecasts, cell_rate,
    category_probabilities, cv_score, fit_predictor, gate_probability,
    select_rho,
)


def row(qid, stratum, setting, direct, observer="a", category=None):
    return {"question_id": qid, "evidence_stratum": stratum, "setting_id": setting,
            "direct_event": direct, "observer": observer,
            "category": category or ("support" if direct else "no_direct")}


class RateTest(unittest.TestCase):
    def test_base_rates_smoothing(self):
        rows = [row("q1", "limited", "s1", True), row("q1", "limited", "s1", False)]
        base = base_rates(rows)
        self.assertEqual(base["n"], 2)
        self.assertAlmostEqual(base["g_d"], 1.5 / 3)
        self.assertAlmostEqual(sum(base["judgment_composition"].values()), 1.0)

    def test_empty_cell_and_inf_pool_to_global(self):
        self.assertEqual(cell_rate(0, 0, 0.4, 1.0), 0.4)
        self.assertEqual(cell_rate(5, 10, 0.4, math.inf), 0.4)

    def test_all_direct_cell(self):
        rows = [row("q1", "limited", "s1", True) for _ in range(4)]
        fitted = fit_predictor(rows, "setting", 1.0)
        r = gate_probability(fitted, row("qX", "limited", "s1", True))
        n_d = 4
        g = (n_d + 0.5) / (4 + 1)
        self.assertAlmostEqual(r, (4 + 1 * 4 * g) / (4 * 2))
        self.assertLess(r, 1.0)

    def test_probabilities_strict_and_sum_to_one(self):
        rows = [row("q1", "limited", "s1", True), row("q1", "limited", "s1", False, category="no_direct"),
                row("q2", "supportive", "s2", True, category="reject")]
        fitted = fit_predictor(rows, "joint", 4.0)
        p = category_probabilities(fitted, {"evidence_stratum": "confounded", "setting_id": "s9"})
        self.assertAlmostEqual(math.fsum(p), 1.0)
        self.assertTrue(all(0 < x < 1 for x in p))


class SelectionTest(unittest.TestCase):
    def setUp(self):
        # Setting s1 strongly predicts direct answers; s2 does not.
        self.rows = []
        for q in ("q1", "q2", "q3", "q4"):
            for obs in ("a", "b"):
                self.rows += [row(q, "limited", "s1", True, obs) for _ in range(3)]
                self.rows += [row(q, "limited", "s2", False, obs) for _ in range(3)]
        self.fold_of = {"q1": 0, "q2": 1, "q3": 2, "q4": 3}

    def test_blocked_folds_do_not_leak(self):
        # Held-out questions never reach the fit: a predictor grouped by
        # question would score perfectly if leakage occurred.
        score = cv_score(self.rows, self.fold_of, "setting", 0.25)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.3)

    def test_setting_beats_global_on_separable_data(self):
        setting = select_rho(self.rows, self.fold_of, "setting")
        glob = select_rho(self.rows, self.fold_of, "global")
        self.assertLess(setting["cv_gate_brier"][setting["rho"]],
                        glob["cv_gate_brier"][glob["rho"]])

    def test_tie_prefers_stronger_shrinkage(self):
        # Global is exactly the pooled baseline; every rho yields the identical
        # predictor, so all candidates tie and inf wins.
        chosen = select_rho(self.rows, self.fold_of, "global")
        self.assertEqual(chosen["rho"], math.inf)
        self.assertEqual(len(set(chosen["cv_gate_brier"].values())), 1)

    def test_global_fit_is_rho_invariant(self):
        fitted = {rho: fit_predictor(self.rows, "global", rho) for rho in RHO_CANDIDATES}
        rates = {rho: f["cell_rates"]["all"] for rho, f in fitted.items()}
        self.assertEqual(len(set(rates.values())), 1)

    def test_branch_rows_stay_separate(self):
        # Observers label the same generations; disagreement makes them
        # separate rows, never merged counts.
        rows = [row("q1", "limited", "s1", True, "a"), row("q1", "limited", "s1", False, "b")]
        fitted_a = fit_predictor([r for r in rows if r["observer"] == "a"], "global", 1.0)
        fitted_b = fit_predictor([r for r in rows if r["observer"] == "b"], "global", 1.0)
        self.assertNotEqual(fitted_a["base"]["g_d"], fitted_b["base"]["g_d"])

    def test_cv_fits_each_branch_separately(self):
        # Fully contradictory branches: a always direct, b never. A pooled fit
        # would land near 0.5 for both; branch-specific fits score near the
        # smoothed extremes of each branch.
        rows = []
        for q in ("q1", "q2", "q3", "q4"):
            rows += [row(q, "limited", "s1", True, "a") for _ in range(4)]
            rows += [row(q, "limited", "s1", False, "b") for _ in range(4)]
        score = cv_score(rows, self.fold_of, "global", 1.0)
        # Branch-specific fits predict near 1.0 for a and near 0.0 for b;
        # a pooled fit would predict ~0.5 for both and score near 0.25.
        self.assertLess(score, 0.05)


class ForecastGridTest(unittest.TestCase):
    def test_complete_grid(self):
        rows = [row("q1", "limited", "s1", True, obs) for obs in ("a", "b")]
        fitted = {obs: {p: fit_predictor([r for r in rows if r["observer"] == obs], p, 1.0)
                        for p in PREDICTORS} for obs in ("a", "b")}
        samples = [{"sample_id": "s1", "evidence_stratum": "limited", "setting_id": "s1"},
                   {"sample_id": "s2", "evidence_stratum": "supportive", "setting_id": "s2"}]
        grid = build_forecasts(fitted, samples)
        self.assertEqual(len(grid), 2 * 2 * 4)
        keys = {(r["sample_id"], r["observer"], r["predictor"]) for r in grid}
        self.assertEqual(len(keys), 16)
        for r in grid:
            self.assertAlmostEqual(math.fsum(r["category_probabilities"]), 1.0)


if __name__ == "__main__":
    unittest.main()

class CategoryModeTest(unittest.TestCase):
    def test_category_fit_and_predictions(self):
        rows = [
            {"question_id": "q1", "evidence_stratum": "supportive", "setting_id": "sA",
             "observer": "observer_a", "direct_event": True, "category": "support"},
            {"question_id": "q1", "evidence_stratum": "supportive", "setting_id": "sB",
             "observer": "observer_a", "direct_event": False, "category": "no_direct"},
            {"question_id": "q2", "evidence_stratum": "limited", "setting_id": "sA",
             "observer": "observer_a", "direct_event": True, "category": "withhold"},
            {"question_id": "q2", "evidence_stratum": "limited", "setting_id": "sB",
             "observer": "observer_a", "direct_event": True, "category": "reject"},
        ]
        fit = fit_predictor(rows, "setting", 1.0, mode="category")
        pa = category_probabilities(fit, {"setting_id": "sA"})
        pb = category_probabilities(fit, {"setting_id": "sB"})
        self.assertNotEqual(pa, pb)
        self.assertAlmostEqual(sum(pa), 1.0)
        # gate probability derives from no_direct mass
        self.assertAlmostEqual(gate_probability(fit, {"setting_id": "sA"}),
                               1.0 - pa[CAT_ORDER.index("no_direct")])
        # inf rho collapses every cell to the pooled base
        fit_inf = fit_predictor(rows, "setting", float("inf"), mode="category")
        for s in ("sA", "sB"):
            self.assertEqual(category_probabilities(fit_inf, {"setting_id": s}),
                             category_probabilities(fit_inf, {"setting_id": "x"}))
        # category-mode CV differentiates where gate cannot
        g = cv_score(rows, {"q1": 0, "q2": 1}, "setting", 1.0, "gate")
        c = cv_score(rows, {"q1": 0, "q2": 1}, "setting", 1.0, "category")
        self.assertIsInstance(g, float) and self.assertIsInstance(c, float)
        sel = select_rho(rows, {"q1": 0, "q2": 1}, "setting", "category")
        self.assertEqual(sel["metric"], "category")
        self.assertIn("cv_category_brier", sel)
