import math
import unittest

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
        # Global grouping ignores rho entirely; every candidate ties and inf wins.
        chosen = select_rho(self.rows, self.fold_of, "global")
        self.assertEqual(chosen["rho"], math.inf)

    def test_branch_rows_stay_separate(self):
        # Observers label the same generations; disagreement makes them
        # separate rows, never merged counts.
        rows = [row("q1", "limited", "s1", True, "a"), row("q1", "limited", "s1", False, "b")]
        fitted_a = fit_predictor([r for r in rows if r["observer"] == "a"], "global", 1.0)
        fitted_b = fit_predictor([r for r in rows if r["observer"] == "b"], "global", 1.0)
        self.assertNotEqual(fitted_a["base"]["g_d"], fitted_b["base"]["g_d"])


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
