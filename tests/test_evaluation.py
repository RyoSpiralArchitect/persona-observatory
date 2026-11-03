"""Analytic score checks and rejection of incomplete or malformed evidence."""
import copy
import math
import unittest

from persona_observatory.evaluation import score_rows, validate_forecasts


def label(sid, judgment="support", mode="direct"):
    if mode == "none":
        judgment = "none"
    return {
        "sample_id": sid,
        "answerer_answer_mode": mode,
        "answerer_judgment": judgment,
        "answerer_stance": "none" if mode == "none" else ("tentative" if judgment == "withhold" else "decisive"),
        "limit_owners": [],
        "other_answer": False,
        "character_description": False,
        "note": "The visible answer states the coded position.",
    }


def forecast(sid, observer="observer_a", predictor="G", probabilities=None):
    return {
        "sample_id": sid,
        "observer": observer,
        "predictor": predictor,
        "category_probabilities": list(probabilities or [0.5, 0.1, 0.1, 0.1, 0.2]),
    }


class ForecastValidationTests(unittest.TestCase):
    def grid(self):
        return [forecast(sid, observer, predictor)
                for sid in ["s1", "s2"]
                for observer in ["observer_a", "observer_b"]
                for predictor in ["G", "H"]]

    def test_complete_grid_and_dimensions(self):
        rows = self.grid()
        result = validate_forecasts(rows, ["s2", "s1"])
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["sample_ids"], ["s2", "s1"])
        self.assertEqual(result["forecast_rows"], 8)
        self.assertEqual(result["observers"], ["observer_a", "observer_b"])
        self.assertEqual(result["predictors"], ["G", "H"])
        self.assertEqual(result["category_order"], ["support", "reject", "withhold", "mixed", "no_direct"])

    def test_incomplete_cartesian_grid(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_forecasts(self.grid()[:-1], ["s1", "s2"])

    def test_duplicate_forecast(self):
        rows = self.grid()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_forecasts(rows + [copy.deepcopy(rows[0])], ["s1", "s2"])

    def test_unknown_sample(self):
        rows = self.grid()
        rows[0]["sample_id"] = "s3"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_forecasts(rows, ["s1", "s2"])

    def test_missing_whole_sample(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_forecasts([forecast("s1")], ["s1", "s2"])

    def test_duplicate_expected_sample(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_forecasts([forecast("s1")], ["s1", "s1"])

    def test_empty_inputs(self):
        for rows, ids in [([], ["s1"]), ([forecast("s1")], []), ({}, ["s1"]), ([forecast("s1")], "s1")]:
            with self.subTest(rows=rows, ids=ids), self.assertRaises(ValueError):
                validate_forecasts(rows, ids)

    def test_exact_record_fields(self):
        for change in ["extra", "missing"]:
            row = forecast("s1")
            if change == "extra":
                row["note"] = "not a forecast field"
            else:
                del row["observer"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_forecasts([row], ["s1"])

    def test_unsafe_identifiers(self):
        for field, value in [("sample_id", "../s1"), ("sample_id", "s1\n"),
                             ("observer", "../observer_a"), ("observer", "1observer"),
                             ("predictor", "model/G"), ("predictor", ""), ("predictor", "A" * 65)]:
            row = forecast("s1")
            row[field] = value
            ids = [value] if field == "sample_id" else ["s1"]
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_forecasts([row], ids)

    def test_invalid_probabilities(self):
        for probabilities in [
            [0.0, 0.2, 0.2, 0.2, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.1, 0.2, 0.2, 0.2, 0.5],
            [float("nan"), 0.1, 0.1, 0.1, 0.2],
            [float("inf"), 0.1, 0.1, 0.1, 0.2],
            [10 ** 1000, 0.1, 0.1, 0.1, 0.2],
            [True, 0.1, 0.1, 0.1, 0.2],
            ["0.5", 0.1, 0.1, 0.1, 0.2],
            [0.49, 0.1, 0.1, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.3],
        ]:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                validate_forecasts([forecast("s1", probabilities=probabilities)], ["s1"])

    def test_sum_tolerance_does_not_renormalize(self):
        probabilities = [0.5000000002, 0.1, 0.1, 0.1, 0.2]
        row = forecast("s1", probabilities=probabilities)
        before = copy.deepcopy(row)
        validate_forecasts([row], ["s1"])
        self.assertEqual(row, before)


class ScoringTests(unittest.TestCase):
    def test_hand_calculated_five_category_and_binary_losses(self):
        forecasts = [forecast("s1"), forecast("s2")]
        labels = {"observer_a": [label("s1"), label("s2", mode="none")]}
        before = copy.deepcopy((forecasts, labels))
        result = score_rows(forecasts, labels)
        first, second = result["per_row"]
        self.assertAlmostEqual(first["category_brier"], 0.32, places=14)
        self.assertAlmostEqual(first["category_nll"], 0.6931471805599453, places=14)
        self.assertAlmostEqual(first["gate_brier"], 0.04, places=14)
        self.assertAlmostEqual(first["gate_nll"], 0.22314355131420976, places=14)
        self.assertAlmostEqual(second["category_brier"], 0.92, places=14)
        self.assertAlmostEqual(second["category_nll"], 1.6094379124341003, places=14)
        self.assertAlmostEqual(second["gate_brier"], 0.64, places=14)
        self.assertAlmostEqual(second["gate_nll"], 1.6094379124341003, places=14)
        means = result["summary"][0]
        self.assertEqual(means["n"], 2)
        self.assertAlmostEqual(means["category_brier"], 0.62, places=14)
        self.assertAlmostEqual(means["category_nll"], 1.1512925464970227, places=14)
        self.assertAlmostEqual(means["gate_brier"], 0.34, places=14)
        self.assertAlmostEqual(means["gate_nll"], 0.916290731874155, places=14)
        self.assertEqual((forecasts, labels), before)

    def test_observers_remain_separate_and_reported_judgment_is_not_direct(self):
        forecasts = [forecast(sid, observer) for sid in ["s1", "s2"]
                     for observer in ["observer_a", "observer_b"]]
        result = score_rows(forecasts, {
            "observer_a": [label("s1"), label("s2", mode="none")],
            "observer_b": [label("s2", judgment="reject", mode="reported"), label("s1", judgment="withhold")],
        })
        self.assertEqual(result["dimensions"]["samples"], 2)
        self.assertEqual(len(result["per_row"]), 4)
        self.assertEqual(len(result["summary"]), 2)
        self.assertTrue(all(row["n"] == 2 for row in result["summary"]))
        self.assertAlmostEqual(result["summary"][1]["category_brier"], 1.02, places=14)
        reported = next(row for row in result["per_row"] if row["sample_id"] == "s2" and row["observer"] == "observer_b")
        self.assertEqual(reported["category"], "no_direct")
        self.assertFalse(reported["direct_event"])

    def test_zero_or_missing_labels_rejected(self):
        cases = [None, {}, {"observer_a": []}, {"observer_a": [label("s2")]},
                 {"observer_a": [label("s1"), label("s1")]},
                 {"observer_b": [label("s1")]},
                 {"observer_a": [label("s1")], "observer_b": [label("s2")]}]
        for labels in cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                score_rows([forecast("s1")], labels)

    def test_label_schema_and_semantics_validated(self):
        for mutation in ["extra", "bad_enum", "missing", "inconsistent", "bad_id"]:
            row = label("s1")
            if mutation == "extra":
                row["category"] = "support"
            elif mutation == "bad_enum":
                row["answerer_judgment"] = "agree"
            elif mutation == "missing":
                del row["note"]
            elif mutation == "inconsistent":
                row["answerer_answer_mode"] = "none"
            else:
                row["sample_id"] = "../s1"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                score_rows([forecast("s1")], {"observer_a": [row]})

    def test_tiny_positive_no_direct_mass_has_finite_unclipped_nll(self):
        probabilities = [0.3, 0.3, 0.2, 0.2, 1e-300]
        result = score_rows([forecast("s1", probabilities=probabilities)],
                            {"observer_a": [label("s1", mode="none")]})
        row = result["per_row"][0]
        self.assertTrue(math.isfinite(row["gate_nll"]))
        self.assertAlmostEqual(row["gate_nll"], 690.7755278982137, places=11)
        self.assertEqual(row["gate_nll"], row["category_nll"])


if __name__ == "__main__":
    unittest.main()
