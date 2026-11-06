import unittest

from persona_observatory.analysis import paired_delta, primary_report


def score_row(sid, observer, predictor, brier, nll=None):
    return {"sample_id": sid, "observer": observer, "predictor": predictor,
            "category": "support", "direct_event": True,
            "category_probabilities": [0.9, 0.02, 0.02, 0.02, 0.04],
            "gate_probability": 0.96, "category_brier": brier,
            "category_nll": 0.1, "gate_brier": brier,
            "gate_nll": nll if nll is not None else brier}


def make_rows(brier_of, samples):
    return [score_row(sid, obs, pred, brier_of[(obs, pred, sid)])
            for sid in samples for obs in ("a", "b")
            for pred in ("global", "evidence", "setting", "joint")]


class PairedDeltaTest(unittest.TestCase):
    def setUp(self):
        self.samples = [f"s_{q}_{i}" for q in ("q1", "q2") for i in range(3)]
        self.questions = {s: {"question_id": s.split("_")[1]} for s in self.samples}

    def test_exact_pairing(self):
        b = {(o, p, s): (0.1 if p == "joint" else 0.2)
             for o in ("a", "b") for p in ("global", "evidence", "setting", "joint")
             for s in self.samples}
        result = paired_delta(make_rows(b, self.samples), self.questions, "joint", "evidence")
        self.assertAlmostEqual(result["delta_ab_mean"], -0.1)
        self.assertEqual(result["incomplete_cells"], 0)

    def test_missing_cell_flags_incomplete(self):
        rows = [r for r in make_rows({k: 0.2 for k in
                [(o, p, s) for o in ("a", "b") for p in ("global", "evidence", "setting", "joint")
                 for s in self.samples]}, self.samples)
                if not (r["observer"] == "a" and r["predictor"] == "joint")]
        result = paired_delta(rows, self.questions, "joint", "evidence")
        self.assertGreater(result["incomplete_cells"], 0)
        report = primary_report(rows, self.questions)
        self.assertEqual(report["primary_decision"], "INCOMPLETE")

    def test_question_weighting_not_row_weighting(self):
        # q1 has 3 gens, q2 has 3 gens: equal weight either way. Change one
        # question's delta and confirm the mean is over questions.
        b = {}
        for o in ("a", "b"):
            for p in ("global", "evidence", "setting", "joint"):
                for s in self.samples:
                    q = self.questions[s]["question_id"]
                    b[(o, p, s)] = 0.0 if p == "joint" else (0.3 if q == "q1" else 0.1)
        result = paired_delta(make_rows(b, self.samples), self.questions, "joint", "evidence")
        self.assertAlmostEqual(result["delta_ab_mean"], -0.2)  # mean of -0.3, -0.1


class PrimaryReportTest(unittest.TestCase):
    def test_support_and_progression(self):
        samples = [f"s_{q}_{i}" for q in ("q1", "q2", "q3") for i in range(4)]
        questions = {s: {"question_id": s.split("_")[1]} for s in samples}
        b = {(o, p, s): (0.2 if p == "joint" else 0.21)
             for o in ("a", "b") for p in ("global", "evidence", "setting", "joint")
             for s in samples}
        report = primary_report(make_rows(b, samples), questions)
        self.assertEqual(report["primary_decision"], "SUPPORT")
        # delta -0.01 <= -0.005, both branches negative, joint<=global, nll equal
        self.assertTrue(report["progression"]["met"])

    def test_not_supported(self):
        samples = ["s_q1_0"]
        questions = {"s_q1_0": {"question_id": "q1"}}
        b = {(o, p, s): 0.2 for o in ("a", "b")
             for p in ("global", "evidence", "setting", "joint") for s in samples}
        report = primary_report(make_rows(b, samples), questions)
        self.assertEqual(report["primary_decision"], "NOT_SUPPORTED")
        self.assertFalse(report["progression"]["met"])


if __name__ == "__main__":
    unittest.main()
