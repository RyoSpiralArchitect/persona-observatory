import copy
import math
import unittest

from persona_observatory.predictors import fit_profile, forecast_targets, predict, select_and_fit, validate_development


def label(sid, category):
    direct = category != "no_direct"
    return {"sample_id": sid, "answerer_answer_mode": "direct" if direct else "none",
            "answerer_judgment": category if direct else "none",
            "answerer_stance": ("mixed" if category == "mixed" else "tentative" if category == "withhold" else "decisive") if direct else "none",
            "limit_owners": [], "other_answer": False, "character_description": False,
            "note": "Invented count fixture, not a semantic annotation."}


class PredictorTests(unittest.TestCase):
    def fixture(self):
        rows = [{"sample_id": f"s{i}", "question_id": f"q{i//2}", "fold": i//2,
                 "setting_id": f"setting{i%2}", "evidence_stratum": "limited"} for i in range(8)]
        labels = {o: [label(r["sample_id"], "support" if i%2 else "no_direct") for i,r in enumerate(rows)]
                  for o in ("observer_a", "observer_b")}
        return rows, labels

    def test_hand_counts_and_global_fallback(self):
        rows,_ = self.fixture()
        outcomes = {r["sample_id"]: "support" if i%2 else "no_direct" for i,r in enumerate(rows)}
        p = fit_profile(rows, outcomes, "Joint", 1.0)
        self.assertEqual(p["global_d"], 0.5)
        self.assertEqual(p["composition"], [0.75,1/12,1/12,1/12])
        self.assertAlmostEqual(1-predict(p,rows[1])[-1],0.75)
        self.assertAlmostEqual(1-predict(p,rows[0])[-1],0.25)
        unseen={**rows[0],"setting_id":"unseen"}
        self.assertAlmostEqual(1-predict(p,unseen)[-1],0.5)
        p["rho"]="inf"
        self.assertEqual(predict(p,rows[0]),predict(p,rows[1]))

    def test_all_direct_and_all_no_direct_are_positive_normalized(self):
        rows,_=self.fixture()
        for category in ("support","no_direct"):
            p=fit_profile(rows,{r['sample_id']:category for r in rows},'Joint',0.25)
            for r in rows:
                probabilities=predict(p,r)
                self.assertTrue(all(0<x<1 for x in probabilities))
                self.assertAlmostEqual(math.fsum(probabilities),1)

    def test_fold_leakage_and_label_coverage_are_rejected(self):
        rows,labels=self.fixture()
        bad=copy.deepcopy(rows);bad[1]['fold']=99
        with self.assertRaisesRegex(ValueError,'Question crosses'):validate_development(bad,labels)
        labels['observer_a'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):validate_development(rows,labels)

    def test_fit_uses_training_only_and_observers_remain_separate(self):
        rows,labels=self.fixture()
        outcomes=validate_development(rows,labels)
        train=[r for r in rows if r['fold']!=0]
        before=fit_profile(train,outcomes['observer_a'],'Global','inf')
        outcomes['observer_a']['s0']='support';outcomes['observer_a']['s1']='reject'
        self.assertEqual(before,fit_profile(train,outcomes['observer_a'],'Global','inf'))
        labels['observer_b']=[label(r['sample_id'],'support') for r in rows]
        fit=select_and_fit(rows,labels)
        self.assertNotEqual(fit['profiles']['Global']['observer_a']['global_d'],fit['profiles']['Global']['observer_b']['global_d'])

    def test_exact_ties_choose_infinite_shrinkage_and_targets_cannot_overlap(self):
        rows=[{'sample_id':f's{i}','question_id':f'q{i//4}','fold':i//4,
               'setting_id':f'setting{(i//2)%2}','evidence_stratum':'limited'} for i in range(16)]
        labels={o:[label(r['sample_id'],'support' if i%2 else 'no_direct') for i,r in enumerate(rows)]
                for o in ('observer_a','observer_b')}
        fit=select_and_fit(rows,labels)
        self.assertTrue(all(v=='inf' for v in fit['selection'].values()))
        with self.assertRaisesRegex(ValueError,'overlaps'):forecast_targets(fit,rows,rows)
        target=[{**rows[0],'sample_id':'new','question_id':'newq'}]
        forecasts=forecast_targets(fit,target,rows)
        self.assertEqual(len(forecasts),8)
        with self.assertRaisesRegex(ValueError,'question ID overlaps'):
            forecast_targets(fit,[{**rows[0],'sample_id':'new'}],rows)

    def test_pooled_composition_cancels_in_categorical_nll_contrasts(self):
        rows,labels=self.fixture();outcomes=validate_development(rows,labels)['observer_a']
        p=fit_profile(rows,outcomes,'Joint',1.0);g=fit_profile(rows,outcomes,'Global','inf')
        for category_index in range(5):
            a,b=predict(p,rows[1]),predict(g,rows[1])
            category_delta=-math.log(a[category_index])+math.log(b[category_index])
            gate_delta=(-math.log(a[-1])+math.log(b[-1])) if category_index==4 else (-math.log1p(-a[-1])+math.log1p(-b[-1]))
            self.assertAlmostEqual(category_delta,gate_delta)


if __name__=='__main__':unittest.main()
