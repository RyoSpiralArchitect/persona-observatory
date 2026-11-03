import math
import unittest

from persona_observatory.component_predictors import CANDIDATES, fit_components, predict_components
from persona_observatory.predictors import fit_profile, predict


class ComponentPredictorTests(unittest.TestCase):
    def fixture(self):
        rows = [{'sample_id': f's{i}', 'setting_id': f'setting{i%2}',
                 'evidence_stratum': f'evidence{(i//2)%2}'} for i in range(12)]
        outcomes = {r['sample_id']: ('support' if i % 3 else 'no_direct') for i,r in enumerate(rows)}
        return rows, outcomes

    def test_zero_weights_are_exactly_setting_and_full_weights_are_pooled_cell(self):
        rows, outcomes = self.fixture()
        profile = fit_components(rows, outcomes)
        setting = fit_profile(rows, outcomes, 'Setting', 1.0)
        for row in rows:
            self.assertEqual(predict_components(profile,row,evidence_weight=0,interaction_weight=0), predict(setting,row))
            cell = next(c for c in profile['cells'] if c['setting_id']==row['setting_id'] and c['evidence_stratum']==row['evidence_stratum'])
            p = predict_components(profile,row,evidence_weight=1,interaction_weight=1)
            self.assertAlmostEqual(1-p[-1], cell['gate'])

    def test_extreme_counts_stay_positive_normalized_for_every_candidate(self):
        rows, outcomes = self.fixture()
        for category in ('support', 'no_direct', None):
            values = outcomes if category is None else {s:category for s in outcomes}
            profile = fit_components(rows,values)
            for e,i in CANDIDATES:
                for row in rows:
                    p = predict_components(profile,row,evidence_weight=e,interaction_weight=i)
                    self.assertTrue(all(0 < x < 1 for x in p))
                    self.assertAlmostEqual(math.fsum(p),1)

    def test_unseen_cell_is_additive_and_unseen_factors_fall_back(self):
        rows, outcomes = self.fixture(); profile=fit_components(rows,outcomes)
        unseen={**rows[0], 'setting_id':'new'}
        a=predict_components(profile,unseen,evidence_weight=1,interaction_weight=0)
        b=predict_components(profile,unseen,evidence_weight=1,interaction_weight=1)
        self.assertEqual(a,b)
        unseen['evidence_stratum']='new'
        self.assertAlmostEqual(1-predict_components(profile,unseen,evidence_weight=1,interaction_weight=1)[-1],profile['global_d'])

    def test_heldout_outcomes_do_not_enter_training_counts(self):
        rows,outcomes=self.fixture(); train=rows[:6]
        before=fit_components(train,outcomes)
        for row in rows[6:]: outcomes[row['sample_id']]='reject'
        self.assertEqual(before,fit_components(train,outcomes))

    def test_invalid_priors_weights_and_sample_duplicates_are_rejected(self):
        rows,outcomes=self.fixture()
        for prior in (0,-1,float('inf'),True):
            with self.assertRaises(ValueError):fit_components(rows,outcomes,cell_prior=prior)
        profile=fit_components(rows,outcomes)
        for e,i in [(0,1),(float('nan'),0),(1,True),(-1,0),(1,2)]:
            with self.assertRaises(ValueError):predict_components(profile,rows[0],evidence_weight=e,interaction_weight=i)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            fit_components(rows+[rows[0]],outcomes)


if __name__ == '__main__':
    unittest.main()
