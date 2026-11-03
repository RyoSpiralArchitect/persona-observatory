"""Hand-calculated gate effects on a synthetic complete 96-question design."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from persona_observatory.evaluation import score_rows
from persona_observatory.storage import sha_file, write_json_exclusive, write_jsonl_exclusive
import test_predictors as fixtures

spec=importlib.util.spec_from_file_location('gate_summary_script',Path(__file__).resolve().parents[1]/'scripts/summarize_gate_study.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class GateSummaryTests(unittest.TestCase):
    def run_fixture(self,joint_probability):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            plan=[{'sample_id':f's{i}','question_id':f'q{i//12}','setting_id':f'c{(i%12)//2}',
                   'evidence_stratum':f'e{i//288}'} for i in range(1152)]
            labels={o:[fixtures.label(r['sample_id'],'support') for r in plan] for o in ('observer_a','observer_b')}
            probabilities={'Global':0.4,'Evidence':0.5,'Setting':0.6,'Joint':joint_probability}
            forecasts=[{'sample_id':r['sample_id'],'observer':o,'predictor':a,'category_probabilities':[p/4]*4+[1-p]}
                       for r in plan for o in labels for a,p in probabilities.items()]
            write_jsonl_exclusive(root/'plan.jsonl',plan)
            write_jsonl_exclusive(root/'forecasts.jsonl',forecasts)
            write_json_exclusive(root/'scores.json',score_rows(forecasts,labels))
            for f in ('run_lock.json','completion.json','forecast_lock.json'):write_json_exclusive(root/f,{'fixture':True})
            observation_hashes={}
            for o in labels:
                write_jsonl_exclusive(root/'observers'/o/'labels.jsonl',labels[o])
                write_json_exclusive(root/'observers'/o/'observation_lock.json',{'fixture':True})
                observation_hashes[o]=sha_file(root/'observers'/o/'observation_lock.json')
            write_json_exclusive(root/'score_lock.json',{'scores_sha256':sha_file(root/'scores.json'),
                'run_lock_sha256':sha_file(root/'run_lock.json'),'completion_sha256':sha_file(root/'completion.json'),
                'forecast_lock_sha256':sha_file(root/'forecast_lock.json'),'observation_locks':observation_hashes})
            # Run/label lock validation is covered by existing lifecycle tests;
            # this fixture isolates full-score recomputation and the declared decision rule.
            with patch.object(summary,'verify_run'),patch.object(summary,'validate_observer'):
                return summary.summarize(root,root/'summary.json')

    def test_known_improvement_reports_all_primary_and_secondary_results(self):
        result=self.run_fixture(0.7)
        self.assertTrue(result['primary_descriptive_support'])
        self.assertTrue(result['advance'])
        self.assertEqual(len(result['contrasts']),5)
        self.assertAlmostEqual(result['contrasts'][0]['ab_mean']['gate_brier'],-0.16)
        self.assertEqual(result['primary_question_signs'],{'negative':96,'tie':0,'positive':0})
        self.assertEqual(result['observer_d_disagreements'],0)
        self.assertAlmostEqual(result['primary_leave_one_question_out_range'][0],-0.16)

    def test_secondary_setting_gain_does_not_rescue_failed_primary(self):
        result=self.run_fixture(0.45)
        self.assertFalse(result['primary_descriptive_support'])
        self.assertFalse(result['advance'])
        self.assertAlmostEqual(result['contrasts'][0]['ab_mean']['gate_brier'],0.0525)
        self.assertLess(result['contrasts'][1]['ab_mean']['gate_brier'],0)


if __name__=='__main__':unittest.main()
