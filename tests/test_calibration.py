import copy
import json
from pathlib import Path
import tempfile
import unittest

from persona_observatory.calibration import (load_packet, prepare_packet, select_cases, submit,
                                            submission_template, validate_submission)
from persona_observatory.storage import read_json, sha_file, write_json_exclusive


class CalibrationTests(unittest.TestCase):
    def packet(self, root):
        root.mkdir()
        cases = [{"case_id": f"C-{i}", "batch_id": "batch_01", "requested_answerer": "Taylor",
                  "question": "Does it work?", "completion": text, "stop_reason": "eos"}
                 for i, text in enumerate(["I cannot tell.", "Taylor says it works."])]
        write_json_exclusive(root / 'cases.json', cases)
        write_json_exclusive(root / 'packet_lock.json', {
            'schema_version': 'persona-calibration-packet-v1', 'case_count': 2,
            'file_hashes': {'cases.json': sha_file(root / 'cases.json')}})
        return cases, sha_file(root / 'packet_lock.json')

    def filled(self, packet):
        value = submission_template(packet, 'batch_01')
        value['rater'].update(actor_kind='human', read_all_visible_text=True,
                              prior_exposure=['synthetic test only'], assistance='none')
        # Invented mechanical fixture, never used to label experimental text.
        for j, quote in zip(value['judgments'], ['I cannot tell.', 'Taylor says']):
            j.update(admissible_modes=['direct', 'reported'], evidence_quotes=[quote],
                     rationale='Synthetic fixture.', uncertainty_reason='Synthetic ambiguity.')
        return value

    def test_set_valued_modes_preserve_d_uncertainty_without_semantic_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'packet'; cases, digest = self.packet(p)
            result = validate_submission(self.filled(p), cases, digest)
            self.assertEqual(result['derived'][0]['admissible_d'], [False, True])
            self.assertTrue(result['declared_human_only'])
            self.assertFalse(result['semantic_validity_established'])
            self.assertFalse(result['human_identity_verified'])

    def test_quotes_coverage_declaration_and_uncertainty_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'packet'; cases, digest = self.packet(p)
            base = self.filled(p)
            mutations = [
                lambda s: s['judgments'][0].update(evidence_quotes=['not in source']),
                lambda s: s['judgments'].pop(),
                lambda s: s['judgments'].__setitem__(1, copy.deepcopy(s['judgments'][0])),
                lambda s: s['judgments'][0].update(uncertainty_reason=''),
                lambda s: s['rater'].update(read_all_visible_text=False),
                lambda s: s['rater'].update(prior_exposure=[]),
                lambda s: s.update(packet_sha256='wrong packet'),
            ]
            for mutation in mutations:
                s = copy.deepcopy(base); mutation(s)
                with self.subTest(s=s), self.assertRaises(ValueError):
                    validate_submission(s, cases, digest)
            base['rater']['actor_kind'] = 'model'
            self.assertFalse(validate_submission(base, cases, digest)['declared_human_only'])

    def test_raw_failures_retained_and_accepted_batches_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / 'packet'; self.packet(p)
            incoming = root / 'incoming.json'; records = root / 'records'
            incoming.write_bytes(b'{broken JSON')
            bad = submit(p, incoming, records)
            self.assertEqual(bad['status'], 'REJECTED')
            self.assertEqual((Path(bad['attempt']) / 'raw_submission.json').read_bytes(), b'{broken JSON')
            incoming.write_text(json.dumps(self.filled(p)), encoding='utf-8')
            good = submit(p, incoming, records)
            self.assertEqual(good['status'], 'ACCEPTED_DECLARED_BATCH_NOT_VALIDATED')
            accepted = list((records / 'accepted').rglob('batch_01.json'))[0]
            old = accepted.read_bytes()
            repeat = submit(p, incoming, records)
            self.assertEqual(repeat['status'], 'REJECTED')
            self.assertIn('already accepted', repeat['error'])
            self.assertEqual(accepted.read_bytes(), old)
            self.assertEqual(len(list((records / 'attempts').iterdir())), 3)

    def test_packet_changes_are_detected_before_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'packet'; self.packet(p)
            with (p / 'cases.json').open('a') as f: f.write(' ')
            with self.assertRaisesRegex(ValueError, 'Changed file'):
                load_packet(p)

    def test_selection_deduplicates_and_masks_source_labels(self):
        rows, labels = [], {o: {} for o in ('observer_a', 'observer_b')}
        for i, modes in enumerate([('direct','none'), ('none','none'), ('both','direct'), ('direct','none')]):
            row = dict(sample_id=f's{i}', requested_answerer='Taylor', question='Question',
                       completion=f'Completion {i if i != 3 else 0}', setting_id='setting',
                       evidence_stratum='limited', stop_reason='eos')
            rows.append(row)
            for o,m in zip(labels,modes): labels[o][f's{i}']={'answerer_answer_mode':m}
        plan = dict(selection_seed='synthetic', disagreement_unique_cases=1,
                    agreed_controls_per_d=1, batch_size=2)
        cases, mapping = select_cases(rows, labels, plan)
        self.assertEqual(len(cases), 3)
        self.assertEqual([len(r['source_ids']) for r in mapping if r['selection_group']=='disagreement'], [2])
        self.assertEqual(set(cases[0]), {'case_id','batch_id','requested_answerer','question','completion','stop_reason'})
        self.assertEqual(select_cases(list(reversed(rows)), labels, plan), (cases, mapping))
        labels['observer_a']['s3']['answerer_answer_mode']='reported'
        with self.assertRaisesRegex(ValueError, 'inconsistent mode'):
            select_cases(rows, labels, plan)

    def test_public_source_packet_is_complete_bound_and_exclusive(self):
        root=Path(__file__).resolve().parents[1]
        archive=root/'research/experiment_10_mps_20261103/evidence_20261103.zip'
        plan=root/'research/attribution_calibration_20261103/selection_plan.json'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'packet'
            result=prepare_packet(archive,plan,output)
            self.assertEqual((result['cases'],result['batches']),(52,9))
            cases,digest=load_packet(output/'participant')
            self.assertEqual(result['packet_sha256'],digest)
            self.assertEqual(len({(c['requested_answerer'],c['question'],c['completion']) for c in cases}),52)
            self.assertEqual(len([c for c in cases if c['batch_id']=='batch_09']),4)
            lock=read_json(output/'participant/packet_lock.json')
            self.assertEqual(sha_file(output/'builder_snapshot.py'),lock['builder_sha256'])
            self.assertEqual(sha_file(output/'selection_plan.json'),lock['selection_plan_sha256'])
            self.assertEqual(submission_template(output/'participant','batch_01')['rater']['actor_kind'],'')
            with self.assertRaises(FileExistsError):prepare_packet(archive,plan,output)


if __name__ == '__main__':
    unittest.main()
