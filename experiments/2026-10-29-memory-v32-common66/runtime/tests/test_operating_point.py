"""CV-probability acceptance and out-of-fold operating points, with real CPU training and CV."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from common import read
from settings import validate_config
from agent import Loop
from confirm_search import confirm
from operating_point import choose, load
from tests.test_v2_loop import Mock, fixture

# TypeA trial_011 against its baseline: a likely gain whose 95% interval still includes zero.
TRIAL_011 = dict(delta_f1=0.0246, interval_95=[-0.0084, 0.056], probability_better=0.941, new_hits=13, lost_hits=23,
                 new_false_alarms=118, removed_false_alarms=359, fault_units=272)


class OperatingPoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='op_')
        cls.session, _ = fixture(Path(cls.tmp.name))
        cls.loop = Loop(cls.session, Mock())
        cls.loop.run()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def decide(self, comparison, **config):
        loop = self.loop
        saved = copy.deepcopy(loop.c)
        loop.c.update(dict(accept_lower_bound=0.0), **config)
        challenger = dict(next(t for t in loop.s['trials'][1:] if t['status'] == 'complete'))
        challenger['metrics'] = dict(loop.trial(loop.s['best'])['metrics'])
        try:
            with mock.patch.object(loop, 'cross_validate', return_value=Path('unused')), \
                 mock.patch('crossval.paired_cv', return_value=dict(comparison)):
                return loop.accept(challenger)
        finally:
            loop.c = saved

    def test_probability_rule_accepts_likely_gain_with_lost_hits(self):
        self.assertTrue(self.decide(TRIAL_011)['accepted'])
        self.assertFalse(self.decide(TRIAL_011, accept_test='interval_95')['accepted'])
        self.assertFalse(self.decide(TRIAL_011, accept_require_net_hits=True)['accepted'])
        self.assertFalse(self.decide(dict(TRIAL_011, probability_better=0.85))['accepted'])
        self.assertFalse(self.decide(dict(TRIAL_011, delta_f1=0.0))['accepted'])

    def test_config_rejects_unknown_rules(self):
        for bad in (dict(accept_test='median'), dict(accept_probability=1.0), dict(operating_point='test')):
            with self.assertRaises(ValueError):
                validate_config(dict(self.loop.c, **bad))

    def test_cv_point_uses_full_model_at_cv_step(self):
        folder = self.loop.folder(self.loop.s['trials'][0])
        self.assertTrue((folder / 'model_full.ubj').exists())
        op = choose(folder)
        oof = read(folder / 'cv/oof_complete.json')
        self.assertEqual(op['source'], 'cv_oof')
        self.assertEqual((op['step'], op['threshold']), (oof['best_step'], oof['cv_metrics']['threshold']))
        self.assertEqual(load(folder, op).num_boosted_rounds(), op['step'])
        old = choose(folder, 'selection_window')
        self.assertEqual(old['threshold'], read(folder / 'result.json')['metrics']['threshold'])

    def test_confirmation_scores_both_points(self):
        summary = Path(self.tmp.name) / 'confirm'
        confirm(self.session, summary)
        reports = read(summary / 'summary.json')['reports']
        baseline = next(r for r in reports if r['trial'] == self.loop.s['trials'][0]['id'])
        self.assertEqual(set(baseline['operating_points']), {'deployed', 'selection_window'})
        self.assertEqual(baseline['operating_points']['deployed']['source'], 'cv_oof')
        self.assertEqual(baseline['metrics'], baseline['operating_points']['deployed']['metrics'])
        self.assertFalse(read(summary / 'summary.json')['test_opened'])


if __name__ == '__main__':
    unittest.main()
