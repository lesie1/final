"""tools/session_report.py on a real (synthetic-data) loop session."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from agent import Loop
from tests.test_v2_loop import Mock, fixture

spec = importlib.util.spec_from_file_location('session_report', Path(__file__).resolve().parents[1]/'tools'/'session_report.py')
session_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_report)


class SessionReport(unittest.TestCase):
    def test_summary_matches_state(self):
        with tempfile.TemporaryDirectory(prefix='report_') as tmp:
            session, _ = fixture(Path(tmp))
            loop = Loop(session, Mock())
            loop.run()
            r = session_report.report(session, goal=0.0)
            self.assertEqual(r['best'], loop.s['best'])
            self.assertEqual(r['agent']['calls'], loop.s['calls'])
            self.assertEqual(r['agent']['total_tokens'], loop.s['tokens'])
            self.assertEqual(r['autonomy']['manual_interventions'], 0)
            self.assertTrue(r['target']['reached'])
            model = r['models'][loop.s['best']]
            self.assertEqual(model['complexity']['trees'], model['operating_point']['step'])
            self.assertIn('batch_1', model['model_only_latency'])
            self.assertIn(loop.s['best'], session_report.markdown(r))
            a = r['automation']
            self.assertEqual((a['manual_steps'], a['step_coverage'], a['trial_coverage']), (0, 1.0, 1.0))
            self.assertTrue(a['full_pipeline_automated'])
            self.assertEqual(model['complexity']['parameters'], model['complexity']['nodes'])

    def test_budget_change_counts_as_manual(self):
        events = [dict(at='2026-10-29T00:00:00Z', event='decision_accepted'),
                  dict(at='2026-10-29T00:01:00Z', event='training_started', trial='trial_001'),
                  dict(at='2026-10-29T00:02:00Z', event='experiment_complete', trial='trial_001'),
                  dict(at='2026-10-29T00:03:00Z', event='limits_changed'),
                  dict(at='2026-10-29T00:04:00Z', event='training_started', trial='trial_002'),
                  dict(at='2026-10-29T00:05:00Z', event='experiment_complete', trial='trial_002')]
        a = session_report.automation(events, 'trial_budget_complete')
        self.assertEqual((a['automatic_steps'], a['manual_steps']), (5, 1))
        self.assertAlmostEqual(a['step_coverage'], 5/6)
        self.assertEqual((a['trials_without_intervention'], a['trial_coverage']), (1, 0.5))
        self.assertFalse(a['full_pipeline_automated'])

    def test_device_accuracy_counts_each_device_once(self):
        import pandas as pd
        rows = [dict(status='eligible', positive=True, alarm=True, hit=True, fp=False, fn=False),     # timely hit
                dict(status='eligible', positive=True, alarm=True, hit=False, fp=True, fn=True),     # late alarm only
                dict(status='eligible', positive=True, alarm=False, hit=False, fp=False, fn=True),   # missed
                dict(status='eligible', positive=False, alarm=True, hit=False, fp=True, fn=False),   # false alarm
                dict(status='eligible', positive=False, alarm=False, hit=False, fp=False, fn=False), # true negative
                dict(status='eligible', positive=False, alarm=False, hit=False, fp=False, fn=False),
                dict(status='unknown_followup', positive=False, alarm=True, hit=False, fp=False, fn=False)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'devices.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            m = session_report.device_metrics(path)
        self.assertEqual((m['eligible_devices'], m['TP'], m['FP'], m['FN'], m['TN']), (6, 1, 2, 2, 2))
        self.assertAlmostEqual(m['accuracy'], 3/6)
        self.assertAlmostEqual(m['f1'], 2/(2+2+2))
        self.assertAlmostEqual(m['balanced_accuracy'], (1/3+2/3)/2)


if __name__ == '__main__':
    unittest.main()
