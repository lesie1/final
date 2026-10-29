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


if __name__ == '__main__':
    unittest.main()
