"""tools/propose_features.py with a scripted LLM: profile, validation/repair, adoption review."""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('propose_features', Path(__file__).resolve().parents[1]/'tools'/'propose_features.py')
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def proposal(i, fields):
    return dict(id=f'P{i:02d}', name=f'feat_{i}', source_fields=fields, computation='count per device', time_window='24h',
                causality='trailing window ending at prediction time', mechanism='m', basis='b', basis_confidence='low',
                expected_effect='both', priority=1, difficulty='low')


class Script:
    """Returns the queued replies in order and records every request body."""
    def __init__(self, *replies):
        self.replies = list(replies); self.bodies = []

    def __call__(self, body, base_url):
        self.bodies.append(body)
        return dict(choices=[dict(message=dict(content=json.dumps(self.replies.pop(0))))], usage=dict(total_tokens=7))


class ProposeFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='propose_')
        self.root = Path(self.tmp.name)
        raw = self.root/'raw'; raw.mkdir()
        pd.DataFrame(dict(LogTime=[100, 200, 300, 900], deviceID=[1, 1, 2, 2], RowId=[5, 6, 7, 8],
                          BitMask=['0x1', '0x3', '0x8', '0xF'])).to_csv(raw/'dev_1.csv', index=False)
        self.raw = raw
        self.good = dict(data_understanding='CE event log', proposals=[proposal(1, ['RowId']), proposal(2, ['BitMask', 'LogTime'])],
                         data_requests=[dict(what='DIMM model', why='vendor effects')], uncertainties=['BitMask encoding'])

    def tearDown(self):
        self.tmp.cleanup()

    def test_proposals_are_written_without_labels_or_late_rows(self):
        send = Script(self.good)
        out = self.root/'round1'
        reply = pf.propose(self.raw, out, 'memory, 15m-7d ahead', ['ce_count_15m'], time_column='LogTime', before=500, send=send)
        self.assertEqual(len(reply['proposals']), 2)
        sent = json.loads(send.bodies[0]['messages'][1]['content'])
        self.assertEqual(sent['current_features'], ['ce_count_15m'])
        self.assertEqual(sent['raw_sources'][0]['sample_rows'], 3)   # the row at LogTime=900 is withheld
        self.assertNotIn('900', json.dumps(sent['raw_sources']))
        saved = json.loads((out/'proposals.json').read_text(encoding='utf-8'))
        self.assertEqual((saved['stage'], saved['labels_shown'], saved['results_shown']), ('before_any_training', False, False))
        self.assertTrue(all(d['decision'] == 'pending' for d in json.loads((out/'adoption.json').read_text(encoding='utf-8'))))
        with self.assertRaises(SystemExit):   # a second round must not overwrite recorded decisions
            pf.propose(self.raw, out, 'task', send=Script(self.good))

    def test_unknown_column_is_repaired(self):
        bad = dict(self.good, proposals=[proposal(1, ['DimmTemperature'])])
        send = Script(bad, self.good)
        pf.propose(self.raw, self.root/'round2', 'task', send=send)
        self.assertEqual(len(send.bodies), 2)
        self.assertIn('DimmTemperature', send.bodies[1]['messages'][-1]['content'])
        with self.assertRaises(SystemExit):
            pf.propose(self.raw, self.root/'round3', 'task', send=Script(bad, bad, bad))

    def test_zip_source_and_dry_run(self):
        z = self.root/'raw.zip'
        with zipfile.ZipFile(z, 'w') as f:
            f.write(self.raw/'dev_1.csv', 'type_B/dev_1.csv')
        self.assertIsNone(pf.propose(z, self.root/'dry', 'task', dry_run=True, send=None))
        prof = json.loads((self.root/'dry'/'input_profile.json').read_text(encoding='utf-8'))
        self.assertEqual(prof['files_profiled'], ['type_B/dev_1.csv'])
        self.assertFalse((self.root/'dry'/'proposals.json').exists())

    def test_review_requires_reasons_and_counts_adoptions(self):
        out = self.root/'round4'
        pf.propose(self.raw, out, 'task', send=Script(self.good))
        decisions = json.loads((out/'adoption.json').read_text(encoding='utf-8'))
        decisions[0].update(decision='adopted', reviewer='lesie1', reason='known DIMM row fault signal',
                            implemented_as=['raw_distinct_rows_24h'], implementation='raw_cache.py')
        decisions[1].update(decision='rejected', reviewer='')
        (out/'adoption.json').write_text(json.dumps(decisions), encoding='utf-8')
        with self.assertRaises(SystemExit):
            pf.review(out)
        decisions[1].update(reviewer='lesie1', reason='encoding unclear')
        (out/'adoption.json').write_text(json.dumps(decisions), encoding='utf-8')
        s = pf.review(out)
        self.assertEqual((s['decisions']['adopted'], s['decisions']['rejected'], s['adopted_and_implemented']), (1, 1, 1))
        self.assertIn('raw_distinct_rows_24h', (out/'review.md').read_text(encoding='utf-8'))

    def test_current_features_from_mapping(self):
        m = self.root/'mapping.json'
        m.write_text(json.dumps(dict(observations=[dict(features=[dict(name='a'), dict(name='b')])])), encoding='utf-8')
        self.assertEqual(pf.current_features(m), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
