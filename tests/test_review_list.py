"""tools/review_list.py: the review list holds the top non-alarmed devices and is scored by timely scores."""
import importlib.util
import unittest
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('review_list', Path(__file__).resolve().parents[1]/'tools'/'review_list.py')
review_list = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_list)


def table():
    rows = [dict(sn_name='alarm_hit', positive=True, alarm=True, hit=True, max_score=.99, max_hit_score=.99),
            dict(sn_name='alarm_normal', positive=False, alarm=True, hit=False, max_score=.98, max_hit_score=None),
            dict(sn_name='near_fault', positive=True, alarm=False, hit=False, max_score=.90, max_hit_score=.90),
            dict(sn_name='near_normal', positive=False, alarm=False, hit=False, max_score=.85, max_hit_score=None),
            dict(sn_name='late_only_fault', positive=True, alarm=False, hit=False, max_score=.80, max_hit_score=.10),
            dict(sn_name='low_normal', positive=False, alarm=False, hit=False, max_score=.20, max_hit_score=None),
            dict(sn_name='unscored', positive=False, alarm=False, hit=False, max_score=None, max_hit_score=None),
            dict(sn_name='unknown', positive=False, alarm=False, hit=False, max_score=.95, max_hit_score=None, status='unknown_followup')]
    return pd.DataFrame([dict(dict(status='eligible'), **r) for r in rows])


class ReviewList(unittest.TestCase):
    def test_list_skips_alarmed_unknown_and_unscored_devices(self):
        listed, m = review_list.review(table(), 3)
        self.assertEqual(listed.sn_name.tolist(), ['near_fault', 'near_normal', 'late_only_fault'])
        self.assertEqual(listed.review_rank.tolist(), [1, 2, 3])
        self.assertEqual((m['alarmed_devices'], m['alarm_hits'], m['fault_devices']), (2, 1, 3))

    def test_fault_needs_a_timely_score_at_the_cutoff(self):
        listed, m = review_list.review(table(), 3)
        # late_only_fault reached the list only through a score after its warning window
        self.assertEqual(listed.set_index('sn_name').caught_in_time.to_dict(),
                         {'near_fault': True, 'near_normal': False, 'late_only_fault': False})
        self.assertEqual(m['review_faults'], 1)
        self.assertAlmostEqual(m['recall_alarm'], 1/3)
        self.assertAlmostEqual(m['recall_alarm_and_review'], 2/3)

    def test_budget_larger_than_pool(self):
        listed, m = review_list.review(table(), 50)
        self.assertEqual(m['review_devices'], 4)
        self.assertAlmostEqual(m['review_cutoff'], .20)


if __name__ == '__main__':
    unittest.main()
