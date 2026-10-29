"""Below-threshold review list: the highest-scoring devices that did not alarm, for a person to judge.

The alarm threshold stays as frozen. Devices that did not alarm are ranked by their highest
score in the period; the top K form a review list. K comes from how many devices people can
review per period (a budget), never from labels. With labels present, the list is also scored:
a fault counts as caught by review when its in-window (timely) score reaches the list's cut-off.

    python tools/review_list.py --devices results/typeb_v3.2/historical_test/trial_001/devices.csv \
        --fraction 0.005 --out review_list.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd

# Review budgets reported side by side so no single budget is picked after seeing results.
BUDGETS = (0.0025, 0.005, 0.01, 0.02)


def review(table, count):
    """(review list, metrics) for one device table and a review budget of `count` devices."""
    e = table.loc[table.status.eq('eligible')]
    pool = e.loc[~e.alarm.astype(bool) & e.max_score.notna()].sort_values('max_score', ascending=False, kind='stable')
    listed = pool.head(count).copy()
    listed.insert(0, 'review_rank', range(1, len(listed)+1))
    metrics = dict(review_budget=count, review_devices=len(listed),
                   review_cutoff=float(listed.max_score.min()) if len(listed) else None, alarmed_devices=int(e.alarm.sum()))
    if 'positive' in e and 'hit' in e:
        cutoff = metrics['review_cutoff'] if metrics['review_cutoff'] is not None else float('inf')
        caught = listed.positive.astype(bool) & (listed.max_hit_score >= cutoff)
        faults, hits = int(e.positive.sum()), int(e.hit.sum())
        metrics.update(fault_devices=faults, alarm_hits=hits, review_faults=int(caught.sum()),
                       review_precision=int(caught.sum())/len(listed) if len(listed) else None,
                       recall_alarm=hits/faults if faults else None,
                       recall_alarm_and_review=(hits+int(caught.sum()))/faults if faults else None)
        listed['caught_in_time'] = caught
    return listed, metrics


def budgets(path, fractions=BUDGETS):
    table = pd.read_csv(path)
    eligible = int(table.status.eq('eligible').sum())
    return {f'{f:.2%}': dict(fraction=f, **review(table, max(1, round(eligible*f)))[1]) for f in fractions}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--devices', required=True, help='devices*.csv from confirm_search, TEST replay or a deployment run')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--fraction', type=float, help='review budget as a share of eligible devices, e.g. 0.005')
    g.add_argument('--count', type=int, help='review budget as a number of devices')
    p.add_argument('--out', help='write the review list as CSV')
    a = p.parse_args()
    table = pd.read_csv(a.devices)
    count = a.count if a.count is not None else max(1, round(int(table.status.eq('eligible').sum())*a.fraction))
    listed, metrics = review(table, count)
    if a.out:
        keep = [c for c in ('review_rank', 'sn_name', 'device', 'max_score', 'positive', 'max_hit_score', 'caught_in_time') if c in listed]
        listed[keep].to_csv(a.out, index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
