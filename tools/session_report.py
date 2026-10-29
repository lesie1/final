"""Competition metrics for one finished session: search, agent cost, autonomy, model size, latency.

Read-only: it opens session files and saved models and never writes into the session. It lives
outside the repository root so adding it does not change the code seal of recorded sessions.

    python tools/session_report.py --session results/typea_v3.2/session --target-f1 0.15 \
        --json results/typea_v3.2/summary.json --md results/typea_v3.2/summary.md
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

# Events written only when a person changes the run; everything else is the controller or an agent.
MANUAL_EVENTS = {'limits_changed'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp(text):
    return datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ')


def trial_rows(session, state):
    rows = []
    for t in state['trials']:
        acc = t.get('acceptance') or {}
        cv = (t.get('cv') or {}).get('metrics') or {}
        m = t.get('metrics') or {}
        rows.append(dict(trial=t['id'], status=t['status'], parent=t.get('parent'),
                         reason=(t.get('reason') or '')[:160], accepted=acc.get('accepted'), rule=acc.get('rule'),
                         probability_better=acc.get('probability_better'), cv_delta_f1=acc.get('cv_delta_f1'),
                         selection_f1=m.get('f1'), cv_f1=cv.get('f1'), cv_TP=cv.get('TP'), cv_FP=cv.get('FP'), cv_FN=cv.get('FN')))
    return rows


def agent_cost(session):
    calls = sorted((session/'calls').glob('*'))
    by_role, tokens, prompt, completion = {}, 0, 0, 0
    for folder in calls:
        role = folder.name.split('_', 1)[1]
        by_role[role] = by_role.get(role, 0)+1
        if (folder/'usage.json').exists():
            u = read(folder/'usage.json')
            tokens += int(u.get('total_tokens', 0))
            prompt += int(u.get('prompt_tokens', 0))
            completion += int(u.get('completion_tokens', 0))
    return dict(calls=len(calls), calls_by_role=by_role, total_tokens=tokens, prompt_tokens=prompt, completion_tokens=completion)


def autonomy(events):
    counts = {}
    for e in events:
        counts[e['event']] = counts.get(e['event'], 0)+1
    decisions = counts.get('decision_accepted', 0)
    fallback = counts.get('controller_fallback', 0)
    manual = sum(counts.get(k, 0) for k in MANUAL_EVENTS)
    return dict(event_counts=counts, manual_interventions=manual, agent_decisions=decisions,
                controller_fallbacks=fallback, response_repairs=counts.get('response_repair', 0),
                agent_decision_share=(decisions-fallback)/decisions if decisions else None,
                note='manual_interventions counts recorded limit changes only; session setup and restarts outside the loop are not logged as events.')


def complexity(booster):
    if hasattr(booster, 'primary'):
        return dict(primary=complexity(booster.primary), auxiliary=complexity(booster.auxiliary))
    if hasattr(booster, 'stage1'):
        return dict(stage1=complexity(booster.stage1), stage2=complexity(booster.stage2))
    trees = booster.trees_to_dataframe()
    leaves = trees.loc[trees.Feature.eq('Leaf')]

    def depth(tree):
        parent = {}
        for _, r in tree.iterrows():
            for child in (r.Yes, r.No):
                if isinstance(child, str):
                    parent[child] = r.ID
        best = 0
        for leaf in tree.loc[tree.Feature.eq('Leaf'), 'ID']:
            d, node = 0, leaf
            while node in parent:
                node, d = parent[node], d+1
            best = max(best, d)
        return best
    return dict(trees=int(trees.Tree.nunique()), nodes=len(trees), leaves=len(leaves),
                split_nodes=len(trees)-len(leaves), max_depth=int(max(depth(g) for _, g in trees.groupby('Tree'))),
                input_features=int(booster.num_features()))


def latency(model, features, repeats=30):
    """Model-only scoring time on synthetic rows; feature construction is reported separately."""
    import numpy as np
    rng = np.random.default_rng(0)
    out = {}
    for batch in (1, 1000, 10000):
        x = rng.random((batch, features), dtype=np.float32)
        model.inplace_predict(x)
        runs = []
        for _ in range(repeats if batch < 10000 else max(5, repeats//6)):
            tick = time.perf_counter()
            model.inplace_predict(x)
            runs.append(time.perf_counter()-tick)
        out[f'batch_{batch}'] = dict(median_ms=statistics.median(runs)*1000,
                                     per_row_us=statistics.median(runs)/batch*1e6)
    return out


def deployed(session, config, trial):
    from operating_point import choose, load
    folder = session/'trials'/trial/'attempt_01'
    op = choose(folder, config.get('operating_point', 'selection_window'))
    model = load(folder, op, 1)
    size = (folder/op['model']).stat().st_size
    timings = read(folder/'result.json').get('timings', {})
    return dict(trial=trial, operating_point={k: op[k] for k in ('source', 'step', 'threshold') if k in op},
                model_file=op['model'], model_file_bytes=size, complexity=complexity(model),
                model_only_latency=latency(model, model.num_features()),
                recorded_training_timings_seconds=timings)


def target(rows, events, start, goal):
    if goal is None:
        return None
    done = {e['trial']: e['at'] for e in events if e['event'] == 'experiment_complete'}
    for i, r in enumerate(rows, 1):
        if r['cv_f1'] is not None and r['cv_f1'] >= goal:
            at = done.get(r['trial'])
            return dict(target_cv_f1=goal, reached=True, trial=r['trial'], rounds=i,
                        hours=(stamp(at)-start).total_seconds()/3600 if at else None)
    return dict(target_cv_f1=goal, reached=False, rounds=len(rows))


def report(session, goal=None):
    session = Path(session)
    config, state = read(session/'config.json'), read(session/'state.json')
    events = [json.loads(line) for line in (session/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    start, end = stamp(events[0]['at']), stamp(events[-1]['at'])
    rows = trial_rows(session, state)
    baseline, best = state['trials'][0]['id'], state['best']
    return dict(session=f"{session.parent.name}/{session.name}", domain=config['domain'], status=state['status'],
                wall_clock_hours=(end-start).total_seconds()/3600,
                policy={k: config.get(k) for k in ('acceptance', 'accept_test', 'accept_probability',
                        'accept_require_net_hits', 'operating_point', 'max_trials', 'cv_folds')},
                trials=rows, completed=sum(r['status'] == 'complete' for r in rows),
                accepted=[r['trial'] for r in rows[1:] if r['accepted']], baseline=baseline, best=best,
                agent=agent_cost(session), autonomy=autonomy(events), target=target(rows, events, start, goal),
                models={name: deployed(session, config, name) for name in dict.fromkeys([baseline, best])},
                environment=dict(python=platform.python_version(), machine=platform.machine(),
                                 processor=platform.processor() or None, cpus=os.cpu_count()))


def markdown(r):
    pct = lambda v: '' if v is None else f'{v*100:.2f}%'
    lines = [f'# Session summary: {r["session"]}', '',
             f'- Domain `{r["domain"]}`, status `{r["status"]}`, wall clock {r["wall_clock_hours"]:.2f} h',
             f'- Baseline `{r["baseline"]}`, final best `{r["best"]}`, accepted {r["accepted"] or "none"}',
             f'- Agent calls {r["agent"]["calls"]} {r["agent"]["calls_by_role"]}, tokens {r["agent"]["total_tokens"]:,}',
             f'- Manual interventions {r["autonomy"]["manual_interventions"]}, agent decisions {r["autonomy"]["agent_decisions"]}, '
             f'controller fallbacks {r["autonomy"]["controller_fallbacks"]}, response repairs {r["autonomy"]["response_repairs"]}']
    if r['target']:
        lines.append(f'- Target CV F1 {pct(r["target"]["target_cv_f1"])}: ' + (
            f'reached by `{r["target"]["trial"]}` at round {r["target"]["rounds"]} ({r["target"]["hours"]:.2f} h)'
            if r['target']['reached'] else 'not reached'))
    lines += ['', '| trial | status | accepted | P(better) | selection F1 | CV F1 | CV TP/FP/FN |', '|---|---|---|---|---|---|---|']
    for t in r['trials']:
        p = '' if t['probability_better'] is None else f'{t["probability_better"]:.3f}'
        lines.append(f'| {t["trial"]} | {t["status"]} | {t["accepted"]} | {p} | {pct(t["selection_f1"])} | {pct(t["cv_f1"])} | '
                     f'{t["cv_TP"]}/{t["cv_FP"]}/{t["cv_FN"]} |')
    lines += ['', '| model | operating point | trees | nodes | max depth | features | file | 1 row | 10k rows |',
              '|---|---|---|---|---|---|---|---|---|']
    for name, m in r['models'].items():
        c, lat, op = m['complexity'], m['model_only_latency'], m['operating_point']
        if 'trees' not in c:
            c = next(iter(c.values()))
        lines.append(f'| {name} | {op["source"]} step {op["step"]}, thr {op["threshold"]:.4f} | {c["trees"]} | {c["nodes"]} | '
                     f'{c["max_depth"]} | {c["input_features"]} | {m["model_file_bytes"]/1024:.0f} KiB | '
                     f'{lat["batch_1"]["median_ms"]:.3f} ms | {lat["batch_10000"]["median_ms"]:.1f} ms |')
    lines += ['', f'Latency is model-only scoring on this machine ({r["environment"]["cpus"]} CPUs, Python {r["environment"]["python"]}); '
              'feature construction and the full validation pass are in `recorded_training_timings_seconds`.']
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--session', required=True)
    p.add_argument('--target-f1', type=float, help='preset goal on pooled CV F1; reports the first trial reaching it')
    p.add_argument('--json')
    p.add_argument('--md')
    a = p.parse_args()
    r = report(a.session, a.target_f1)
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    if a.md:
        Path(a.md).write_text(markdown(r), encoding='utf-8')
    if not (a.json or a.md):
        print(markdown(r))


if __name__ == '__main__':
    main()
