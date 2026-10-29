"""Pre-training feature proposals: the feature agent suggests, people adopt and implement.

Runs once per dataset BEFORE any training. The agent sees raw-source column names and a few
TRAIN-period sample rows plus the features already available; it never sees labels, failure
times or results. It returns implementable feature proposals with mechanism and basis. People
record a decision for every proposal in adoption.json; `review` renders that into review.md.

    python tools/propose_features.py propose --raw /data/raw_ce_logs --time-column LogTime \
        --before 1704067200 --current-features <session>/mapping.json --task-file task.txt --out proposals/typeb
    # edit proposals/typeb/adoption.json: decision, reviewer, reason, implemented_as
    python tools/propose_features.py review --out proposals/typeb

It lives outside the repository root, so adding it does not change any session's code seal.
"""
import argparse
import io
import json
import os
import socket
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from providers import parse_reply  # strict JSON: duplicate keys and NaN are rejected

PROMPT = Path(__file__).resolve().parent/'prompts'/'feature_proposal.txt'
TABLES = ('.csv', '.feather', '.parquet')
EFFECTS = {'reduce_false_alarms', 'recover_missed_faults', 'both', 'coverage'}
LEVELS = {'high', 'medium', 'low'}
DECISIONS = {'pending', 'adopted', 'rejected', 'deferred'}
FIELDS = ('id', 'name', 'source_fields', 'computation', 'time_window', 'causality', 'mechanism',
          'basis', 'basis_confidence', 'expected_effect', 'priority', 'difficulty')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_table(handle, name, rows):
    import pandas as pd
    if name.endswith('.csv'):
        return pd.read_csv(handle, nrows=rows*50)
    if name.endswith('.feather'):
        return pd.read_feather(handle)
    return pd.read_parquet(handle)


def sources(raw):
    """(display name, opener) for every table under a directory, a single file, or a zip."""
    raw = Path(raw)
    if raw.suffix == '.zip':
        z = zipfile.ZipFile(raw)
        return [(n, lambda n=n: io.BytesIO(z.read(n))) for n in sorted(z.namelist()) if n.lower().endswith(TABLES)]
    files = [raw] if raw.is_file() else sorted(p for p in raw.rglob('*') if p.suffix.lower() in TABLES)
    return [(str(p.relative_to(raw.parent if raw.is_file() else raw)), lambda p=p: p.open('rb')) for p in files]


def profile(raw, rows=20, max_files=5, time_column=None, before=None):
    """Column profile from a few files. With --before, only rows earlier than that time are shown."""
    found = sources(raw)
    if not found:
        raise SystemExit('No .csv/.feather/.parquet files found under '+str(raw))
    picked = found[:max_files]
    out = []
    for name, opener in picked:
        with opener() as handle:
            frame = read_table(handle, name.lower(), rows)
        if before is not None:
            if time_column not in frame:
                raise SystemExit(f'{name}: time column {time_column!r} missing; cannot restrict samples to TRAIN')
            frame = frame.loc[frame[time_column] < before]
        frame = frame.head(rows)
        columns = []
        for col in frame.columns:
            values = frame[col].dropna()
            examples = [str(v)[:80] for v in values.drop_duplicates().head(3).tolist()]
            columns.append(dict(name=str(col), dtype=str(frame[col].dtype), non_null_in_sample=int(values.size), examples=examples))
        out.append(dict(file=name, sample_rows=len(frame), columns=columns))
    return dict(files_total=len(found), files_profiled=[x['file'] for x in out], sources=out,
                sample_restricted_before=before, time_column=time_column)


def current_features(path):
    """Feature names from a mapping.json, a trial result.json, a JSON list, or one name per line."""
    if not path:
        return []
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix != '.json':
        return [x.strip() for x in text.splitlines() if x.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return [str(x) for x in value]
    if 'observations' in value:
        return [f['name'] for o in value['observations'] for f in o['features']]
    if 'features' in value:
        return list(value['features'])
    raise SystemExit('Unrecognized --current-features file')


def check(reply, columns):
    if not isinstance(reply.get('data_understanding'), str) or not reply['data_understanding'].strip():
        raise ValueError('data_understanding must be a nonempty string')
    props = reply.get('proposals')
    if not isinstance(props, list) or not 1 <= len(props) <= 30:
        raise ValueError('proposals must be a list of 1..30 items')
    ids, names = set(), set()
    for p in props:
        missing = [k for k in FIELDS if k not in p]
        if missing:
            raise ValueError(f'proposal {p.get("id")} misses {missing}')
        if p['id'] in ids or p['name'] in names:
            raise ValueError('duplicate proposal id or name: '+str(p['id']))
        ids.add(p['id']); names.add(p['name'])
        unknown = [f for f in p['source_fields'] if f not in columns]
        if not p['source_fields'] or unknown:
            raise ValueError(f'proposal {p["id"]} uses columns not in raw_sources: {unknown or "none given"}; '
                             'move information the data lacks to data_requests')
        if p['expected_effect'] not in EFFECTS or p['basis_confidence'] not in LEVELS or p['difficulty'] not in LEVELS:
            raise ValueError(f'proposal {p["id"]} has an invalid expected_effect, basis_confidence or difficulty')
        if p['priority'] not in (1, 2, 3):
            raise ValueError(f'proposal {p["id"]} priority must be 1, 2 or 3')
    for k in ('data_requests', 'uncertainties'):
        if not isinstance(reply.get(k, []), list):
            raise ValueError(k+' must be a list')
    return reply


def deepseek(body, base_url, timeout=600):
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise SystemExit('DEEPSEEK_API_KEY missing')
    req = urllib.request.Request(base_url.rstrip('/')+'/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.load(res)
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise SystemExit(f'LLM HTTP {e.code}') from None
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            if attempt == 2:
                raise SystemExit('LLM network failed') from None
        time.sleep(2**attempt*2)


def propose(raw, out, task, current=(), rows=20, max_files=5, time_column=None, before=None,
            model='deepseek-v4-pro', base_url='https://api.deepseek.com', max_tokens=32768, repairs=2,
            dry_run=False, send=deepseek):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'proposals.json').exists():
        raise SystemExit(f'{out} already holds proposals; use a new folder so earlier decisions stay intact')
    prof = profile(raw, rows, max_files, time_column, before)
    payload = dict(task='feature_proposal', task_context=task, current_features=list(current), raw_sources=prof['sources'])
    save(out/'input_profile.json', dict(prof, current_features=list(current), task_context=task))
    columns = {c['name'] for s in prof['sources'] for c in s['columns']}
    messages = [dict(role='system', content=PROMPT.read_text(encoding='utf-8')),
                dict(role='user', content=json.dumps(payload, ensure_ascii=False))]
    body = dict(model=model, messages=messages, response_format={'type': 'json_object'}, max_tokens=max_tokens)
    save(out/'request_without_credentials.json', body)
    if dry_run:
        return None
    usage, error = [], None
    for attempt in range(repairs+1):
        answer = send(body, base_url)
        usage.append(answer.get('usage', {}))
        text = answer['choices'][0]['message'].get('content') or ''
        (out/f'reply_{attempt+1}.txt').write_text(text, encoding='utf-8')
        try:
            reply = check(parse_reply(text), columns)
            break
        except ValueError as e:
            error = str(e)
            body = dict(body, messages=messages+[dict(role='assistant', content=text),
                        dict(role='user', content='Repair this response only. Problem: '+error)])
    else:
        save(out/'usage.json', usage)
        raise SystemExit('Proposal response still invalid after repair: '+str(error))
    save(out/'usage.json', usage)
    save(out/'proposals.json', dict(reply, model=model, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                    stage='before_any_training', labels_shown=False, results_shown=False))
    save(out/'adoption.json', [dict(id=p['id'], name=p['name'], decision='pending', reviewer='', reason='',
                                    implemented_as=[], implementation='') for p in reply['proposals']])
    (out/'proposals.md').write_text(render(reply), encoding='utf-8')
    return reply


def render(reply, decisions=None):
    decisions = {d['id']: d for d in decisions or []}
    lines = ['# 特征 Agent 的建议（训练前）', '', reply['data_understanding'], '',
             '| ID | 特征 | 来源字段 | 窗口 | 预期作用 | 优先级 | 难度 | 依据可信度 | 决定 |',
             '|---|---|---|---|---|---|---|---|---|']
    for p in reply['proposals']:
        d = decisions.get(p['id'], {}).get('decision', 'pending')
        lines.append(f'| {p["id"]} | `{p["name"]}` | {", ".join(p["source_fields"])} | {p["time_window"]} | '
                     f'{p["expected_effect"]} | {p["priority"]} | {p["difficulty"]} | {p["basis_confidence"]} | {d} |')
    for p in reply['proposals']:
        lines += ['', f'## {p["id"]} `{p["name"]}`', '', f'- **计算**：{p["computation"]}', f'- **因果性**：{p["causality"]}',
                  f'- **机理**：{p["mechanism"]}', f'- **依据**（{p["basis_confidence"]}）：{p["basis"]}']
        d = decisions.get(p['id'])
        if d and d['decision'] != 'pending':
            lines.append(f'- **人工决定**：{d["decision"]}（{d["reviewer"]}）：{d["reason"]}')
            if d['implemented_as']:
                lines.append(f'- **实现为**：{", ".join(d["implemented_as"])}（{d["implementation"]}）')
    if reply.get('data_requests'):
        lines += ['', '## 数据中缺少、Agent 希望补充的信息', ''] + [f'- {r.get("what")}：{r.get("why")}' for r in reply['data_requests']]
    if reply.get('uncertainties'):
        lines += ['', '## Agent 说明的不确定之处', ''] + [f'- {u}' for u in reply['uncertainties']]
    return '\n'.join(lines)+'\n'


def review(out):
    out = Path(out)
    reply = json.loads((out/'proposals.json').read_text(encoding='utf-8'))
    decisions = json.loads((out/'adoption.json').read_text(encoding='utf-8'))
    known = {p['id'] for p in reply['proposals']}
    if {d['id'] for d in decisions} != known:
        raise SystemExit('adoption.json must hold exactly one decision per proposal id')
    for d in decisions:
        if d['decision'] not in DECISIONS:
            raise SystemExit(f'{d["id"]}: decision must be one of {sorted(DECISIONS)}')
        if d['decision'] != 'pending' and not (d['reviewer'].strip() and d['reason'].strip()):
            raise SystemExit(f'{d["id"]}: a decision needs a reviewer and a reason')
        if d['implemented_as'] and d['decision'] != 'adopted':
            raise SystemExit(f'{d["id"]}: only adopted proposals can list implemented features')
    counts = {k: sum(d['decision'] == k for d in decisions) for k in sorted(DECISIONS)}
    implemented = sum(bool(d['implemented_as']) for d in decisions)
    summary = dict(proposals=len(decisions), decisions=counts, adopted_and_implemented=implemented,
                   implemented_features=sorted({f for d in decisions for f in d['implemented_as']}))
    save(out/'review_summary.json', summary)
    (out/'review.md').write_text(render(reply, decisions)+'\n## 汇总\n\n'+'\n'.join(
        [f'- 建议 {summary["proposals"]} 条：' + '，'.join(f'{k} {v}' for k, v in counts.items()),
         f'- 已采纳并实现 {implemented} 条，对应 {len(summary["implemented_features"])} 个特征']) + '\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('propose', help='ask the feature agent for proposals (before any training)')
    a.add_argument('--raw', required=True, help='raw data: a directory, a single file, or a .zip of csv/feather/parquet')
    a.add_argument('--out', required=True, help='new folder for this proposal round')
    a.add_argument('--task-file', required=True, help='text describing the device, prediction task and evaluation')
    a.add_argument('--current-features', help='mapping.json, result.json, JSON list, or one name per line')
    a.add_argument('--time-column'); a.add_argument('--before', type=float, help='only show sample rows earlier than this time (TRAIN end)')
    a.add_argument('--rows', type=int, default=20); a.add_argument('--max-files', type=int, default=5)
    a.add_argument('--model', default='deepseek-v4-pro'); a.add_argument('--base-url', default='https://api.deepseek.com')
    a.add_argument('--dry-run', action='store_true', help='write the profile and request only; no API call')
    r = sub.add_parser('review', help='validate adoption.json and write review.md')
    r.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'review':
        print(json.dumps(review(args.out), ensure_ascii=False, indent=2))
        return
    if args.before is not None and not args.time_column:
        raise SystemExit('--before needs --time-column')
    reply = propose(args.raw, args.out, Path(args.task_file).read_text(encoding='utf-8'), current_features(args.current_features),
                    args.rows, args.max_files, args.time_column, args.before, args.model, args.base_url, dry_run=args.dry_run)
    print('DRY RUN: request written to', Path(args.out)/'request_without_credentials.json' if reply is None
          else f'{len(reply["proposals"])} proposals written to {Path(args.out)/"proposals.md"}; record decisions in adoption.json')


if __name__ == '__main__':
    main()
