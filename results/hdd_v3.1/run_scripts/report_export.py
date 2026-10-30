"""Export a reviewable HDD final result without raw data, caches, or credentials."""
import collections, datetime, gzip, hashlib, html, json, re, shutil, sys, zipfile
from pathlib import Path

ROOT=Path('/root/autodl-tmp/hdd_final_test_20261029')
SOURCE=Path('/root/autodl-tmp/final_v31_20261028')
TASK=SOURCE/'hdd';SESSION=TASK/'session';EVAL=ROOT/'evaluation';OUT=ROOT/'publication'
TEXT={'.json','.jsonl','.csv','.log','.txt','.md','.html','.py','.toml','.yaml','.yml'}
PATTERNS=[re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),re.compile(rb'Bearer\s+[A-Za-z0-9._-]{24,}'),re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}')]

def read(p):return json.loads(Path(p).read_text())
def save(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(files):
    for p,h in files.items():assert sha(p)==h,'Protected file changed: '+str(p)
def scan(p):
    with Path(p).open('rb') as f:
        previous=b''
        for block in iter(lambda:f.read(1024*1024),b''):
            data=previous+block
            assert not any(rx.search(data) for rx in PATTERNS),'Credential-like content: '+str(p)
            previous=data[-512:]
def process_running(pid,script):
    p=Path('/proc')/str(pid)/'cmdline'
    try:return str(script).encode() in p.read_bytes().split(b'\0')
    except FileNotFoundError:return False
def metric_row(label,m):return f'| {label} | {m["TP"]} | {m["FP"]} | {m["FN"]} | {m["precision"]:.2%} | {m["recall"]:.2%} | {m["f1"]:.2%} |'

def main():
    assert read(ROOT/'test_watch/status.json')['phase']=='complete','TEST watcher incomplete'
    attempt=read(ROOT/'test_watch/attempt.json');assert not process_running(attempt['pid'],ROOT/'evaluate_hdd.py')
    assert not process_running(read(ROOT/'test_watch_launcher.json')['pid'],ROOT/'await_final_test.py')
    assert not process_running(670219,ROOT/'prepare_test_cache.py')
    assert read(TASK/'result_status.json')['status']=='complete'
    s=read(SESSION/'state.json');assert s['status']=='trial_budget_complete' and len(s['trials'])==13 and s['best']=='trial_004'
    assert (s['calls'],s['tokens'])==(50,989281)
    frozen=read(EVAL/'frozen.json');report=read(EVAL/'report_test.json');m=report['metrics']
    assert report['trial']==s['best']==frozen['trial']
    assert report['model_sha256']==frozen['model_sha256'] and m['threshold']==frozen['threshold']
    assert not report['threshold_reselected'] and not report['retrained'] and not report['agent_feedback']
    assert abs(m['f1']-2*m['TP']/(2*m['TP']+m['FP']+m['FN']))<1e-12
    assert not report['independent_blind_test']
    verify(frozen['source_seals']);verify(frozen['runtime_and_input_seals'])
    verify({EVAL/p:h for p,h in read(EVAL/'complete.json')['files'].items()})
    recovery=read(ROOT/'recovery_20261029/preservation.json');verify(recovery['files'])
    backup=ROOT/'recovery_20261029/before_recovery'
    verify({backup/p:h for p,h in recovery['backup_files'].items()})
    old_events=(backup/'hdd/session/events.jsonl').read_bytes();assert (SESSION/'events.jsonl').read_bytes().startswith(old_events)
    OUT.mkdir(exist_ok=True)
    confirmation=read(TASK/'confirmation/summary.json')['reports']
    chosen=next(t for t in s['trials'] if t['id']==s['best']);baseline=s['trials'][0]
    rows=[]
    for t in s['trials']:
        cv=(t.get('cv') or {}).get('metrics',{});a=t.get('acceptance') or {}
        rows.append(dict(trial=t['id'],status=t['status'],selected=t['id']==s['best'],accepted=a.get('accepted'),rule=a.get('rule'),
                         selection_f1=t.get('metrics',{}).get('f1'),cv_f1=cv.get('f1'),cv_TP=cv.get('TP'),cv_FP=cv.get('FP'),cv_FN=cv.get('FN'),
                         cv_delta_f1=a.get('cv_delta_f1'),cv_interval_95=a.get('cv_interval_95'),new_hits=a.get('new_hits'),lost_hits=a.get('lost_hits')))
    dates=[datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat() for t in frozen['test_window']]
    summary=dict(selected=s['best'],trials=len(s['trials']),agent_calls=s['calls'],reported_tokens=s['tokens'],test=report,
                 confirmation=confirmation,search_trials=rows,source_runtime_commit='e35150a021c52ed5605c4bb3a2c436b59113e8c0',
                 test_used_for_selection=False,fresh_blind_test_claim=False,original_event_prefix_preserved=True,source_seals_verified=len(recovery['files']))
    save(OUT/'summary.json',summary)
    import csv
    with (OUT/'search_trials.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# HDD 最终测试报告','',f'13 组搜索结束，按原 v3.1 配对 CV 规则最终选用 **{s["best"]}**。本次只对该模型执行最终 TEST，没有按 TEST 改模型、树数或阈值。',
           '',f'最终 TEST：**F1 {m["f1"]:.2%}，Precision {m["precision"]:.2%}，Recall {m["recall"]:.2%}**。',
           '', '## 最终 TEST', '| 模型 | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|',metric_row(s['best'],m),
           '',f'评价协议 `{m["protocol"]}`。可评价设备 {m["eligible_devices"]:,} 台，其中故障设备 {m["eligible_fault_devices"]:,} 台；未知随访设备 {m["unknown_devices"]:,} 台单独报告，不当作健康负例。',
           '',f'TEST 时间窗（UTC，左闭右开）：`{dates[0]}` 至 `{dates[1]}`。',
           '', '历史 TEST 是否从未使用未得到独立确认，因此不能宣称这是全新盲测。本次执行期间，TEST 不用于搜索、阈值选择或给 Agent 反馈。只测试已选模型，未额外测试基线或其他候选，因此不报告“TEST 相对基线提升”。',
           '', '## 模型与工作点',
           '', f'- 运行库：`e35150a021c52ed5605c4bb3a2c436b59113e8c0`（v3.1）；不是后来内存实验的 v3.2 工作点规则。',
           f'- 冻结模型 SHA256：`{frozen["model_sha256"]}`。',f'- 实际模型输入：{len(frozen["features"])} 列，即 {frozen["base_feature_count"]} 个基础列加 4 个已采纳的时序特征。',
           f'- 树数：{frozen["best_step"]}；阈值：`{frozen["threshold"]}`。',
           '- CV 用于候选采纳；部署模型和阈值沿用原程序的前段验证工作点，与后段验证确认完全一致。没有替换为 CV 全量重训模型或 CV 阈值。',
           '', '新增四列：`fe_187_recent_nonzero_7d_count`、`fe_187_recent_increase_7d`、`fe_187_rolling_mean_7d`、`fe_187_expanding_mean`。特征程序先按冻结定义生成完整历史视图，再由原推理代码读取 TEST。',
           '', '## CV 与后段验证确认',
           '', '各阶段的数据不同，下面的分数不能混作 TEST 分数。',
           '| 阶段／模型 | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|',
           metric_row('CV baseline '+baseline['id'],baseline['cv']['metrics']),metric_row('CV selected '+chosen['id'],chosen['cv']['metrics'])]
    for x in confirmation:lines.append(metric_row('later-VAL '+x['trial'],x['metrics']))
    last=s['trials'][-1]['acceptance']
    lines+=['',f'第 13 组未采纳：相对第 4 组，CV F1 差值为 {last["cv_delta_f1"]:+.4f}，配对 95% 区间为 {last["cv_interval_95"]}；新增命中 {last["new_hits"]}、丢失命中 {last["lost_hits"]}。最终仍保留第 4 组。',
            '', '## 全部搜索记录','| 组 | CV F1 | 本组采纳 | 采纳规则 | 最终模型 |','|---|---:|---|---|---|']
    for row in rows:
        value='未执行' if row['cv_f1'] is None else f'{row["cv_f1"]:.2%}'
        lines.append(f'| {row["trial"]} | {value} | {row["accepted"]} | {row["rule"]} | {row["selected"]} |')
    lines+=['', '## 数据与完整性',
            '', '原始 366 日文件共 105,379,761 行，基础缓存包含 330,927 个设备登记项；其中 330,902 台有可写入的观测行。这些计数是原始／缓存范围，不能当作 TEST 可评价设备数。原始文件、标签、时间切分、采样和评分定义均未修改。',
            '', '预算仍为 13 组，50 次 Agent 调用、989,281 reported tokens。缓存准备和最终测试不调用模型 API。',
            '', '中途发生两类资源调度问题：原 supervisor 的内存保护中断第 13 组 CV；之后缓存程序也因未计入可回收的活跃文件缓存而等待。保留中断前状态、事件和折模型后，复用已完成的主模型重算同一组 CV，并通过仅释放干净文件缓存恢复预处理。没有追加候选、修改评分规则或重训最终模型。',
            '', f'复核 {len(recovery["files"])} 项原运行库／输入／主模型哈希，并确认中断前事件字节序列完整保留为当前事件前缀。最终 TEST 完成回执进一步确认搜索状态、原事件、模型和冻结工作点未改变。',
            '', '## 审阅文件',
            '', '- `hdd/session/events.jsonl`：完整搜索事件；`hdd/session/state.json`：13 组状态和采纳证据。',
            '- `evaluation/report_test.json`、`evaluation/frozen.json`：最终指标和冻结依据。',
            '- `evaluation/devices_test.csv.gz`：完整设备级 TEST 明细的无损压缩副本。',
            '- `recovery_20261029/`、`resource_repair_20261029T1738.json`：中断、备份和资源修复记录。',
            '- `source_runtime/`：本次实际使用的封存运行库；仓库根目录的新版本不替代它。',
            '', '这是审阅快照，不是独立可续跑备份。原始数据、大缓存、训练矩阵、逐行分数、候选阈值扫描和 CV 折模型留在服务器；排除项及原因写入 `export_manifest.json`。原完成回执可能引用这些未导出的文件。']
    report_md='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report_md,encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>HDD 最终测试</title><style>body{font:16px/1.7 system-ui;max-width:1050px;margin:36px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><pre>'+html.escape(report_md)+'</pre>',encoding='utf-8')
    (OUT/'README.md').write_text('# HDD 最终实验\n\n[阅读结果报告](report.md) · [完整搜索事件](hdd/session/events.jsonl) · [最终测试指标](evaluation/report_test.json)\n\n使用封存 v3.1 运行库，13 组搜索，最终采用 trial_004；报告明确区分 CV、后段确认和 TEST。\n',encoding='utf-8')
    compressed={}
    for name in ['devices_test.csv','unknown_devices_test.csv']:
        source=EVAL/name;scan(source);dest=OUT/(name+'.gz')
        with source.open('rb') as src,dest.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as target:shutil.copyfileobj(src,target)
        compressed['evaluation/'+name+'.gz']=dict(source_sha256=sha(source),source_bytes=source.stat().st_size)
    included={};excluded=[];events={}
    blocked={'cache','cache_test','caches','raw_sources','matrices','feature_views','feature_samples','training_views','__pycache__','publication'}
    def reason(p,rel):
        if p==SOURCE/'runtime/docs/competition_requirements.md':return 'context_reference_document_not_required_for_inference'
        if any(x in blocked for x in rel.parts):return 'raw_or_regenerable_cache'
        if any(x.startswith('scores_') for x in rel.parts):return 'per_row_scores'
        if p.suffix in {'.parquet','.npz','.npy','.zip','.sqlite'}:return 'large_array_database_or_archive'
        if p.suffix=='.ubj' and 'cv' in rel.parts:return 'CV_fold_model_retained_remotely'
        if p.suffix=='.csv' and (p.name.startswith(('devices_','unknown_devices_','threshold_search_')) or p.name=='devices.csv'):return 'device_or_threshold_intermediate; final_TEST_has_gzip_copy'
        if p.suffix=='.lock' or p.name=='live.json':return 'transient'
        if p.suffix not in TEXT|{'.ubj'}:return 'non_review_file'
        if p.stat().st_size>20*2**20:return 'large_intermediate_retained_remotely'
        return None
    def add(z,p,rel):
        assert rel not in included and not rel.startswith('/') and '..' not in Path(rel).parts
        if p.suffix in TEXT:scan(p)
        h=sha(p);z.write(p,rel);assert sha(p)==h,'File changed during export: '+str(p)
        included[rel]=dict(sha256=h,bytes=p.stat().st_size)
        if p.name=='events.jsonl':
            records=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            events[rel]=dict(lines=len(records),counts=dict(collections.Counter(x['event'] for x in records)),sha256=h)
    with zipfile.ZipFile(OUT/'snapshot.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder,prefix in [(ROOT,''),(TASK,'hdd/'),(SOURCE/'runtime','source_runtime/')]:
            for p in sorted(folder.rglob('*')):
                if not p.is_file():continue
                rel=p.relative_to(folder);why=reason(p,rel)
                if why:excluded.append(dict(path=prefix+rel.as_posix(),bytes=p.stat().st_size,reason=why))
                else:add(z,p,prefix+rel.as_posix())
        for name in ['README.md','report.md','report.html','summary.json','search_trials.csv']:add(z,OUT/name,name)
        for rel in compressed:add(z,OUT/Path(rel).name,rel)
        add(z,ROOT/'cache_test/manifest.json','data_quality/test_cache_manifest.json')
        add(z,Path(read(SESSION/'config.json')['active_cache'])/'manifest.json','data_quality/trainval_cache_manifest.json')
        manifest=dict(files=included,excluded_files=excluded,events=events,compressed_representations=compressed,
                      original_event_prefix_preserved=True,source_seals_verified=len(recovery['files']),complete_resume_checkpoint=False,
                      test_used_for_selection=False,fresh_blind_test_claim=False)
        z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    verify(frozen['source_seals']);verify(frozen['runtime_and_input_seals']);verify(recovery['files'])
    save(OUT/'delivery.json',dict(archive=str(OUT/'snapshot.zip'),sha256=sha(OUT/'snapshot.zip'),bytes=(OUT/'snapshot.zip').stat().st_size,files=len(included),events=events))
    print(json.dumps(dict(export_ready=True,files=len(included),excluded=len(excluded),events={k:v['lines'] for k,v in events.items()},test_metrics=m),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
