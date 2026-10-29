"""Review export only: raw data/caches and credentials never go into Git."""
import collections,html,json,re,sys,time,zipfile
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.path.insert(0,str(R));sys.dont_write_bytecode=True
from common import read,save,sha,require
from supervisor import preserved

def main():
    preserved();s=read(R/'typeb/session/state.json')
    require(s['status'] in ('stopped','trial_budget_complete'),'Search unfinished')
    require(len(s['trials'])<=13,'Trial budget exceeded')
    reports=read(R/'typeb/historical_test/summary.json')['reports']
    out=R/'publication';out.mkdir(exist_ok=True)
    for x in reports:require(x['metrics']['threshold']==x['operating_point']['threshold'],'Threshold drift in report')
    rows=[dict(trial=x['trial'],step=x['operating_point']['step'],**x['metrics']) for x in reports]
    pd.DataFrame(rows).to_csv(out/'historical_test.csv',index=False)
    q=read(R/'raw_cache/quality.json');audit=read(R/'archive_audit.json')
    trials=[]
    for t in s['trials']:
        cv=(t.get('cv') or {}).get('metrics',{});decision=t.get('acceptance') or {}
        record=dict(trial=t['id'],status=t['status'],adopted_final=t['id']==s['best'],
            cv_f1=cv.get('f1'),cv_TP=cv.get('TP'),cv_FP=cv.get('FP'),cv_FN=cv.get('FN'),
            accepted=decision.get('accepted'),probability_better=decision.get('probability_better'))
        p=R/'typeb/session/trials'/t['id']/'attempt_01/result.json'
        if p.exists():
            r=read(p);record.update(selection_f1=r['metrics']['f1'],features=len(r['features']),seconds=r['seconds'])
        trials.append(record)
    pd.DataFrame(trials).to_csv(out/'search_trials.csv',index=False)
    confirmation=read(R/'typeb/confirmation/summary.json')['reports']
    lines=['# TypeB v3.2：screen_margin=1.0 重跑',
        '\n相对上一轮 135 特征原始日志实验，仅将 screen_margin 从 0.02 改为 1.0；因此合法候选不会因前段验证 F1 偏低而跳过交叉验证。最终 CV 配对采纳规则、数据、特征、标签和时间切分不变。',
        '\n本次是新的自适应搜索，模型从头训练；随着新证据出现，Agent 的候选序列可能不同，因此不是对完全相同候选逐一配对重放。',
        '\n历史 TEST 已经查看过；本次是回顾性实验，不是独立盲测。没有按 TEST 重选模型、轮数或阈值。',
        f'\n搜索预算最多 13 组，实际记录 {len(s["trials"])} 组，最终采纳 `{s["best"]}`。',
        '\n停止原因（Agent 的决定，不是穷尽所有可能特征的证明）：'+str(s.get('stop_reason','预算结束')),
        '\n## 数据和控制条件',
        '\n保留原来的 66 个预处理特征、设备、观测时间、故障标签、随访和切分，增加 65 个原始日志衍生统计及 4 个位图可观测性计数，共 135 个基线输入。本轮直接复用上一轮已验证的 135 特征缓存，不重新解析或改变任何输入值。首组训练参数与上一轮原始日志基线一致；后续允许 Agent 在相同预算内提出对照。',
        f'\n原始包 SHA256：`{audit["sha256"]}`；包含 {audit["typeb_files"]} 台 TypeB 设备、{audit["rows"]:,} 条日志。68 台故障设备均有原始文件。原特征库 7,175 台中的 1,354 台没有对应原始文件，仍保留在原队列中，新增字段为 NaN。',
        '\n约 25% 的原始日志缺解码后的 burst_info/deviceID。缺失位图不是健康零；未知芯片不计为一个物理芯片。可观测性字段记录有效位图数量。原始统计描述已观测日志，并不证明日志完整。',
        f'\n复用缓存的原始构造通过 {q["causal_prefix_checks"]} 次真实设备前缀一致性检查，逐行验证旧列和元数据不变。输入只使用当前时刻及之前的日志，TRAIN/VAL 构造不使用后续 TEST 记录；六月七月无完整故障标签，继续排除。',
        '\n控制器沿用提交 `8ebafc8d311e734d2089df9d1106c962e2026b66`，评分为 `memory_15m_7d_device_f1_v2`，采纳规则仍为 CV 配对改善概率 >= 0.9 且点估计提高，部署工作点取训练期 CV 折外预测。',
        '\n特征 Agent 可使用新增数值字段及其语义、诊断证据，构造受因果约束的派生特征。它并不是直接读取任意原始 CSV 的通用代码代理；原始解析和数值化由已校验的适配器完成。',
        '\n## 冻结后的历史 TEST',
        '| 模型 | TP | FP | FN | Precision | Recall | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:lines.append(f'| {x["trial"]} | {x["TP"]} | {x["FP"]} | {x["FN"]} | {x["precision"]:.2%} | {x["recall"]:.2%} | {x["f1"]:.2%} |')
    lines+=['\n旧的预处理 66 特征 v3.2 基线在相同历史 TEST 的 CV 工作点为 TP=0、FP=4、FN=12，F1=0。请结合确认窗口、CV、误报数及实际故障样本数解释新结果，不能只凭一次回顾性 F1 宣称泛化改善。',
        '\n## 确认窗口与训练期 CV',
        '| 模型 | 确认 TP | 确认 FP | 确认 FN | 确认 F1 |',
        '|---|---:|---:|---:|---:|']
    for x in confirmation:
        v=x['metrics'];lines.append(f'| {x["trial"]} | {v["TP"]} | {v["FP"]} | {v["FN"]} | {v["f1"]:.2%} |')
    lines+=['\n确认窗口和 TEST 的样本不同，不能把确认分数当作测试成绩。历史确认窗口此前也有使用记录。',
        '| 组 | CV TP | CV FP | CV FN | CV F1 | 改善概率 | 本组采纳 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for t in trials:
        score='未执行' if t['cv_f1'] is None else f'{t["cv_f1"]:.2%}'
        lines.append(f'| {t["trial"]} | {t["cv_TP"]} | {t["cv_FP"]} | {t["cv_FN"]} | {score} | {t["probability_better"]} | {t["accepted"]} |')
    lines+=['\n## 控制检查',
        '\nconfig_diff.json 记录唯一实验配置变化为 screen_margin；其余差异仅为新目录。部署前验证 margin=1.0 允许最低可能的前段 F1 差距进入 CV，而改善概率 0.89 仍不采纳；TEST 缓存对搜索进程不可读，评价入口导入不会触发测试。旧会话及历史结果保留。',
        '\n## 导出内容',
        '\n包含新 session、events、Agent 调用记录、配置、主模型、诊断、确认和 TEST 结果，以及原始接入代码、来源与质量审计。旧会话哈希复核通过。',
        '\n`export_manifest.json` 列明文件哈希和大文件排除清单。原始 CSV、缓存、训练矩阵及 CV 折模型留在服务器。此导出是可审阅快照，不是独立可续跑备份；原始完成回执可能引用被排除的文件。']
    report='\n'.join(lines)+'\n';(out/'report.md').write_text(report,encoding='utf-8')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>TypeB 原始日志实验</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}td,th{padding:8px}</style>'+pd.DataFrame(rows).to_html(index=False)+'<pre>'+html.escape(report)+'</pre>',encoding='utf-8')
    save(out/'summary.json',dict(trials=trials,test=reports,confirmation=confirmation,adopted=s['best'],stop_reason=s.get('stop_reason'),old_sessions_unchanged=True,test_previously_exposed=True,screen_margin=1.0,reference_screen_margin=0.02,config_diff=read(R/'config_diff.json')))
    included={};excluded=[];events={}
    text_ext={'.json','.jsonl','.csv','.log','.txt','.md','.html','.py'}
    blocked={'raw','raw_cache','raw_cache_test','cache','cache_test','caches','matrices','feature_views','feature_samples','training_views','__pycache__','publication'}
    patterns=[re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),re.compile(rb'Bearer\s+[A-Za-z0-9._-]{24,}'),re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----')]
    archive=out/'snapshot.zip'
    def add(z,p,rel):
        require(p.stat().st_size<90*2**20,'Oversized review file '+rel)
        if p.suffix in text_ext:require(not any(rx.search(p.read_bytes()) for rx in patterns),'Credential-like content in '+rel)
        h=sha(p);z.write(p,rel);require(sha(p)==h,'File changed during export '+rel)
        included[rel]=dict(sha256=h,bytes=p.stat().st_size)
        if p.name=='events.jsonl':
            rr=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            events[rel]=dict(lines=len(rr),counts=dict(collections.Counter(x['event'] for x in rr)),sha256=h)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(R.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(R);why=None
            if any(x in blocked for x in rel.parts):why='raw_or_regenerable_cache'
            elif any(x.startswith('scores_') for x in rel.parts):why='per_row_scores'
            elif p.suffix in ('.parquet','.npz','.npy','.zip'):why='large_arrays_or_archive'
            elif p.suffix=='.ubj' and 'cv' in rel.parts:why='cv_fold_model'
            elif p.suffix=='.lock' or p.name.endswith('_pid.json'):why='transient'
            elif p.suffix not in text_ext|{'.ubj'}:why='non_review_file'
            if why:
                if 'publication' not in rel.parts:excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason=why))
            else:add(z,p,rel.as_posix())
        for n in ['report.md','report.html','summary.json','historical_test.csv','search_trials.csv']:add(z,out/n,n)
        for label,folder in [('train_val','raw_cache'),('test','raw_cache_test')]:
            for name in ['quality.json','manifest.json']:add(z,R/folder/name,'data_quality/'+label+'_'+name)
        z.writestr('export_manifest.json',json.dumps(dict(files=included,excluded_files=excluded,events=events,
            old_sessions_unchanged=True,complete_resume_checkpoint=False,test_previously_exposed=True),ensure_ascii=False,indent=2))
    preserved();save(out/'delivery.json',dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(included),events=events))
    print('REVIEW_EXPORT_READY',len(included),flush=True)

if __name__=='__main__':main()
