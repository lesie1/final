"""Build review reports and a credential-checked snapshot only after the batch completes."""
import collections,html,json,re,sys,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
import pandas as pd
from common import read,save,sha,require
from supervisor import preserved

def table(rows,columns):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join(
        '| '+' | '.join(str(r.get(k,'')) for k in columns)+' |\n' for r in rows)

def report():
    require(read(R/'status.json')['status']=='complete','Batch is not complete; do not publish it as final')
    preserved();out=R/'publication';out.mkdir(exist_ok=True)
    protocol=read(R/'protocol.json');search=[];trials=[];supplemental=[]
    for label in ('typea','typeb'):
        state=read(R/label/'session/state.json')
        for tr in state['trials']:
            cv=(tr.get('cv') or {}).get('metrics',{});decision=tr.get('acceptance') or {}
            trials.append(dict(task=label,trial=tr['id'],status=tr['status'],accepted=decision.get('accepted'),
                selection_f1=tr.get('metrics',{}).get('f1'),cv_f1=cv.get('f1'),
                probability_better=decision.get('probability_better'),rule=decision.get('rule'),
                current_best=(tr['id']==state['best'])))
        for r in read(R/label/'historical_test/summary.json')['reports']:
            m=r['metrics'];op=r['operating_point']
            search.append(dict(task=label,trial=r['trial'],role='baseline' if r['trial']==state['trials'][0]['id'] else 'accepted_best',
                point=r['point'],actual_source=op['source'],step=op['step'],**m))
    extra=R/'typeb/supplemental_trial013/summary.json'
    if extra.exists():
        evidence=read(extra)
        require(evidence['post_hoc_trial_selection'] and not evidence['threshold_reselected'], 'Unexpected supplemental protocol')
        for row in evidence['comparison']:
            supplemental.append(dict(task='typeb',role='baseline_reference' if row['trial']=='trial_001' else 'user_requested_post_hoc',**row))
    results=read(R/'controlled/evaluation/summary.json')['reports']
    controls=[dict(arm=r['arm'],seed=r['seed'],target=r['target'],split=r['split'],step=r['operating_point']['step'],**r['metrics']) for r in results]
    pooled=[]
    for seed in protocol['controlled']['seeds']:
        for arm in protocol['controlled']['arms']:
            for split in ('confirmation','test'):
                group=[r for r in controls if r['arm']==arm and r['seed']==seed and r['split']==split]
                require(len(group)==2,'Missing source evaluation')
                tp,fp,fn=[sum(r[k] for r in group) for k in ['TP','FP','FN']]
                pooled.append(dict(arm=arm,seed=seed,split=split,TP=tp,FP=fp,FN=fn,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                    precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.))
    for name,rows in [('search_trials',trials),('search_test',search),('controlled_results',controls),('pooled_results',pooled)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    if supplemental:pd.DataFrame(supplemental).to_csv(out/'supplemental_typeb_test.csv',index=False,encoding='utf-8-sig')
    btest=[r for r in controls if r['target']=='typeb' and r['split']=='test']
    bconfirm=[r for r in controls if r['target']=='typeb' and r['split']=='confirmation']
    def display(rows):
        return [dict(r,F1=f'{r["f1"]*100:.2f}%',Precision=f'{r["precision"]*100:.2f}%',Recall=f'{r["recall"]*100:.2f}%') for r in rows]
    cols=['arm','seed','TP','FP','FN','F1','Precision','Recall']
    lines=['# v3.2 内存搜索与共同 66 特征数据对照',
        '\n本报告记录已完成的预定实验。历史 TEST 先前已经查看过，后段 VAL 也有历史使用，因此所有新结果均为回顾性实验，不是全新独立盲测。',
        '\n## 实验范围',
        f'\n运行库：`{read(R/"release.json")["commit"]}`。A、B 搜索各最多 13 组；固定对照为四种训练数据 × 三个种子，共 12 组。失败和提前停止如实记录，旧 session 保留。',
        '\n新版搜索保留旧配置的数据切分、特征、采样及预算；仅采用新的概率采纳规则、取消净命中否决，并使用可用的 CV OOF 工作点。双分支等回退情况在 actual_source 字段列明。',
        '\n## 新版搜索的历史 TEST',
        '\n预定评估在搜索结束后比较基线和被规则采纳的最终最佳模型。下表同时给出新部署工作点和原选择窗口工作点；比较结果未送回 Agent。用户要求 B 先于 A 完成后续评估，其测试授权与执行回执另行保留。',
        table(display(search),['task','trial','role','point','actual_source','step','TP','FP','FN','F1']),
        *(['\n## 用户追加的 B 第 13 组历史 TEST',
           '\n用户在查看 B 基线 TEST 成绩后，明确要求追加评估未被搜索规则采纳的第 13 组。这是事后选定候选的探索性检查，不是独立盲测，也没有更改 session 的最佳模型。两套轮数、阈值均来自测试前已保存的训练期结果；未按 TEST 重选阈值。',
           table(display(supplemental),['trial','role','point','step','threshold','TP','FP','FN','F1'])] if supplemental else []),
        '\n## 固定 66 特征：B 确认窗口',table(display(bconfirm),cols),
        '\n## 固定 66 特征：B 历史 TEST',table(display(btest),cols),
        '\n## 如何解释数据对照',
        '\n四组均使用相同固定参数、每设备总权重为 1 的规则，以及全部满足成熟期要求的 TRAIN 事件行；没有额外特征搜索，也没有沿用 A 旧搜索的负样本抽样。故不应将这些对照与 164 特征的 A 搜索直接当成单因素比较。',
        '\nA-only 的树只用 A 训练，但工作点使用 B TRAIN 折外预测及其标签校准；它不是完全未见 B 标签的部署方案。A+B 每折训练时排除该折全部 B 设备。所有组都用相同 B 训练期折、时间段和固定检查步数选择阈值；后续 VAL/TEST 不重新选择阈值。',
        '\n小规模 A 按训练期有效正样本设备数和其余训练设备数与 B 匹配，并保留每台设备的完整因果历史；事件行数不强制相等。每个种子独立抽样，样本名单已保存。该对照可检查 A 内部缩小设备规模的影响，但不能单独证明 B 的困难只有样本量原因。',
        '\nA、B 的 observed_start、observed_end 和故障记录各自保留；未知随访不会当健康负样本。共同字段仅按已提供的同名统计及历史窗口对应，数据分布差异仍可能存在。',
        '\nB 为主要评价对象，A 和合并 TP/FP/FN 的结果保存在 CSV。合并 F1 会受 A 规模影响，不能代替 B 的结果。三个种子共享评价设备，不是三份独立测试集；报告全部种子，不按 TEST 挑选赢家。',
        *(['\n## 本轮结果的含义',*read(R/'publication_findings.json')['paragraphs']] if (R/'publication_findings.json').exists() else []),
        '\n## 文件',
        '\n- `typea/session/`、`typeb/session/`：完整保留导出范围内的搜索事件、Agent 请求/回复、状态、参数、诊断和主模型。',
        '- `controlled/`：固定对照的协议、状态、events、抽样设备、模型、训练审计、OOF 汇总及评价明细。',
        '- `protocol.json`、`release.json`、`source_preservation.json`：预定实验规则、程序版本及旧会话校验。',
        '- `typeb/supplemental_trial013/`：追加测试的冻结设置、events、两套评分与完整设备明细；`supplemental_typeb_test.csv` 为基线对比表。',
        '- `export_manifest.json`：每个导出文件的哈希和明确的排除清单。大型特征缓存、训练矩阵、逐行分数和部分 CV 折模型保留在服务器；该 GitHub 快照不声称可独立续跑。',
        '\n原始 complete.json 回执未改写，其中可能引用未导出的大文件。所有凭据均排除。']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text,encoding='utf-8')
    # Self-contained HTML: preserve full report text and render sortable-source tables as ordinary HTML tables.
    tables=''.join('<h2>'+html.escape(title)+'</h2>'+pd.DataFrame(display(rows)).to_html(index=False,escape=True)
        for title,rows in [('新版搜索：历史 TEST',search),*([('用户追加：B 第 13 组',supplemental)] if supplemental else []),('B 确认窗口',bconfirm),('B 历史 TEST',btest)])
    page='<!doctype html><meta charset="utf-8"><title>v3.2 实验报告</title><style>body{font:16px/1.65 system-ui;margin:32px;color:#172033}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5df}th{background:#eaf0f7}pre{white-space:pre-wrap;background:#f5f7fa;padding:20px}</style><h1>v3.2 实验报告</h1><p>历史数据回顾性评价；未按 TEST 重新调阈值。</p>'+tables+'<h2>协议与说明</h2><pre>'+html.escape(text)+'</pre>'
    (out/'report.html').write_text(page,encoding='utf-8')
    save(out/'summary.json',dict(search=search,supplemental=supplemental,controlled=controls,pooled=pooled,independent_blind_test=False,old_sessions_unchanged=True))
    return out

def bundle():
    out=report();archive=out/'snapshot.zip';included={};excluded=[];events={}
    blocked={'cache','cache_test','caches','evaluation_cache_test','matrices','feature_views','feature_samples','training_views','__pycache__','publication'}
    text_ext={'.json','.jsonl','.csv','.log','.txt','.md','.html','.py'}
    patterns=[re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),re.compile(rb'Bearer\s+[A-Za-z0-9._-]{24,}')]
    def add(z,p,rel):
        require(p.stat().st_size<90*2**20,'Oversized Git file: '+rel)
        if p.suffix in text_ext:
            data=p.read_bytes();require(not any(rx.search(data) for rx in patterns),'Credential-like value in '+rel)
        before=sha(p);z.write(p,rel);require(sha(p)==before,'File changed during export: '+rel)
        included[rel]=dict(sha256=before,bytes=p.stat().st_size)
        if p.name=='events.jsonl':
            rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            events[rel]=dict(lines=len(rows),counts=dict(collections.Counter(r['event'] for r in rows)),sha256=before)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(R.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(R);why=None
            if any(x in blocked for x in rel.parts):why='regenerable_data_or_build_cache'
            elif any(x.startswith('scores_') for x in rel.parts):why='per_row_scores'
            elif p.suffix in ('.parquet','.npz','.zip'):why='large_scores_or_existing_archive'
            elif p.suffix=='.npy' and not (rel.parts[0]=='controlled' and len(rel.parts)==2):why='training_array'
            elif p.suffix=='.ubj' and 'cv' in rel.parts:why='cv_fold_model'
            elif p.name.startswith(('devices_val','unknown_devices_val')):why='selection_device_table'
            elif p.suffix=='.lock' or p.name.endswith('_pid.json'):why='transient_process_file'
            elif p.suffix not in text_ext|{'.ubj','.npy'}:why='non_review_file'
            if why:
                if 'publication' not in rel.parts:excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason=why))
            else:add(z,p,rel.as_posix())
        for p in sorted(out.iterdir()):
            if p.suffix in {'.md','.html','.json','.csv'} and p.name!='delivery.json':add(z,p,p.name)
        manifest=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=included,
            excluded_files=excluded,events=events,old_sessions_unchanged=True,independent_blind_test=False,
            complete_resume_checkpoint=False,note='Original completion receipts may refer to omitted data/OOF arrays and fold models; these remain on the server.')
        z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    preserved()
    delivery=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(included),events=events)
    save(out/'delivery.json',delivery);print(json.dumps(delivery),flush=True)

if __name__=='__main__':bundle()
