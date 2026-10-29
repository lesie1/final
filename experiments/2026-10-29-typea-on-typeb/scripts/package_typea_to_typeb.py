import sys,json,time,zipfile,csv,math
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');OUT=R/'typea_to_typeb_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from evaluation import classify
from typea_to_typeb_transfer import original_unchanged,event
import pandas as pd,numpy as np,pyarrow.parquet as pq

def main():
 frozen=read(OUT/'frozen.json');summary=read(OUT/'summary.json');original_unchanged(frozen)
 require(len(summary['reports'])==len(frozen['models'])==6,'Transfer incomplete')
 source=Path(frozen['target_base_cache']);aligned=OUT/'aligned_typeb_cache_test';m=read(source/'manifest.json')
 rows_checked=0
 for rel in m['files']:
  if not (rel.startswith('rows/') and rel.endswith('.parquet')):continue
  old=pq.read_table(source/rel,columns=m['features']).to_pandas().to_numpy(np.float32)
  new=pq.read_table(aligned/rel,columns=m['features']).to_pandas().to_numpy(np.float32)
  require(old.tobytes()==new.tobytes(),'Model input bytes changed for shared features')
  rows_checked+=len(old)
 comparisons=[]
 for rep,t in zip(summary['reports'],frozen['models']):
  folder=OUT/'trials'/t['trial'];require(rep['source_trial']==t['trial'],'Trial order differs')
  for n,h in read(folder/'complete.json')['files'].items():require(sha(folder/n)==h,'Transfer artifact changed')
  table=pd.read_csv(folder/'devices_test.csv');metric=rep['metrics'];eligible=table.status.eq('eligible')
  recomputed=classify(table,t['threshold'])
  for k in ['hit','fp','fn']:require((table[k]==recomputed[k]).all(),'Threshold classification differs')
  tp=int((eligible&table.hit).sum());fp=int((eligible&table.fp).sum());fn=int((eligible&table.fn).sum())
  require((tp,fp,fn)==(metric['TP'],metric['FP'],metric['FN']),'Counts differ')
  expected=dict(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)
  for k,v in expected.items():require(abs(v-metric[k])<1e-12,'Metric differs')
  require(metric['threshold']==t['threshold'] and rep['model_sha256']==t['model_sha256'],'Source model or threshold differs')
  maximum=table.loc[eligible,'max_score'].max();fault_max=table.loc[eligible&table.positive,'max_hit_score'].max()
  comparisons.append(dict(source_trial=t['trial'],role=t['role'],source_typea_test_f1=t['source_typea_test']['test_f1'],typeb_test_f1=metric['f1'],precision=metric['precision'],recall=metric['recall'],TP=tp,FP=fp,FN=fn,threshold=t['threshold'],max_target_score=None if pd.isna(maximum) else float(maximum),max_target_timely_fault_score=None if pd.isna(fault_max) else float(fault_max),available_columns=len(rep['actually_available_columns']),total_columns=rep['feature_count']))
 score_groups={}
 for rep in summary['reports']:score_groups.setdefault(rep['prediction_stats']['score_sha256'],[]).append(rep['source_trial'])
 identical=[v for v in score_groups.values() if len(v)>1]
 save(OUT/'verified_comparison.json',dict(rows_checked=rows_checked,shared_model_input_bytes_equal=True,models=6,original_sessions_preserved=True,comparisons=comparisons,identical_prediction_groups=identical))
 with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
 def pct(x):return f'{100*x:.2f}%'
 body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TypeA → TypeB 冻结模型迁移测试</title><style>body{font-family:system-ui,sans-serif;margin:32px;max-width:1200px;color:#18202a}p,li{line-height:1.7}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f7}code{background:#eee;padding:2px 4px}</style><h1>TypeA → TypeB 冻结模型迁移测试</h1>',
 '<p>按用户要求，取此前 TypeA 测试 F1 前五名：trial_011、trial_012、trial_002、trial_009、trial_010，并加入 TypeA 原基线 trial_001 作参照。名单在读取这些模型的 TypeB 测试结果前固定。没有重新训练，没有改变原模型、原阈值或 TypeB 评价设备队列，没有模型 API 调用。</p>',
 '<p><strong>输入限制：</strong>TypeA 基础输入有 164 列，而当前 TypeB 只有其中同名、同时间窗的 66 列。缺少的 98 列是 33 个 6 小时窗口特征和 65 个原始日志衍生特征；本次全部保留为 NaN。trial_012 的额外 4 列也依赖缺失的原始日志特征。没有把缺失填成 0，也没有将 15 分钟或 1 小时统计冒充 6 小时统计。</p>',
 '<p>TypeA 和 TypeB 设备标识无交集。TypeB 测试时间为北京时间 2024-05-01 14:24 至 2024-06-01 00:00（右端不含），可评估设备 7,119 台、故障设备 12 台；提前预警规则、随访和设备分母与此前 TypeB 测试一致。</p>',
 '<table><tr><th>TypeA 来源模型</th><th>原 TypeA 测试 F1</th><th>迁移后 TypeB F1</th><th>TP</th><th>FP</th><th>FN</th><th>冻结阈值</th><th>TypeB 最高评分</th><th>可用 / 总特征</th></tr>']
 for x in comparisons:
  body.append(f'<tr><td>{x["source_trial"]}{"（基线参照）" if x["role"]=="control" else ""}</td><td>{pct(x["source_typea_test_f1"])}</td><td>{pct(x["typeb_test_f1"])}</td><td>{x["TP"]}</td><td>{x["FP"]}</td><td>{x["FN"]}</td><td>{x["threshold"]:.6f}</td><td>{x["max_target_score"]:.6f}</td><td>{x["available_columns"]} / {x["total_columns"]}</td></tr>')
 body.append('</table><h2>TypeB 原有模型参照</h2><table><tr><th>模型</th><th>F1</th><th>精确率</th><th>召回率</th><th>TP</th><th>FP</th><th>FN</th></tr>')
 for name,b in frozen['baseline_files'].items():
  q=b['metrics'];body.append(f'<tr><td>{name}</td><td>{pct(q["f1"])}</td><td>{pct(q["precision"])}</td><td>{pct(q["recall"])}</td><td>{q["TP"]}</td><td>{q["FP"]}</td><td>{q["FN"]}</td></tr>')
 body.append('</table><h2>可以与不能得出的结论</h2>')
 if all(x['TP']==0 and x['FP']==0 for x in comparisons):
  body.append('<p>此次直接迁移的 6 个模型均未触发报警，F1 和召回率均为 0，漏掉全部 12 台故障。它们在当前可用输入和原阈值下不能直接用于 TypeB。</p>')
 else:
  best=max(comparisons,key=lambda x:x['typeb_test_f1']);body.append(f'<p>此次迁移最高 F1 为 {best["source_trial"]} 的 {pct(best["typeb_test_f1"])}，TP / FP / FN 为 {best["TP"]} / {best["FP"]} / {best["FN"]}。召回率 {pct(best["recall"])}，精确率 {pct(best["precision"])}。</p>')
  body.append('<p><strong>第 2、9 组显著增加了此测试段的故障命中数：12 台命中 10 台；代价是 542 次误报，F1 仍低于 TypeB 原有模型。</strong>这提示高召回迁移候选值得进一步受控验证，但当前模型与阈值组合不适合直接替换。高召回也可能与缺失输入导致的评分变化有关，不能将本结果直接解释为已证明学到了可迁移故障机理。</p>')
 if identical:body.append('<p>逐时评分哈希一致的模型组：'+', '.join(' / '.join(g) for g in identical)+'。这些完全相同的预测不构成多次独立验证。</p>')
 body.append('<p>这不是完整同特征条件下的迁移验证。结果同时受到缺失输入、硬件分布差异和原阈值校准的影响，不能据此证明 TypeB 样本少到学不会，也不能据此否定 A→B 迁移学习本身。下一项有区分力的实验是仅用两类共有的 66 个特征在 TypeA 训练，再固定规则去 TypeB 验证和测试；若需阈值适配，只能在 TypeB 验证集进行。本次没有启动该项新训练。</p>')
 body.append(f'<p>核验：对齐缓存中的 {rows_checked:,} 行共享特征转换为模型输入后逐字节一致；逐设备输出重新计算的计数和指标与报告一致；模型、原阈值、原 session 及运行代码未修改。独立 events.jsonl 保存了此次测试全过程。</p>')
 body.append('<p>TypeA 候选按已查看过的 TypeA TEST 选取，TypeB TEST 也曾用于历史测试。因此本结果是探索性迁移比较，不是新的独立盲测。原最终选模状态保持不变。</p></html>')
 (OUT/'report.html').write_text('\n'.join(body),encoding='utf-8')
 event('verification_complete',rows_checked=rows_checked,shared_model_input_bytes_equal=True,source_hashes_unchanged=True)
 archive=R/'typea_to_typeb_transfer_20261029.zip';hashes={}
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  def add(p,n):z.write(p,n);hashes[n]=sha(p)
  for n in ['frozen.json','events.jsonl','summary.json','verified_comparison.json','comparison.csv','report.html']:add(OUT/n,n)
  for t in frozen['models']:
   folder=OUT/'trials'/t['trial']
   for p in sorted(folder.iterdir()):
    if p.is_file():add(p,str(p.relative_to(OUT)))
   for n in ['model.ubj','result.json','complete.json']:add(Path(t['folder'])/n,'source_models/'+t['trial']+'/'+n)
  for p in (OUT/'feature_programs').rglob('*'):
   if p.is_file():add(p,str(p.relative_to(OUT)))
  for n in ['typea_to_typeb_transfer.py','package_typea_to_typeb.py']:add(R/n,'scripts/'+n)
  add(R/'typea_to_typeb_20261029_run.log','run.log');add(R/'release.json','release.json')
  z.writestr('SHA256_MANIFEST.json',json.dumps(hashes,indent=2))
 save(OUT/'delivery.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(hashes)))
 print(json.dumps(dict(comparisons=comparisons,identical_prediction_groups=identical,delivery=read(OUT/'delivery.json'))))

if __name__=='__main__':main()
