import sys,json
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');OUT=R/'rejected_comparison_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save
import pandas as pd

def main():
 paths=dict(v31_trial_001=R/'final_test_20261029/typeb',v31_trial_013=OUT/'typeb/trial_013',old_trial_015=OUT/'historical_typeb/old_trial_015')
 rows=[];summaries={}
 for name,folder in paths.items():
  df=pd.read_csv(folder/'devices_test.csv');df=df.loc[df.status.eq('eligible')&df.positive].copy()
  rep=read(folder/'report_test.json');m=rep['metrics'] if 'metrics' in rep else rep['current_full_cohort']
  df['model']=name;df['threshold']=m['threshold']
  df['reason']=df.apply(lambda r:'timely_hit' if r.hit else ('no_timely_scored_opportunity' if r.scored_timely_opportunities==0 else 'timely_scores_below_frozen_threshold'),axis=1)
  summaries[name]=dict(faults=len(df),hit=int(df.hit.sum()),missed=int(df.fn.sum()),reason_counts=df.reason.value_counts().to_dict(),outside_window_alarm_without_hit=int((df.alarm&~df.hit).sum()))
  rows.append(df[['model','sn_name','failure','threshold','max_score','max_hit_score','hit','fp','fn','scored_timely_opportunities','timely_history_opportunities','timely_prediction_opportunities','reason']])
 pd.concat(rows,ignore_index=True).to_csv(OUT/'typeb_fault_review.csv',index=False,encoding='utf-8-sig')
 save(OUT/'typeb_fault_review.json',summaries);print(json.dumps(summaries))

if __name__=='__main__':main()
