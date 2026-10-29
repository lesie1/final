"""New raw-B session: immutable v3.2 controller plus independently sealed adapter."""
import copy, io, json, os, shutil, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(__file__).resolve().parent
PREV=Path('/root/autodl-tmp/final_v32_20261029')
OLD=Path('/root/autodl-tmp/final_v31_20261028')
SOURCE=Path('/root/autodl-tmp/raw_upload/stage1_csv.zip')
EXPECTED='85c67e8d6466238ada5f19eadaa3f3693593f02bb214a8d8f21d4a6b875b52f5'
if not (R/'runtime').exists():shutil.copytree(PREV/'runtime',R/'runtime')
sys.path.insert(0,str(R/'runtime'));sys.path.insert(0,str(R));sys.dont_write_bytecode=True
from common import read,save,sha,digest,require,code_hashes,event
from history_math import verify
from raw_cache import META
from agent import Loop
from settings import validate_config
import raw_typeb as raw

def one_part(job):
    part,source_cache,out,registry,cutoff=job
    source=Path(source_cache)/'rows'/part;dest=Path(out)/'rows'/part
    f=pq.read_table(source).to_pandas()
    codes=f.device.to_numpy();cuts=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(f)]
    extra=np.full((len(f),len(raw.FEATURES)),np.nan,np.float32)
    stats=dict(part=part,devices=0,with_raw=0,raw_events=0,decoded_events=0,causal_checks=0,bitmap_chip_checks=0)
    for a,b in zip(cuts[:-1],cuts[1:]):
        stats['devices']+=1;sn=registry[int(codes[a])];p=R/'raw/type_B'/(sn+'.csv')
        if not p.exists():continue
        frame=pd.read_csv(p,usecols=raw.RAW_COLUMNS,dtype={'burst_info':str,'error_type_full_name':str},low_memory=False)
        frame=frame[pd.to_numeric(frame.LogTime)<cutoff]
        require(all(pd.to_numeric(frame[c],errors='coerce').notna().all() for c in ['RankId','BankgroupId','BankId','RowId','ColumnId']), 'Missing spatial coordinates')
        # Validate x4 bitmap decoding on every distinct known (bitmap,chip) pair.
        for text,chip in frame[['burst_info','deviceID']].dropna().drop_duplicates().itertuples(index=False,name=None):
            import ast
            value=ast.literal_eval(text)
            bits=[int(bit) for arr in value.values() for bit in arr]
            require(all(bit//4==int(chip) for bit in bits),'Bitmap chip semantics differ for '+sn)
            stats['bitmap_chip_checks']+=1
        ev=raw.events(frame)
        anchors=f.time.to_numpy()[a:b]
        extra[a:b]=raw.features_at(ev,anchors)
        if len(anchors) and len(ev):
            cut=int(anchors[len(anchors)//2]);at=np.array([cut])
            np.testing.assert_allclose(raw.features_at(ev,at),raw.features_at(ev[ev.time<=cut].reset_index(drop=True),at),equal_nan=True)
            stats['causal_checks']+=1
        stats['with_raw']+=1;stats['raw_events']+=len(ev);stats['decoded_events']+=int(np.isfinite(ev.err_bits).sum())
    require(not np.isinf(extra).any(),'Invalid raw feature infinity')
    result=pd.concat([f,pd.DataFrame(extra,columns=raw.FEATURES,index=f.index)],axis=1)
    keep=[c for c in result if c not in META]+META
    pq.write_table(pa.Table.from_pandas(result[keep],preserve_index=False),dest,compression='zstd',row_group_size=65536)
    # Verify the old features/labels/cohort/timestamps exactly, not just their counts.
    restored=pq.read_table(dest,columns=list(f.columns)).to_pandas()
    pd.testing.assert_frame_equal(f,restored,check_exact=True)
    stats['finite_counts']={n:int(np.isfinite(extra[:,i]).sum()) for i,n in enumerate(raw.FEATURES)}
    stats['rows']=len(f);stats['source_sha256']=sha(source);stats['output_sha256']=sha(dest)
    save(Path(out)/(part+'.audit.json'),stats)
    print('PART_READY',part,'rows',len(f),'raw_devices',stats['with_raw'],flush=True)
    return stats

def build(cache,out,final_test=False):
    m=verify(cache);require(bool(m['test_opened'])==bool(final_test),'Wrong cache partition')
    if (out/'manifest.json').exists():return verify(out)
    out.mkdir(exist_ok=True);(out/'rows').mkdir(exist_ok=True)
    reg=pd.read_parquet(cache/'device_metadata.parquet').set_index('device').sn_name.to_dict()
    cutoff=m['windows']['test' if final_test else 'val'][1]
    jobs=[(p.name,str(cache),str(out),reg,cutoff) for p in sorted((cache/'rows').glob('*.parquet'))]
    with ProcessPoolExecutor(2) as pool:stats=list(pool.map(one_part,jobs))
    for n in ('device_metadata.parquet','devices.json','excluded.json'):shutil.copy2(cache/n,out/n)
    result=copy.deepcopy(m);result['features']=m['features']+raw.FEATURES
    result['raw_feature_descriptions']=raw.mapping_features()
    result['raw_feature_spec']=dict(version=raw.VERSION,source_manifest=sha(cache/'manifest.json'),raw_archive_sha256=EXPECTED,
        adapter_sha256=sha(R/'raw_typeb.py'),parser_sha256=sha(R/'runtime/raw_events.py'),cutoff=cutoff,
        final_test=final_test,policy='66 original features retained, 65 raw descriptors plus 4 bitmap-observability counts. Missing raw files stay NaN. Events <= anchor only; cutoff strictly excludes partition end. No labels in adapter.')
    result['files']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*.parquet')}
    result['files'].update({n:sha(out/n) for n in ('devices.json','excluded.json')})
    save(out/'manifest.json',result)
    finite={n:sum(x['finite_counts'][n] for x in stats) for n in raw.FEATURES}
    require(all(finite[n]>0 for n in raw.FEATURES),'Empty raw feature columns')
    save(out/'quality.json',dict(parts=stats,finite_counts=finite,source_rows_and_labels_identical=True,
        causal_prefix_checks=sum(x['causal_checks'] for x in stats),features=len(result['features'])))
    return verify(out)

def main():
    if (R/'ready.json').exists():print('ALREADY_READY');return
    event(R,'preparation_started',maximum_trials=13)
    require(sha(SOURCE)==EXPECTED,'Raw archive differs from local supplied archive')
    prior={}
    for label,path in [('v31_typeb',OLD/'typeb/session'),('v32_typeb',PREV/'typeb/session'),('v32_typea',PREV/'typea/session')]:
        prior[label]=dict(path=str(path.resolve()),seals={n:sha(path/n) for n in ['config.json','state.json','events.jsonl','mapping.json','code_seal.json']})
    save(R/'source_preservation.json',prior)
    data=R/'raw';data.mkdir(exist_ok=True)
    if not (data/'extracted.json').exists():
        with zipfile.ZipFile(SOURCE) as z:
            names=[n for n in z.namelist() if n.startswith('type_B/') and n.endswith('.csv')]
            require(len(names)==5821,'Raw file inventory changed')
            for n in names:
                require((data/n).resolve().is_relative_to(data.resolve()),'Unsafe archive path')
                z.extract(n,data)
            (data/'failure_ticket_all.csv').write_bytes(z.read('failure_ticket.csv'))
        save(data/'extracted.json',dict(source=str(SOURCE),sha256=EXPECTED,files=len(names)))
    oldc=read(PREV/'typeb/session/config.json');source_cache=Path(oldc['active_cache'])
    m=build(source_cache,R/'raw_cache')
    task=R/'typeb';session=task/'session';session.mkdir(parents=True,exist_ok=True)
    require(not (session/'state.json').exists(),'Session already created; inspect before rerunning preparation')
    c=copy.deepcopy(oldc);c.update(output_root=str(session),workspace=str(task/'workspace'),active_cache=str(R/'raw_cache'),
        max_trials=13,max_total_features=len(m['features'])+24,
        instructions='TypeB only. New raw-data representation: retain the original 66 precomputed features and add 69 causal raw-CE descriptors including bitmap observability. Raw descriptors are available to feature hypotheses, diagnostics and transform programs. Use their definitions in raw_feature_descriptions. Missing bitmaps/chips/files are unknown, not healthy zero. Keep source samples, labels, weights and v3.2 scoring unchanged for baseline. Do not infer healthy labels beyond May 2024. Historical TEST has been seen previously; never use it for search or thresholds.')
    validate_config(c)
    mapping=read(PREV/'typeb/session/mapping.json')
    mapping['description']+=' New active cache additionally contains 69 causal raw CE descriptors; the cache manifest documents definitions and raw source identity.'
    save(session/'config.json',c);save(session/'mapping.json',mapping);save(session/'code_seal.json',code_hashes())
    fingerprint=digest(dict(base=read(PREV/'typeb/session/state.json')['dataset'],raw=EXPECTED,adapter=sha(R/'raw_typeb.py'),cache=sha(R/'raw_cache/manifest.json')))
    save(session/'inventory.json',dict(fingerprint=fingerprint,source_cache=str(source_cache),raw_archive_sha256=EXPECTED))
    save(session/'state.json',dict(version=1,status='ready',config_hash=digest(c),dataset=fingerprint,mapping_hash=digest(mapping),calls=0,tokens=0,
        trials=[],best=None,pending=None,analyze_next=None,diagnostics={},feature_jobs=0,experience=None,experience_tried=True,last_failure=None))
    save(session/'bootstrap.json',dict(source=str(PREV/'typeb/session'),model_weights_imported=False,base_rows_unchanged=True,
        old_session_kind='precomputed_66_features',new_session_kind='precomputed_66_plus_raw69',test_previously_exposed=True))
    save(task/'isolation.json',dict(forbidden_reads=[str(R/'raw'),str(SOURCE),str(OLD/'typeb/input'),str(OLD/'final_test_20261029'),
        str(PREV/'typeb/historical_test'),str(PREV/'typeb/supplemental_trial013'),str(PREV/'controlled')],
        readonly_roots=[str(OLD),str(PREV),str(R/'raw_cache'),str(R/'runtime')]))
    Loop(session).prepare()
    save(R/'release.json',dict(runtime_commit='8ebafc8d311e734d2089df9d1106c962e2026b66',runtime_modified=False,
        runtime_hashes=code_hashes(),adapter_hashes={n:sha(R/n) for n in ['raw_typeb.py','prepare.py','test_raw_typeb.py']}))
    for v in prior.values():
        for n,h in v['seals'].items():require(sha(Path(v['path'])/n)==h,'Old session changed')
    save(R/'ready.json',dict(status='ready',max_trials=13,features=len(m['features']),raw_features=69,test_opened=False,
        existing_source_verified=True,source_sha256=EXPECTED,quality=str(R/'raw_cache/quality.json')))
    save(R/'status.json',dict(status='ready',at=time.time()))
    event(R,'preparation_complete',features=len(m['features']),old_sessions_unchanged=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(R/'status.json',dict(status='preparation_failed',error=str(e),at=time.time()));raise
