"""Four frozen 66-input data controls. No agent calls or TEST feedback into fitting."""
import copy,gc,hashlib,json,os,shutil,sys,time
from pathlib import Path
R=Path(os.environ.get('V32_ROOT',Path(__file__).resolve().parent))
sys.path.insert(0,os.environ.get('V32_RUNTIME',str(R/'runtime')))
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from common import read,save,sha,require,event
from history_math import verify
from sample_builder import build_training
from crossval import folds,periods,score_devices,summarize
from prediction import predict
from evaluation import metrics,lead_summary

C=R/'controlled'

def update_state(**changes):
    state=read(C/'state.json') if (C/'state.json').exists() else dict(status='preparing',trials={},pending=None)
    state.update(changes);save(C/'state.json',state)

def project_cache(source,dest,features):
    source,dest=Path(source),Path(dest)
    m=verify(source);stamp=dict(source=str(source),source_manifest_sha256=sha(source/'manifest.json'),features=features)
    if (dest/'manifest.json').exists():
        out=verify(dest);require(out['projection']==stamp,'Projection identity differs');return out
    require(set(features)<=set(m['features']),'Missing common input')
    dest.mkdir(exist_ok=True,parents=True);rows=0
    for rel in m['files']:
        src=source/rel;out=dest/rel;out.parent.mkdir(exist_ok=True,parents=True)
        if rel.startswith('rows/') and rel.endswith('.parquet'):
            cols=pq.ParquetFile(src).schema_arrow.names
            keep=features+[x for x in cols if x not in m['features']]
            table=pq.read_table(src,columns=keep)
            pq.write_table(table,out,compression='zstd',row_group_size=65536)
            require(table.equals(pq.read_table(out)),'Projection changed values or missingness')
            rows+=len(table)
        else:shutil.copy2(src,out)
    result=copy.deepcopy(m);result.update(features=features,projection=stamp)
    result['files']={rel:sha(dest/rel) for rel in m['files']}
    save(dest/'manifest.json',result)
    require(sha(source/'device_metadata.parquet')==sha(dest/'device_metadata.parquet'),'Registry changed')
    return verify(dest)

def component_arrays(folder):
    p=Path(folder)
    return tuple(np.load(p/n,mmap_mode='r',allow_pickle=False) for n in
                 ['train_matrix.npy','labels.npy','weights.npy','devices.npy'])

def eligible_devices(folder):
    x,y,w,d=component_arrays(folder)
    all_ids=np.unique(d);pos=np.unique(d[y==1]);neg=np.setdiff1d(all_ids,pos)
    return pos,neg

def sample_small_a(a_folder,b_folder,seed):
    ap,an=eligible_devices(a_folder);bp,bn=eligible_devices(b_folder)
    require(len(ap)>=len(bp) and len(an)>=len(bn),'Insufficient A devices for matched control')
    rng=np.random.default_rng(seed)
    return np.sort(np.r_[rng.choice(ap,len(bp),replace=False),rng.choice(an,len(bn),replace=False)])

def build_matrix(config,manifest,recipe,folder):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    if (folder/'receipt.json').exists():
        done=read(folder/'receipt.json')
        for n,h in done['files'].items():require(sha(folder/n)==h,'Prepared matrix changed')
        return done
    require(not config.get('sampling_reference'),'Controls use full mature TRAIN rows')
    x,y,w,names,ratio,audit,devices=build_training(config,recipe,manifest,folder,
        config['minimum_lead_seconds'],config['memory_limit_gb'],return_devices=True)
    require(names==manifest['features'],'Matrix feature order differs')
    for n,v in [('labels',y),('weights',w),('devices',devices)]:np.save(folder/(n+'.npy'),v,allow_pickle=False)
    x.flush();del x;gc.collect()
    save(folder/'audit.json',audit)
    done=dict(files={n:sha(folder/n) for n in ['train_matrix.npy','labels.npy','weights.npy','devices.npy','audit.json']},
              features=names,manifest_sha256=sha(Path(config['active_cache'])/'manifest.json'))
    save(folder/'receipt.json',done);return done

def fit(components,recipe,out,rounds,device='cuda:0',threads=8):
    """Each mask selects whole devices; each included device retains unit base weight."""
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'fit_complete.json').exists():
        done=read(out/'fit_complete.json')
        for n,h in done['files'].items():require(sha(out/n)==h,'Fitted artifact changed')
        b=xgb.Booster();b.load_model(out/'model_full.ubj');b.set_param(dict(device='cpu',nthread=threads));return b
    selected=[];audit=[];total=0;width=None
    for name,folder,mask in components:
        x,y,w,d=component_arrays(folder)
        ix=None if mask is None else np.flatnonzero(np.asarray(mask,dtype=bool)[d])
        count=len(y) if ix is None else len(ix)
        if not count:continue
        require(width is None or width==x.shape[1],'Input width mismatch');width=x.shape[1]
        ids=d if ix is None else d[ix];ww=w if ix is None else w[ix];yy=y if ix is None else y[ix]
        masses=np.bincount(ids,weights=ww)
        require(np.allclose(masses[masses>0],1.,rtol=2e-6),'Device weight changed under whole-device mask')
        audit.append(dict(source=name,rows=count,devices=int(len(np.unique(ids))),positive_devices=int(len(np.unique(ids[yy==1]))),
            positive_rows=int(yy.sum()),device_ids_sha256=hashlib.sha256(np.unique(ids).astype('<i8').tobytes()).hexdigest(),
            weighted_positive=float(ww[yy==1].sum(dtype=np.float64)),weighted_negative=float(ww[yy==0].sum(dtype=np.float64))))
        selected.append((x,y,w,ix));total+=count
    require(total>0,'No training rows')
    scratch=out/'combined_matrix.npy'
    require(shutil.disk_usage(out).free>total*width*4+8*2**30,'Insufficient scratch disk')
    combined=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float32,shape=(total,width))
    labels=np.empty(total,np.uint8);weights=np.empty(total,np.float32);offset=0
    for x,y,w,ix in selected:
        count=len(y) if ix is None else len(ix)
        for lo in range(0,count,262144):
            hi=min(count,lo+262144);take=slice(lo,hi) if ix is None else ix[lo:hi]
            combined[offset+lo:offset+hi]=x[take];labels[offset+lo:offset+hi]=y[take];weights[offset+lo:offset+hi]=w[take]
        offset+=count
    require(len(np.unique(labels))==2,'Need both classes')
    ratio=float(weights[labels==0].sum(dtype=np.float64)/weights[labels==1].sum(dtype=np.float64))
    positive_weight={'ratio':ratio,'sqrt_ratio':ratio**.5,'unit':1.}[recipe['weight_policy']]
    dm=xgb.QuantileDMatrix(combined,label=labels,weight=weights,max_bin=recipe['params']['max_bin'],nthread=threads)
    del combined,selected;gc.collect();scratch.unlink()
    params=dict(recipe['params']);params['eta']=params.pop('learning_rate')
    params.update(objective='binary:logistic',tree_method='hist',device=device,nthread=threads,subsample=1.,
                  scale_pos_weight=positive_weight,base_score=.5,seed=recipe['seed'],disable_default_eval_metric=1)
    save(out/'train_audit.json',dict(components=audit,params=params,rows=total,features=width,
                                   rounds=rounds,fit_uses_only_train=True,base_weight='one per included device'))
    b=xgb.Booster(params,cache=[dm])
    for step in range(rounds):
        b.update(dm,step)
        if (step+1)%100==0:print('CONTROL_TRAIN',str(out),step+1,flush=True)
    actual=json.loads(b.save_config())['learner']['generic_param']['device']
    require(not device.startswith('cuda') or actual.startswith('cuda'),'Unexpected CPU fallback')
    model=xgb.Booster();model.load_model(b.save_raw(raw_format='ubj'));model.set_param(dict(device='cpu',nthread=threads))
    model.save_model(out/'model_full.ubj');del b,dm;gc.collect()
    save(out/'fit_complete.json',dict(files={n:sha(out/n) for n in ['model_full.ubj','train_audit.json']}))
    return model

def prepare():
    C.mkdir(exist_ok=True)
    if (C/'ready.json').exists():return
    protocol=read(R/'protocol.json')['controlled'];recipe=protocol['recipe']
    source={label:read(R/label/'session/config.json') for label in ('typea','typeb')}
    ma=verify(source['typea']['active_cache']);mb=verify(source['typeb']['active_cache'])
    features=mb['features'];require(len(features)==66 and set(features)<=set(ma['features']),'Unexpected common schema')
    require(ma['windows']==mb['windows'],'Splits differ')
    a_ids=set(read(Path(source['typea']['active_cache'])/'devices.json'))
    b_ids=set(read(Path(source['typeb']['active_cache'])/'devices.json'))
    require(not a_ids.intersection(b_ids),'Source device identities overlap')
    event(C,'preparation_started',features=66,arms=protocol['arms'])
    a_cache=C/'caches/typea66';project_cache(source['typea']['active_cache'],a_cache,features)
    entries={}
    for label,cache in [('typea',a_cache),('typeb',Path(source['typeb']['active_cache']))]:
        config=dict(source[label],active_cache=str(cache),output_root=str(C/label),sampling_reference=None,
                    max_total_features=66,memory_limit_gb=80,num_threads=8,score_workers=4)
        folder=C/'matrices'/label;m=read(cache/'manifest.json')
        done=build_matrix(config,m,recipe,folder)
        config_path=C/(label+'_config.json');save(config_path,config)
        entries[label]=dict(cache=str(cache),matrix=str(folder),config=str(config_path),
            matrix_receipt_sha256=sha(folder/'receipt.json'),manifest_sha256=sha(cache/'manifest.json'))
    registry=pd.read_parquet(Path(entries['typeb']['cache'])/'device_metadata.parquet')
    fold_ids=folds(registry,mb['windows']['train'],protocol['folds'],protocol['fold_seed'])
    np.save(C/'b_folds.npy',fold_ids,allow_pickle=False)
    small={}
    for seed in protocol['seeds']:
        ids=sample_small_a(entries['typea']['matrix'],entries['typeb']['matrix'],seed)
        path=C/f'a_small_seed{seed}.npy';np.save(path,ids,allow_pickle=False);small[str(seed)]=sha(path)
    save(C/'prepared.json',dict(sources=entries,features=features,windows=mb['windows'],
        b_folds_sha256=sha(C/'b_folds.npy'),a_small_sha256=small,protocol_sha256=sha(R/'protocol.json'),
        source_followup_preserved=True,source_device_overlap=0,threshold_target='B TRAIN OOF',test_opened=False))
    save(C/'isolation.json',dict(forbidden_reads=[str(R/'typea/historical_test'),str(R/'typeb/historical_test'),
        '/root/autodl-tmp/final_v31_20261028/final_test_20261029',
        '/root/autodl-tmp/final_v31_20261028/rejected_comparison_20261029',
        '/root/autodl-tmp/final_v31_20261028/typea_to_typeb_20261029'],
        readonly_roots=[str(R/'runtime'),str(C/'matrices'),str(C/'caches'),str(Path(source['typeb']['active_cache']))]))
    save(C/'ready.json',dict(status='ready',trial_count=12))
    update_state(status='ready',trials={},pending=None,max_trials=12,test_opened=False)
    event(C,'preparation_complete',sources=entries,training_fault_devices={k:len(eligible_devices(v['matrix'])[0]) for k,v in entries.items()})

def train(arm,seed):
    protocol=read(R/'protocol.json')['controlled'];prepared=read(C/'prepared.json');src=prepared['sources']
    require(arm in protocol['arms'] and seed in protocol['seeds'],'Unplanned control')
    require(sha(R/'protocol.json')==prepared['protocol_sha256'],'Protocol changed')
    require(sha(C/'b_folds.npy')==prepared['b_folds_sha256'],'Folds changed')
    for source in src.values():
        require(sha(Path(source['matrix'])/'receipt.json')==source['matrix_receipt_sha256'],'Matrix receipt changed')
        require(sha(Path(source['cache'])/'manifest.json')==source['manifest_sha256'],'Cache manifest changed')
    root=C/'trials'/f'{arm}_seed{seed}';root.mkdir(exist_ok=True,parents=True)
    if (root/'complete.json').exists():
        for n,h in read(root/'complete.json')['files'].items():require(sha(root/n)==h,'Completed control changed')
        return
    event(C,'control_started',arm=arm,seed=seed)
    update_state(status='running',pending=dict(arm=arm,seed=seed))
    recipe=dict(protocol['recipe'],seed=seed);save(root/'recipe.json',recipe)
    c=read(src['typeb']['config']);m=read(Path(src['typeb']['cache'])/'manifest.json')
    folds_b=np.load(C/'b_folds.npy',allow_pickle=False);plist=periods(m,c)
    steps=list(range(protocol['eval_every'],protocol['rounds']+1,protocol['eval_every']))
    if protocol['rounds'] not in steps:steps.append(protocol['rounds'])
    a_mask=None
    if arm=='A_small':
        path=C/f'a_small_seed{seed}.npy';require(sha(path)==prepared['a_small_sha256'][str(seed)],'Small-A sample changed')
        a_mask=np.zeros(read(Path(src['typea']['cache'])/'manifest.json')['included_devices'],bool)
        a_mask[np.load(path,allow_pickle=False)]=True
    def components(fold=None):
        out=[]
        if arm!='B_only':out.append(('typea',src['typea']['matrix'],a_mask))
        if arm in ('B_only','A_plus_B'):out.append(('typeb',src['typeb']['matrix'],None if fold is None else folds_b!=fold))
        return out
    full=None
    if arm in ('A_only','A_small'):
        full=fit(components(),recipe,root/'full',protocol['rounds'])
    frames=[]
    for fold in range(protocol['folds']):
        folder=root/'cv'/f'fold_{fold}';folder.mkdir(exist_ok=True,parents=True)
        table_path=folder/'oof.parquet'
        if table_path.exists() and (folder/'score_complete.json').exists():
            require(sha(table_path)==read(folder/'score_complete.json')['sha256'],'OOF changed')
            frames.append(pd.read_parquet(table_path));continue
        model=full if full is not None else fit(components(fold),recipe,folder,protocol['rounds'])
        held=np.flatnonzero(folds_b==fold)
        # The only B rows used in fitting have whole-device IDs outside this held set.
        if arm in ('B_only','A_plus_B'):require(not np.any((folds_b!=fold)[held]),'B fold leakage')
        table=score_devices(model,steps,c,recipe,m,plist,held);table['fold']=fold
        require(set(table.device)<=set(held),'OOF scores include training devices')
        table.to_parquet(table_path,index=False);save(folder/'score_complete.json',dict(sha256=sha(table_path),held_devices=len(held)))
        frames.append(table);event(C,'control_fold_complete',arm=arm,seed=seed,fold=fold)
        if full is None:del model;gc.collect()
    oof=pd.concat(frames,ignore_index=True)
    require(not oof.duplicated(['period','device','step']).any(),'OOF device-period duplicate')
    oof.to_parquet(root/'cv/oof.parquet',index=False);summary=summarize(oof)
    save(root/'cv/oof_complete.json',dict(summary,oof_sha256=sha(root/'cv/oof.parquet'),folds=protocol['folds'],
        calibration_target='B TRAIN only',periods=plist,recipe=recipe))
    if full is None:full=fit(components(),recipe,root/'full',protocol['rounds'])
    op=dict(source='B_train_oof',model='full/model_full.ubj',model_sha256=sha(root/'full/model_full.ubj'),
        step=summary['best_step'],threshold=summary['cv_metrics']['threshold'],cv_metrics=summary['cv_metrics'],
        oof_sha256=sha(root/'cv/oof.parquet'),target='typeb',test_opened=False)
    save(root/'operating_point.json',op)
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='complete.json'}
    save(root/'complete.json',dict(files=files,arm=arm,seed=seed,trained=True,test_opened=False))
    trials=read(C/'state.json')['trials'];trials[root.name]=dict(status='complete',arm=arm,seed=seed,operating_point=op)
    update_state(trials=trials,pending=None)
    event(C,'control_complete',arm=arm,seed=seed,operating_point=op)

def evaluate():
    protocol=read(R/'protocol.json')['controlled'];prepared=read(C/'prepared.json')
    entries=[]
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            root=C/'trials'/f'{arm}_seed{seed}';done=read(root/'complete.json')
            for n,h in done['files'].items():require(sha(root/n)==h,'Control artifact changed')
            entries.append(dict(arm=arm,seed=seed,folder=str(root),operating_point=read(root/'operating_point.json'),
                complete_sha256=sha(root/'complete.json')))
    dest=C/'evaluation';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():require(read(dest/'frozen.json')==entries,'Evaluation freeze changed')
    else:save(dest/'frozen.json',entries)
    event(C,'controls_frozen_before_evaluation',trials=len(entries),test_previously_exposed=True)
    source_a=Path('/root/autodl-tmp/typea_v2_frozen_test_20261028/out/raw_cache_test')
    source_b=Path('/root/autodl-tmp/final_v31_20261028/final_test_20261029/typeb/cache_test')
    a_test=C/'evaluation_cache_test/typea66';project_cache(source_a,a_test,prepared['features'])
    results=[]
    for entry in entries:
        trial=Path(entry['folder']);op=entry['operating_point'];recipe=read(trial/'recipe.json')
        require(sha(trial/op['model'])==op['model_sha256'],'Model changed')
        b=xgb.Booster();b.load_model(trial/op['model']);b=b[:op['step']];b.set_param(dict(device='cpu',nthread=2))
        for label in ('typea','typeb'):
            c=read(prepared['sources'][label]['config'])
            for split in ('confirmation','test'):
                path=dest/trial.name/label/split;path.mkdir(parents=True,exist_ok=True)
                if (path/'complete.json').exists():
                    for n,h in read(path/'complete.json')['files'].items():require(sha(path/n)==h,'Evaluation changed')
                    results.append(read(path/'report.json'));continue
                cache=Path(prepared['sources'][label]['cache']) if split=='confirmation' else a_test if label=='typea' else source_b
                m=copy.deepcopy(read(cache/'manifest.json'));mode='val' if split=='confirmation' else 'test'
                if mode=='val':m['windows']['val']=c['confirmation_window']
                require(m['features']==prepared['features'],'Evaluation feature mismatch')
                cc=dict(c,active_cache=str(cache),num_threads=2,score_workers=4)
                table,stats=predict(b,cc,recipe,m,mode,op['threshold']);table.to_csv(path/'devices.csv',index=False)
                report=dict(arm=entry['arm'],seed=entry['seed'],target=label,split=split,operating_point=op,
                    metrics=metrics(table,op['threshold']),lead_summary=lead_summary(table),prediction_stats=stats,
                    window=m['windows'][mode],threshold_reselected=False,independent_blind_test=False)
                save(path/'report.json',report);save(path/'complete.json',dict(files={n:sha(path/n) for n in ['report.json','devices.csv']}))
                results.append(report);event(C,'control_evaluated',arm=entry['arm'],seed=entry['seed'],target=label,split=split,metrics=report['metrics'])
    save(dest/'summary.json',dict(reports=results,primary_target='typeb',all_seeds_reported=True,independent_blind_test=False))
    flat=[dict(arm=r['arm'],seed=r['seed'],target=r['target'],split=r['split'],step=r['operating_point']['step'],**r['metrics']) for r in results]
    pd.DataFrame(flat).to_csv(dest/'comparison.csv',index=False,encoding='utf-8-sig')
    event(C,'controlled_study_complete',trials=len(entries),reports=len(results),api_calls=0)
    update_state(status='complete',pending=None,test_opened=True,independent_blind_test=False)

if __name__=='__main__':
    seal=read(R/'controlled_ready.json')
    for name,h in seal['files'].items():require(sha(R/name)==h,'Control runner changed: '+name)
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='train':train(sys.argv[2],int(sys.argv[3]))
        elif action=='evaluate':evaluate()
        else:raise ValueError(action)
    except Exception as e:
        if C.exists():update_state(status='needs_attention',last_error=str(e))
        raise
