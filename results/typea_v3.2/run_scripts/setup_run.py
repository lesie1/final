"""New-session deployment. The reviewed runtime and all old sessions stay immutable."""
import copy, json, sys, time, zipfile
from pathlib import Path

R = Path(__file__).resolve().parent
OLD = Path('/root/autodl-tmp/final_v31_20261028')
COMMIT = '8ebafc8d311e734d2089df9d1106c962e2026b66'
runtime = R/'runtime'
if not runtime.exists():
    runtime.mkdir()
    with zipfile.ZipFile(R/'runtime.zip') as z:
        for n in z.namelist():
            assert (runtime/n).resolve().is_relative_to(runtime.resolve())
        z.extractall(runtime)
sys.path.insert(0, str(runtime)); sys.dont_write_bytecode = True
from common import read, save, sha, digest, code_hashes, require, event
from agent import Loop
from settings import validate_config

def initialize():
    if (R/'ready.json').exists():
        print('Already initialized; existing sessions retained'); return
    prior = {}
    new_policy = dict(accept_test='probability', accept_probability=0.9,
                      accept_require_net_hits=False, operating_point='cv_oof')
    for label in ('typea','typeb'):
        source = (OLD/label/'session').resolve()
        task = R/label; session = task/'session'
        require(not (session/'state.json').exists(), 'Partial initialization: inspect '+label)
        task.mkdir(exist_ok=True); session.mkdir(exist_ok=True)
        original = read(source/'config.json'); old_state = read(source/'state.json')
        require(old_state['status']=='trial_budget_complete', 'Source is still active')
        prior[label] = dict(path=str(source), seals={n:sha(source/n) for n in
            ['config.json','state.json','events.jsonl','mapping.json','code_seal.json']})
        c = copy.deepcopy(original)
        c.update(new_policy)
        c.update(output_root=str(session), workspace=str(task/'workspace'), max_trials=13,
                 active_cache=str(Path(original.get('active_cache') or source/'cache').resolve()))
        validate_config(c)
        mapping = read(source/'mapping.json')
        save(session/'config.json',c); save(session/'mapping.json',mapping)
        save(session/'code_seal.json',code_hashes())
        save(session/'inventory.json',dict(fingerprint=old_state['dataset'],reused_source=str(source)))
        save(session/'state.json',dict(version=1,status='ready',config_hash=digest(c),
            dataset=old_state['dataset'],mapping_hash=digest(mapping),calls=0,tokens=0,
            trials=[],best=None,pending=None,analyze_next=None,diagnostics={},feature_jobs=0,
            experience=None,experience_tried=True,last_failure=None))
        changed = {k:dict(before=original.get(k),after=v) for k,v in c.items() if original.get(k)!=v}
        save(session/'bootstrap.json',dict(source=str(source),source_seals=prior[label]['seals'],
            changed_config=changed,model_weights_imported=False,test_opened=False,
            note='Retain original search budgets/sampling; only v3.2 policy and output paths change. Historical validation and TEST have prior exposure.'))
        save(task/'isolation.json',dict(forbidden_reads=[original['data_dir'],str(OLD/'final_test_20261029'),
            str(OLD/'rejected_comparison_20261029'),str(OLD/'typea_to_typeb_20261029')],
            readonly_roots=[str(OLD),str(source),c['active_cache'],str(runtime)]))
        Loop(session).prepare()
        event(session,'v32_session_initialized',source=str(source),max_trials=13,policy=new_policy,
              source_sampling_preserved=True,models_imported=False,test_opened=False)
        save(task/'ready.json',dict(session=str(session),max_trials=13))
    save(R/'source_preservation.json',prior)
    save(R/'release.json',dict(commit=COMMIT,repo='lesie1/final',runtime_hashes=code_hashes(),
                              runtime_modified=False,local_tests_passed=71))
    recipe=read(Path(prior['typeb']['path'])/'trials/trial_001/attempt_01/plan.json')
    recipe['base_weight']='device'
    plan=dict(version='v32_common66_20261029',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        search=dict(tasks=['typea','typeb'],max_trials_each=13,policy=new_policy),
        controlled=dict(arms=['B_only','A_only','A_plus_B','A_small'],seeds=[42,43,44],max_trials=12,
            recipe=recipe,rounds=400,eval_every=50,folds=5,fold_seed=20261027,
            feature_policy='Exact 66 TypeB names/order projected from A; preserve source values and NaNs; no added features.',
            sampling='All mature TRAIN event rows; no legacy negative subsampling in any controlled arm.',
            weights='Each included TRAIN device totals one base weight; positive class ratio recomputed from weighted TRAIN only.',
            small_A='Sample A positive/nonpositive TRAIN devices to match B TRAIN device counts; preserve each selected device history. Three seeds, no validation-informed sampling.',
            cv='Same B held-out devices and TRAIN periods in all arms. A-only models fit no B rows; B labels are used only for B-target OOF operating-point calibration. A+B fold fits all A and only non-held-out B.',
            operating_point='Max pooled B TRAIN OOF device F1 across fixed steps; then fewer FP, earlier step; threshold frozen before later VAL or TEST.',
            scoring='Primary B; also A and pooled counts at the same B-calibrated operating point. Report all seeds, no TEST winner selection.'),
        test_previously_exposed=True,independent_blind_test=False,
        source_label_policy='Keep each source observed_start/end and failure metadata unchanged. Unknown follow-up never becomes negative.',
        publication='New sessions, events, models, provenance and reports on GitHub; omit credentials, raw data and regenerable large caches; document exclusions.')
    save(R/'protocol.json',plan)
    save(R/'ready.json',dict(status='ready',source_preserved=True,search_trials_max=26,controlled_trials=12))
    print('READY',str(R),flush=True)

if __name__=='__main__':initialize()
