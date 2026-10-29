"""Deployment step and threshold for a completed trial.

Why: the selection window holds few faults (TypeB: 10), so a step and threshold tuned there
are optimistic. Pooled out-of-fold CV covers every TRAIN device and several times more fault
units, so cv_oof takes the CV-selected step and threshold and applies them to the full-TRAIN
model truncated to that step. The fold models saw 80% of the devices; the threshold is reused
as a score, which assumes comparable score scales and is recorded as such.
"""
import argparse
import json
from pathlib import Path
from common import read, sha, require


def choose(folder, mode='cv_oof'):
    folder = Path(folder)
    r = read(folder/'result.json')
    selection = dict(source='selection_window', model='model.ubj', model_sha256=r['model_sha256'],
                     step=r['best_step'], threshold=r['metrics']['threshold'])
    if mode == 'selection_window':
        return dict(selection, reason='Configured operating_point=selection_window')
    require(mode == 'cv_oof', 'Unknown operating point mode: '+str(mode))
    cv = folder/'cv'/'oof_complete.json'
    full = folder/'model_full.ubj'
    if r['recipe'].get('rerank', 'none') != 'none':
        return dict(selection, reason='Two-branch models keep their composite selection; CV step is not a tree count')
    if not cv.exists():
        return dict(selection, reason='Trial has no CV run (screened out or legacy acceptance)')
    if not full.exists():
        return dict(selection, reason='No model_full.ubj (trial trained before this release)')
    o = read(cv)
    from rerank import load_model
    trees = load_model(full).num_boosted_rounds()
    if o['best_step'] > trees:
        return dict(selection, reason=f'CV step {o["best_step"]} exceeds the {trees} saved rounds')
    return dict(source='cv_oof', model='model_full.ubj', model_sha256=sha(full), step=int(o['best_step']),
                threshold=float(o['cv_metrics']['threshold']), cv_metrics=o['cv_metrics'],
                oof_sha256=o['oof_sha256'], fault_units=o['fault_units'],
                note='Threshold and step from pooled out-of-fold CV; applied to the full-TRAIN model at that step.')


def load(folder, op, threads=1):
    from rerank import load_model
    path = Path(folder)/op['model']
    require(sha(path) == op['model_sha256'], 'Operating-point model changed: '+op['model'])
    model = load_model(path, threads)
    if op['source'] != 'cv_oof':
        return model
    model = model[:op['step']]
    model.set_param(dict(device='cpu', nthread=threads))
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Print the deployment operating point of a trial folder')
    p.add_argument('--trial', required=True, help='trials/<id>/attempt_01')
    p.add_argument('--mode', default='cv_oof', choices=['cv_oof', 'selection_window'])
    a = p.parse_args()
    print(json.dumps(choose(a.trial, a.mode), ensure_ascii=False, indent=2))
