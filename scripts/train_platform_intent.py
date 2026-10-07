"""Train human next-platform intent; temporal diagnostics are not live efficiency."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.demo_policy import features,PlatformIntent
from autofarm.realtime.semantic import atomic_json


def labelled_rows(rows):
    groups=[]
    for r in rows:
        if r.get('reason') or not r.get('floor_id') or not r.get('player') or abs(r['player']['vy'])>=70:continue
        if groups and groups[-1]['id']==r['floor_id'] and r['t']-groups[-1]['end']<=.45:
            groups[-1]['end']=r['t'];groups[-1]['samples']+=1
        else:groups.append(dict(id=r['floor_id'],start=r['t'],end=r['t'],samples=1))
    stable=[g for g in groups if g['samples']>=3 and g['end']-g['start']>=.25]
    labelled=[]
    for r in rows:
        if (not r.get('foreground') or not r.get('key_focus') or r.get('key_sample_age',1)>.05
                or r.get('reason') or not r.get('player') or not r.get('floor_id')
                or abs(r['player']['vy'])>=70 or r['t']>rows[-1]['t']-4):continue
        arrival=next((g for g in stable if r['t']+.15<g['start']<=r['t']+4 and g['id']!=r['floor_id']),None)
        keys=set(r.get('keys') or [])
        # Learn navigational choices, not the duration of a held attack key.
        if 'shift' in keys or (arrival is None and not keys.intersection({'left','right','up','down','alt'})):continue
        labelled.append((r,arrival['id'] if arrival else r['floor_id']))
    return labelled,stable


def train(folder,out):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((folder/'report.json').read_text())
    session=json.loads((Path(provenance['demo'])/'session.json').read_text())
    if provenance['mode']!='OFFLINE_HUMAN_STATE_EXTRACTION' or session['mode']!='HUMAN_DEMONSTRATION_READ_ONLY':
        raise ValueError('Human training requires a human demonstration source')
    rows=[json.loads(x) for x in (folder/'states.jsonl').read_text().splitlines()]
    ids=sorted({p['id'] for r in rows for p in r['platforms'] if not p['id'].startswith('local_')})
    labelled,stable=labelled_rows(rows)
    data=[(r,target,features(r,ids)) for r,target in labelled if target in ids]
    data=[v for v in data if v[2] is not None]
    train_rows=[v for v in data if v[0]['t']<420]
    validation=[v for v in data if v[0]['t']>=450]
    if len(train_rows)<100 or len(validation)<30:raise ValueError('Insufficient temporally separated navigation samples')
    def fit(items):
        model=RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=8,
                                     max_features=.7,class_weight='balanced_subsample',random_state=25,n_jobs=2)
        model.fit([v[2] for v in items],[v[1] for v in items]);return model
    diagnostic=fit(train_rows);truth=[v[1] for v in validation];pred=diagnostic.predict([v[2] for v in validation])
    classes=sorted(set(truth)|set(pred));current=[v[0]['floor_id'] for v in validation]
    # The exported candidate uses all labelled frames only after the temporal
    # diagnostic is saved. It still needs new-session closed-loop evaluation.
    model=fit(data)
    trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_
        trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
            feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,:].tolist()))
    artifact=dict(version=1,platform_ids=ids,classes=model.classes_.tolist(),trees=trees,
        source_sha256=hashlib.sha256((folder/'states.jsonl').read_bytes()).hexdigest(),
        trained_samples=len(data),role='candidate platform suggestions only; no key commands',
        feature_schema='relative_actor_platform_enemy_v1')
    pure=PlatformIntent(artifact)
    # JSON inference must numerically agree with the fitted model before use.
    probes=data[::max(1,len(data)//100)]
    for r,_,x in probes:
        exported=dict(pure.predict(r));expected=model.predict_proba([x])[0]
        if max(abs(exported[k]-p) for k,p in zip(model.classes_,expected))>1e-8:raise ValueError('Export disagrees with fitted model')
    report=dict(mode='OFFLINE_SAME_DEMONSTRATION_TEMPORAL_DIAGNOSTIC',train_seconds=[0,420],validation_seconds=[450,600],
        train_samples=len(train_rows),validation_samples=len(validation),all_samples=len(data),
        accuracy=float(accuracy_score(truth,pred)),balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
        remain_on_current_floor_accuracy=float(accuracy_score(truth,current)),
        class_counts=dict(Counter(v[1] for v in data)),confusion_classes=classes,
        confusion_matrix=confusion_matrix(truth,pred,labels=classes).tolist(),stable_arrivals=stable,
        live_exp_per_minute=None,independent_evaluation=False,
        warning='Entire demonstration was already reviewed during development. Temporal diagnostic is not a held-out skill claim. Export trained on all labelled samples; live shadow/closed-loop validation remains required.')
    atomic_json(out/'policy.json',artifact);atomic_json(out/'report.json',report)
    return {k:v for k,v in report.items() if k not in ('confusion_matrix','stable_arrivals')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('states');p.add_argument('output');a=p.parse_args()
    print(json.dumps(train(a.states,a.output),indent=2))
