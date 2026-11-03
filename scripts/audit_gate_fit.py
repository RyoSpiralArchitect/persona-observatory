#!/usr/bin/env python3
"""Recompute gate fits directly from archived labels using a separate formula path.

This is arithmetic verification, not independent semantic or research review.
It deliberately does not import persona_observatory.predictors.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(l) for l in Path(p).read_text().splitlines()]
def close(a,b):
    if abs(a-b)>1e-12: raise ValueError(f'Arithmetic mismatch: {a} vs {b}')


def audit(design,root):
    design=Path(design);root=Path(root)
    lock=read(design/'fit_lock.json')
    assert lock['generation_performed'] is False
    for name,h in lock['output_hashes'].items(): assert digest(design/name)==h,name
    for name,h in lock['source_hashes'].items(): assert digest(root/name)==h,name
    assert digest(root/'archives/experiment_09_20261103.zip')==lock['archive_sha256']
    rows=[];labels={}
    with zipfile.ZipFile(root/'archives/experiment_09_20261103.zip') as z:
        def original(name):
            raw=z.read('persona_shrinkage_pilot/'+name)
            assert hashlib.sha256(raw).hexdigest()==lock['source_inputs'][name]
            return [json.loads(l) for l in raw.splitlines()]
        for row in original('development/records.jsonl'):
            r=row['source_record']
            rows.append({k:r[k] for k in ('question_id','setting_id','evidence_stratum')}|
                        {'sample_id':row['sample_id'],'fold':row['development_fold'],'source_role':'v9_development'})
        for r in original('data/target_generations.jsonl'):
            rows.append({k:r[k] for k in ('sample_id','question_id','setting_id','evidence_stratum')}|
                        {'fold':r['topic_index']+7,'source_role':'v9_evaluation_reassigned_to_development'})
        for observer in ('observer_a','observer_b'):
            raw=original('development/labels_'+observer+'.jsonl')+original('observers/target/'+observer+'/labels.jsonl')
            assert raw==lines(design/('labels_'+observer+'.jsonl'))
            labels[observer]={r['sample_id']:r['answerer_judgment'] if r['answerer_answer_mode'] in ('direct','both') else 'no_direct' for r in raw}
    assert rows==lines(design/'development.jsonl')
    assert set(Counter(r['question_id'] for r in rows).values())=={12}
    categories=['support','reject','withhold','mixed']
    def distribution(train,observer,algorithm,rho,target):
        y=labels[observer];nd=sum(y[r['sample_id']]!='no_direct' for r in train)
        g=(nd+0.5)/(len(train)+1)
        fields={'Global':(), 'Evidence':('evidence_stratum',),'Setting':('setting_id',),'Joint':('setting_id','evidence_stratum')}[algorithm]
        local=[r for r in train if all(r[k]==target[k] for k in fields)]
        r=g
        if algorithm!='Global' and rho!='inf' and local:
            empirical=sum(y[x['sample_id']]!='no_direct' for x in local)/len(local)
            weight=1/(1+rho)
            r=weight*empirical+(1-weight)*g
        composition=[(sum(y[x['sample_id']]==k for x in train)+0.5)/(nd+2) for k in categories]
        return [r*h for h in composition]+[1-r]
    fit=read(design/'fit.json');chosen={};cv_checks=0
    for algorithm in ('Global','Evidence','Setting','Joint'):
        candidates=[]
        for rho in (['inf'] if algorithm=='Global' else [0.25,1,4,'inf']):
            losses=[]
            for fold in range(13):
                train=[r for r in rows if r['fold']!=fold]
                test=[r for r in rows if r['fold']==fold]
                for observer in labels:
                    for target in test:
                        p=distribution(train,observer,algorithm,rho,target)
                        losses.append((1-p[-1]-int(labels[observer][target['sample_id']]!='no_direct'))**2)
            score=math.fsum(losses)/len(losses)
            recorded=next(c for c in fit['cv'] if c['algorithm']==algorithm and c['rho']==rho)
            close(score,recorded['gate_brier']);cv_checks+=1
            candidates.append((rho,score))
        minimum=min(s for _,s in candidates)
        chosen[algorithm]=max((r for r,s in candidates if s<=minimum+1e-12),key=lambda r:math.inf if r=='inf' else r)
    assert chosen==fit['selection']
    targets={r['sample_id']:r for r in lines(design/'targets.jsonl')}
    forecasts=lines(design/'forecasts.jsonl')
    assert len(forecasts)==len(targets)*8
    seen=set()
    for f in forecasts:
        key=f['sample_id'],f['observer'],f['predictor'];assert key not in seen;seen.add(key)
        expected=distribution(rows,f['observer'],f['predictor'],chosen[f['predictor']],targets[f['sample_id']])
        for a,b in zip(expected,f['category_probabilities']):close(a,b)
    return {'status':'ARITHMETIC_VERIFIED','fit_lock_sha256':digest(design/'fit_lock.json'),'cv_candidates_checked':cv_checks,
            'forecast_rows_checked':len(forecasts),'selection':chosen,
            'scope':'Separate formula implementation, same assistant author. Original labels unchanged; no new inference or independent semantic validation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--design-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.design_dir,Path(__file__).resolve().parents[1])
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
