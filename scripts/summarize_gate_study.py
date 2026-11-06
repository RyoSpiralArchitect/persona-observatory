#!/usr/bin/env python3
"""Read a completed locked study; report all prespecified paired gate contrasts."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from persona_observatory.storage import read_json, read_jsonl, require, sha_file, write_json_exclusive, now
from persona_observatory.runs import verify_run
from persona_observatory.observations import validate_observer
from persona_observatory.evaluation import score_rows

CONTRASTS=(('Joint','Evidence'),('Setting','Global'),('Joint','Setting'),('Joint','Global'),('Evidence','Global'))
METRICS=('gate_brier','gate_nll','category_brier','category_nll')


def summarize(run_dir,output):
    run=Path(run_dir)
    verify_run(run,require_complete=True)
    lock=read_json(run/'score_lock.json')
    require(lock['scores_sha256']==sha_file(run/'scores.json'),'Score artifact changed')
    require(lock['run_lock_sha256']==sha_file(run/'run_lock.json') and
            lock['completion_sha256']==sha_file(run/'completion.json') and
            lock['forecast_lock_sha256']==sha_file(run/'forecast_lock.json'),'Score provenance differs')
    labels={}
    for o in ('observer_a','observer_b'):
        validate_observer(run,o)
        require(lock['observation_locks'][o]==sha_file(run/'observers'/o/'observation_lock.json'),'Observation lock changed')
        labels[o]=read_jsonl(run/'observers'/o/'labels.jsonl')
    recomputed=score_rows(read_jsonl(run/'forecasts.jsonl'),labels)
    require(recomputed==read_json(run/'scores.json'),'Recomputed scores differ')
    plan=read_jsonl(run/'plan.jsonl');by_id={r['sample_id']:r for r in plan}
    require(len(plan)==1152 and len({r['question_id'] for r in plan})==96,'This summary is fixed to the 96-question MPS study')
    scores={(r['sample_id'],r['observer'],r['predictor']):r for r in recomputed['per_row']}
    summaries=[];question_differences=[]
    for left,right in CONTRASTS:
        branches={}
        for observer in ('observer_a','observer_b'):
            blocks=defaultdict(lambda:defaultdict(list))
            for sid,p in by_id.items():
                a,b=scores[(sid,observer,left)],scores[(sid,observer,right)]
                for metric in METRICS:blocks[p['question_id']][metric].append(a[metric]-b[metric])
                require(abs((a['category_nll']-b['category_nll'])-(a['gate_nll']-b['gate_nll']))<1e-10,
                        'Shared composition NLL identity failed')
            means={q:{m:math.fsum(v)/len(v) for m,v in values.items()} for q,values in blocks.items()}
            branches[observer]={m:math.fsum(v[m] for v in means.values())/len(means) for m in METRICS}
            question_differences.extend({'left':left,'right':right,'observer':observer,'question_id':q,**v} for q,v in sorted(means.items()))
        mean={m:math.fsum(branches[o][m] for o in branches)/2 for m in METRICS}
        summaries.append({'left':left,'right':right,'observers':branches,'ab_mean':mean})
    primary=summaries[0];baseline=next(r for r in summaries if r['left']=='Joint' and r['right']=='Global')
    delta=primary['ab_mean']['gate_brier']
    gates={'absolute_brier_improvement':delta<=-0.005,
           'both_observers_improve':all(v['gate_brier'] < -1e-12 for v in primary['observers'].values()),
           'not_worse_than_global':baseline['ab_mean']['gate_brier']<=1e-12,
           'gate_nll_not_worse':primary['ab_mean']['gate_nll']<=1e-12}
    questions=sorted({r['question_id'] for r in plan})
    qdelta={q:math.fsum(r['gate_brier'] for r in question_differences if r['left']=='Joint' and r['right']=='Evidence' and r['question_id']==q)/2 for q in questions}
    loo=[math.fsum(v for k,v in qdelta.items() if k!=q)/(len(questions)-1) for q in questions]
    strata={r['question_id']:r['evidence_stratum'] for r in plan}
    stratum_means={s:math.fsum(qdelta[q] for q in questions if strata[q]==s)/sum(strata[q]==s for q in questions) for s in sorted(set(strata.values()))}
    setting_means={}
    for setting in sorted({r['setting_id'] for r in plan}):
        values=[scores[(r['sample_id'],o,'Joint')]['gate_brier']-scores[(r['sample_id'],o,'Evidence')]['gate_brier']
                for r in plan if r['setting_id']==setting for o in ('observer_a','observer_b')]
        setting_means[setting]=math.fsum(values)/len(values)
    disagree=sum(scores[(sid,'observer_a','Global')]['direct_event']!=scores[(sid,'observer_b','Global')]['direct_event'] for sid in by_id)
    result={'status':'SCORED_AND_RECOMPUTED','summarized_utc':now(),'generations':1152,'questions':96,
            'primary_descriptive_support':delta < -1e-12,'advance':all(gates.values()),'advance_gates':gates,
            'contrasts':summaries,'question_differences':question_differences,
            'primary_question_signs':{'negative':sum(v < -1e-12 for v in qdelta.values()),'tie':sum(abs(v)<=1e-12 for v in qdelta.values()),'positive':sum(v>1e-12 for v in qdelta.values())},
            'primary_leave_one_question_out_range':[min(loo),max(loo)],'primary_strata':stratum_means,'primary_settings':setting_means,
            'observer_d_disagreements':disagree,'observer_d_disagreement_rate':disagree/len(plan),
            'score_lock_sha256':sha_file(run/'score_lock.json'),
            'scope':'Descriptive authored-bank comparison; no significance, causal personality or cross-device identity claim.'}
    write_json_exclusive(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.run_dir,a.output)
    print({k:result[k] for k in ('status','primary_descriptive_support','advance','advance_gates')})
