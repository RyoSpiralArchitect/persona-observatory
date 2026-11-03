#!/usr/bin/env python3
"""Freeze development-only fits and target forecasts before any new inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from persona_observatory.predictors import select_and_fit, forecast_targets
from persona_observatory.storage import read_json, require, sha_file, write_json_exclusive, write_jsonl_exclusive, now
from persona_observatory.study import load_study, design_rows
from verify_archive import verify_repository


def prepare(config_path, protocol_path, output):
    root=Path(__file__).resolve().parents[1]
    verify_repository(root)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config,questions,settings,manifest=load_study(config_path)
    protocol=read_json(protocol_path)
    require(protocol['status'] in {'FROZEN_BEFORE_GENERATION','PREDICTOR_FROZEN_OBSERVER_ASSIGNMENT_PENDING'},
            'Predictor rules must be frozen before fitting')
    require(protocol['primary_contrast']==['Joint','Evidence'] and protocol['primary_metric']=='gate_brier','Unexpected protocol')
    require(len(questions)==protocol['questions'] and len(settings)==6 and config['replicates']==2,'Design budget differs')
    rows=[];labels={};input_hashes={}
    from hashlib import sha256
    with zipfile.ZipFile(root/'archives/experiment_09_20261103.zip') as z:
        def load(name):
            raw=z.read('persona_shrinkage_pilot/'+name);input_hashes[name]=sha256(raw).hexdigest()
            return [json.loads(line) for line in raw.splitlines()] if name.endswith('.jsonl') else json.loads(raw)
        dev=load('development/records.jsonl');old_target=load('data/target_generations.jsonl')
        prior_questions=load('development/source_question_bank.json')+load('question_bank.json')
        for r in dev:
            source=r['source_record']
            rows.append({k:source[k] for k in ('question_id','setting_id','evidence_stratum')}|
                        {'sample_id':r['sample_id'],'fold':r['development_fold'],'source_role':'v9_development'})
        for r in old_target:
            rows.append({k:r[k] for k in ('sample_id','question_id','setting_id','evidence_stratum')}|
                        {'fold':r['topic_index']+7,'source_role':'v9_evaluation_reassigned_to_development'})
        for o in ('observer_a','observer_b'):
            labels[o]=load('development/labels_'+o+'.jsonl')+load('observers/target/'+o+'/labels.jsonl')
    require(len(rows)==624 and len({r['sample_id'] for r in rows})==624,'Unexpected development count/duplicates')
    require(len({r['question_id'] for r in rows})==52,'Unexpected development questions')
    require(set(Counter((r['question_id'],r['setting_id']) for r in rows).values())=={2},'Unbalanced development cells')
    require(set(Counter((r['fold'],r['evidence_stratum']) for r in rows).values())=={12},'Unbalanced folds')
    require(not ({q['topic'] for q in questions}&{q['topic'] for q in prior_questions}),'Existing topic string reused')
    require(not ({q['question_id'] for q in questions}&{q['question_id'] for q in prior_questions}),'Existing question ID reused')
    require(len({q['topic'] for q in questions})==len(questions),'Duplicate target topics')
    require(set(Counter(q['evidence_stratum'] for q in questions).values())=={protocol['questions']//4},'Unbalanced target strata')
    targets=design_rows(config,questions,settings)
    fitted=select_and_fit(rows,labels)
    forecasts=forecast_targets(fitted,targets,rows)
    write_jsonl_exclusive(output/'development.jsonl',rows)
    for o in labels:write_jsonl_exclusive(output/('labels_'+o+'.jsonl'),labels[o])
    write_jsonl_exclusive(output/'targets.jsonl',targets)
    write_jsonl_exclusive(output/'forecasts.jsonl',forecasts)
    write_json_exclusive(output/'fit.json',fitted)
    write_json_exclusive(output/'protocol.json',protocol)
    write_json_exclusive(output/'config.json',config)
    write_json_exclusive(output/'questions.json',questions)
    write_json_exclusive(output/'settings.json',settings)
    write_json_exclusive(output/'model_manifest.json',manifest)
    source={p.relative_to(root).as_posix():sha_file(p) for p in (root/'src/persona_observatory').rglob('*.py')}
    source['scripts/prepare_gate_study.py']=sha_file(Path(__file__))
    write_json_exclusive(output/'fit_lock.json',{'schema_version':'persona-observatory-fit-lock-v1','locked_utc':now(),
        'archive_sha256':sha_file(root/'archives/experiment_09_20261103.zip'),'source_inputs':input_hashes,
        'source_hashes':source,'original_protocol_sha256':sha_file(protocol_path),
        'output_hashes':{p.name:sha_file(p) for p in output.iterdir() if p.is_file()},
        'generation_performed':False,'development_outputs':624,'target_outputs':len(targets),
        'forecast_rows':len(forecasts),'scope':'Development selection and fixed forecasts; no semantic novelty guarantee or new subject inference.'})
    return {'status':'FORECASTS_FROZEN','target_outputs':len(targets),'forecast_rows':len(forecasts),'selection':fitted['selection'],
            'output':str(output),'fit_lock_sha256':sha_file(output/'fit_lock.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.config,a.protocol,a.output),indent=2))
