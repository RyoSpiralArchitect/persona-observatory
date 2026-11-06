#!/usr/bin/env python3
"""Post-hoc diagnostics of frozen Experiment 10 forecasts and annotations.

This does not fit predictors, generate text, assign labels, or revise the study.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_observatory.observations import derived_outcome
from persona_observatory.storage import require, write_json_exclusive, write_jsonl_exclusive

OBSERVERS = ("observer_a", "observer_b")
PREDICTORS = ("Global", "Evidence", "Setting", "Joint")
CONTRASTS = (("Joint", "Evidence"), ("Setting", "Global"), ("Joint", "Setting"),
             ("Joint", "Global"), ("Evidence", "Global"))


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def correlation(x, y):
    a, b = mean(x), mean(y)
    xx, yy = [v-a for v in x], [v-b for v in y]
    denominator = math.sqrt(math.fsum(v*v for v in xx) * math.fsum(v*v for v in yy))
    return math.fsum(u*v for u, v in zip(xx, yy)) / denominator if denominator else None


def analyze(study, output_dir):
    study, output_dir = Path(study), Path(output_dir)
    plan_path = output_dir / "analysis_plan.json"
    specification = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in specification["inputs"].items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, "Analysis source changed: " + name)
    require(not (output_dir / "diagnostics.json").exists() and not (output_dir / "d_disagreements.jsonl").exists(), "Analysis output already exists")
    with zipfile.ZipFile(study / "evidence_20261103.zip") as z:
        obj = lambda name: json.loads(z.read(name))
        rows = lambda name: [json.loads(line) for line in z.read(name).splitlines() if line.strip()]
        generations = rows("run/generations.jsonl")
        scores = obj("run/scores.json")
        labels = {o: {r["sample_id"]: r for r in rows("run/observers/"+o+"/labels.jsonl")} for o in OBSERVERS}
        settings = {r["setting_id"]: r for r in obj("run/inputs/settings.json")}
    fitted = json.loads((study/"fit.json").read_text())
    report = json.loads((study/"results_20261103.json").read_text())
    by_id = {r["sample_id"]: r for r in generations}
    indexed = {(r["sample_id"], r["observer"], r["predictor"]): r for r in scores["per_row"]}
    outcome = {o: {sid: derived_outcome(label) for sid, label in branch.items()} for o, branch in labels.items()}
    strata = sorted({r["evidence_stratum"] for r in generations})
    names = sorted(settings)
    groups = defaultdict(list)
    for r in generations:
        groups[(r["setting_id"], r["evidence_stratum"])].append(r["sample_id"])
    require(len(groups) == 24 and {len(v) for v in groups.values()} == {48}, "Expected a balanced 6 by 4 evaluation grid")
    require(len({r["question_id"] for r in generations}) == 96, "Question inventory differs")
    cells, marginals, decomposition, loss_increments, mode_counts = [], [], {}, [], []
    forecasts = {}
    for o in OBSERVERS:
        profiles = {p: fitted["profiles"][p][o] for p in PREDICTORS}
        dev_g = profiles["Global"]["n_direct"] / profiles["Global"]["n"]
        dev_e = {c["key"][0]: c["d"]/c["n"] for c in profiles["Evidence"]["cells"]}
        dev_s = {c["key"][0]: c["d"]/c["n"] for c in profiles["Setting"]["cells"]}
        dev_se = {tuple(c["key"]): c["d"]/c["n"] for c in profiles["Joint"]["cells"]}
        require({c["n"] for c in profiles["Joint"]["cells"]} == {26}, "Development grid differs")
        require(all(profiles[p]["rho"] == 1.0 for p in ("Evidence", "Setting", "Joint")), "Decomposition requires frozen half shrinkage")
        y = {sid: float(outcome[o][sid]["direct_event"]) for sid in by_id}
        eval_g = mean(y.values())
        eval_e = {e: mean(y[sid] for sid, r in by_id.items() if r["evidence_stratum"] == e) for e in strata}
        eval_s = {s: mean(y[sid] for sid, r in by_id.items() if r["setting_id"] == s) for s in names}
        eval_se = {key: mean(y[sid] for sid in ids) for key, ids in groups.items()}
        for key, ids in sorted(groups.items()):
            s, e = key
            p = {a: mean(indexed[(sid, o, a)]["gate_probability"] for sid in ids) for a in PREDICTORS}
            require(all(max(indexed[(sid,o,a)]["gate_probability"] for sid in ids)
                        - min(indexed[(sid,o,a)]["gate_probability"] for sid in ids) < 1e-12 for a in PREDICTORS), "Forecast varies within a cell")
            forecasts[(o,s,e)] = p
            cells.append({"observer":o,"setting":s,"stratum":e,"development_n":26,"development_questions":13,
                          "evaluation_n":48,"evaluation_questions":24,"development_d_rate":dev_se[key],
                          "evaluation_d_rate":eval_se[key],"probabilities":p,
                          "brier":{a:mean(indexed[(sid,o,a)]["gate_brier"] for sid in ids) for a in PREDICTORS}})
        for grouping, keys, old, new, n_dev, n_eval in (("global",["all"],{"all":dev_g},{"all":eval_g},624,1152),
                ("evidence",strata,dev_e,eval_e,156,288),("setting",names,dev_s,eval_s,104,192)):
            for k in keys:
                ids = [sid for sid,r in by_id.items() if grouping == "global" or
                       (grouping == "evidence" and r["evidence_stratum"] == k) or
                       (grouping == "setting" and r["setting_id"] == k)]
                marginals.append({"observer":o,"grouping":grouping,"key":k,"development_n":n_dev,"evaluation_n":n_eval,
                                  "development_d_rate":old[k],"evaluation_d_rate":new[k],
                                  "probabilities":{a:mean(indexed[(sid,o,a)]["gate_probability"] for sid in ids) for a in PREDICTORS}})
        b = {(s,e):0.5*(dev_e[e]-dev_g) for s,e in groups}
        interaction = {(s,e):0.5*(dev_se[(s,e)]-dev_s[s]-dev_e[e]+dev_g) for s,e in groups}
        eval_b = {(s,e):eval_e[e]-eval_g for s,e in groups}
        eval_interaction = {(s,e):eval_se[(s,e)]-eval_s[s]-eval_e[e]+eval_g for s,e in groups}
        pieces = {}
        for name, change, target in (("evidence_main",b,eval_b),("interaction",interaction,eval_interaction)):
            cost = mean(v*v for v in change.values())
            alignment = 2*mean(change[k]*target[k] for k in groups)
            pieces[name] = {"squared_change_cost":cost,"twice_outcome_alignment":alignment,"brier_increment":cost-alignment,
                            "development_evaluation_correlation":correlation(list(change.values()), [target[k] for k in change])}
        components_sum = math.fsum(p["brier_increment"] for p in pieces.values())
        recorded = mean(indexed[(sid,o,"Joint")]["gate_brier"]-indexed[(sid,o,"Setting")]["gate_brier"] for sid in by_id)
        require(abs(components_sum-recorded) < 1e-12, "Component decomposition failed")
        common_within = mean(q*(1-q) for q in eval_se.values())
        empirical_calibration = {a:mean((forecasts[(o,*k)][a]-eval_se[k])**2 for k in groups) for a in PREDICTORS}
        for a in PREDICTORS:
            require(abs(common_within+empirical_calibration[a]-mean(indexed[(sid,o,a)]["gate_brier"] for sid in by_id)) < 1e-12, "Within-cell decomposition failed")
        decomposition[o] = {"joint_minus_setting":pieces,"sum":components_sum,
                            "common_empirical_within_cell_dispersion":common_within,
                            "mean_squared_error_to_empirical_cell_rate":empirical_calibration,
                            "setting_development_evaluation_correlation":correlation([dev_s[s] for s in names],[eval_s[s] for s in names]),
                            "scope":"Algebra on finite observed cell rates; neither irreducible noise nor estimated population effects."}
        for left,right in CONTRASTS:
            change = {sid:indexed[(sid,o,left)]["gate_probability"]-indexed[(sid,o,right)]["gate_probability"] for sid in by_id}
            cost = mean(x*x for x in change.values())
            alignment = 2*mean(change[sid]*(y[sid]-indexed[(sid,o,right)]["gate_probability"]) for sid in by_id)
            actual = mean(indexed[(sid,o,left)]["gate_brier"]-indexed[(sid,o,right)]["gate_brier"] for sid in by_id)
            require(abs(cost-alignment-actual) < 1e-12, "Loss increment identity failed")
            loss_increments.append({"observer":o,"left":left,"right":right,"squared_change_cost":cost,"twice_residual_alignment":alignment,"delta":actual})
        for s in names:
            ids = [sid for sid,r in by_id.items() if r["setting_id"] == s]
            mode_counts.append({"observer":o,"setting":s,"n":len(ids),
                                "mode":dict(Counter(labels[o][sid]["answerer_answer_mode"] for sid in ids)),
                                "category":dict(Counter(outcome[o][sid]["category"] for sid in ids))})
    agreement = {}
    fields = ("answerer_answer_mode","answerer_judgment","answerer_stance","limit_owners","other_answer","character_description")
    def code(o,sid,k):
        value = labels[o][sid][k]
        return sorted(value) if k == "limit_owners" else value
    for field in fields:
        matches = sum(code(OBSERVERS[0],sid,field) == code(OBSERVERS[1],sid,field) for sid in by_id)
        agreement[field] = {"same":matches,"different":len(by_id)-matches,"rate":matches/len(by_id)}
    for field in ("direct_event","category","strict_event"):
        matches = sum(outcome[OBSERVERS[0]][sid][field] == outcome[OBSERVERS[1]][sid][field] for sid in by_id)
        agreement[field] = {"same":matches,"different":len(by_id)-matches,"rate":matches/len(by_id)}
    agreement["all_six_codes"] = {"same":sum(all(code(OBSERVERS[0],sid,k)==code(OBSERVERS[1],sid,k) for k in fields) for sid in by_id)}
    discordant = [sid for sid in sorted(by_id) if outcome[OBSERVERS[0]][sid]["direct_event"] != outcome[OBSERVERS[1]][sid]["direct_event"]]
    require(len(discordant)==report["observer_d_disagreements"]==28, "D-disagreement count differs")
    disagreements = [{**{k:by_id[sid][k] for k in ("sample_id","question_id","setting_id","evidence_stratum","requested_answerer","question","completion","stop_reason")},
                      "original_labels":{o:labels[o][sid] for o in OBSERVERS}} for sid in discordant]
    sensitivity = []
    length_ids = {sid for sid,r in by_id.items() if r["stop_reason"] == "length"}
    require(len(length_ids)==3, "Length-stop inventory differs")
    for left,right in CONTRASTS:
        minima, maxima, qblocks = [], [], defaultdict(list)
        for sid,r in by_id.items():
            candidates = (0,1) if sid in discordant else (int(outcome[OBSERVERS[0]][sid]["direct_event"]),)
            alternatives = [mean((indexed[(sid,o,left)]["gate_probability"]-y)**2-
                                 (indexed[(sid,o,right)]["gate_probability"]-y)**2 for o in OBSERVERS) for y in candidates]
            minima.append(min(alternatives));maxima.append(max(alternatives))
            delta = mean(indexed[(sid,o,left)]["gate_brier"]-indexed[(sid,o,right)]["gate_brier"] for o in OBSERVERS)
            qblocks[r["question_id"]].append((sid,delta))
        qdelta = {q:mean(v for _,v in values) for q,values in qblocks.items()}
        actual = mean(qdelta.values())
        expected = next(c["ab_mean"]["gate_brier"] for c in report["contrasts"] if (c["left"],c["right"])==(left,right))
        require(abs(actual-expected)<1e-12, "Published comparison differs")
        loo = [mean(v for k,v in qdelta.items() if k!=q) for q in qdelta]
        agreed_contribution = math.fsum(v for values in qblocks.values() for sid,v in values if sid not in discordant)/len(by_id)
        discordant_contribution = math.fsum(v for values in qblocks.values() for sid,v in values if sid in discordant)/len(by_id)
        sensitivity.append({"left":left,"right":right,"recorded_delta":actual,
                            "common_D_on_disagreements_range":[mean(minima),mean(maxima)],
                            "recorded_agreement_subset_contribution":agreed_contribution,
                            "recorded_disagreement_subset_contribution":discordant_contribution,
                            "leave_one_question_out_range":[min(loo),max(loo)],
                            "without_length_stops_question_weighted":mean(mean(v for sid,v in values if sid not in length_ids) for values in qblocks.values()),
                            "question_signs":{"negative":sum(v < -1e-12 for v in qdelta.values()),"positive":sum(v>1e-12 for v in qdelta.values()),"tie":sum(abs(v)<=1e-12 for v in qdelta.values())}})
    cv_eval = []
    for p in PREDICTORS:
        cv = next(c for c in fitted["cv"] if c["algorithm"]==p and c["rho"]==fitted["selection"][p])
        cv_eval.append({"predictor":p,"selected_rho":cv["rho"],"development_cv_brier":cv["gate_brier"],
                        "evaluation_brier":mean(s["gate_brier"] for s in scores["summary"] if s["predictor"]==p)})
    result = {"status":"POST_HOC_DESCRIPTIVE_DIAGNOSTICS","created_utc":datetime.now(timezone.utc).isoformat(),
              "analysis_plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "generations":len(by_id),"questions":96,"marginals":marginals,"cells":cells,
              "loss_increment_decomposition":loss_increments,"joint_setting_components":decomposition,
              "cv_vs_evaluation":cv_eval,"agreement":agreement,"mode_category_counts":mode_counts,"sensitivity":sensitivity,
              "settings":settings,"scope":"Post-hoc explanation of frozen measurements. No new labels, refits, significance claims, or revised primary decision. Sensitivity bounds assume all agreed D labels are correct; shared observer error remains unbounded."}
    write_jsonl_exclusive(output_dir/"d_disagreements.jsonl",disagreements)
    write_json_exclusive(output_dir/"diagnostics.json",result)
    return {k:result[k] for k in ("status","generations","questions","cv_vs_evaluation","agreement","sensitivity","joint_setting_components")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.study_dir,args.output_dir),ensure_ascii=False,indent=2))
