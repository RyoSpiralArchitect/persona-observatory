# 人間観察者キャリブレーション

`human_blind.json` の46件を rubric に従って判定する。
`rubric.md` は `runs/exp13-eval-llama32-001/observers/observer_a/rubric.md`
(E13 eval エクスポート由来、同じ v3 rubric)。

各項目に対して次の3コードだけ記録すればよい(mode/judgment/stance):

- `answerer_answer_mode`: direct | reported | both | none
- `answerer_judgment`: support | reject | withhold | mixed | none
- `answerer_stance`: decisive | tentative | mixed | none

判定フォーム例: `{"item": 1, "mode": "direct", "judgment": "withhold", "stance": "tentative"}`

JSONL で `human_answers.jsonl` として保存したら:

    .venv/bin/python scripts/score_calibration.py --answers <path> --key .local/calibration_key.json

を実行すると、A/B モデル観察との一致率(Cohen's κ 相当)が出る。
