# 実験10 MPS：予測固定済み・観察担当待ち

**新規評価の生成と採点は未実行。** 旧データを使う予測器選択・予測保存・
別計算による算術照合・MPS run の準備まで完了している。

[初期案](../../docs/experiment_10_design_20261103.md)から、主試験を CPU から
PyTorch/MPS へ変更し、新規問題数を48から96へ増やした。
これは評価出力を得る前の設計変更である。実験9の共同主基準未達と、
H_gate の記述的な最良順位はそのまま保持する。

| 項目 | 固定した内容 |
|---|---|
| 対象 | 同じ SmolLM2-360M-Instruct、revision `a10cc1512eabd3dde888204e902eca88bddb4951` |
| 主実行系 | Transformers 4.57.6 / Torch 2.15.0、MPS float32、CPU sampling、batch 8 |
| 開発 | 旧52題・624生成。元 A/B ラベルを変更せず使用 |
| 新規評価 | 96題（4証拠層×24題）×6既知条件×2反復＝1,152生成 |
| 生成上限 | 各96 token、計110,592 token。一回の試行 |
| 予測 | 4候補×2観察枝×1,152生成＝9,216行 |
| 主比較 | Joint−Evidence の D 二値 Brier |
| 観察 | 別々の盲検 A/B、全文を読んだ意味ラベル。担当方法は選択待ち |

予測器の規則は [protocol.json](protocol.json)、新規問題は
[questions.json](questions.json)、runner 設定は
[configs/experiment_10_mps.json](../../configs/experiment_10_mps.json)。
問題は設計を担当する assistant が作成し、過去60題と題材・操作の重複を点検した。
独立した人間の新規性評価を受けたという意味ではない。証拠構造のテンプレートは
意図的に共通で、自然会話を無作為抽出した評価ではない。

13-fold の開発 D Brier で Evidence、Setting、Joint はすべて `rho=1` を選んだ。
Global は `rho=inf`。A/B の件数は別々に集計し、各候補の縮約強度を共通にした。
この選択結果や CV スコアを新規評価の成功証拠には使わない。

保存した [fit.json](fit.json)、[forecasts.jsonl](forecasts.jsonl)、
[fit_lock.json](fit_lock.json)と [算術照合](arithmetic_audit.json)を参照する。
照合は元アーカイブのラベルから別の数式実装で13候補・全9,216行を再計算した。
同じ assistant による別計算であり、独立した研究者のレビューや意味ラベルの
妥当性検証とは区別する。

## 再計算と次の境界

新しい出力先を指定すれば、モデルを使わずに開発計算を再実行できる。
出力先が既に存在すれば停止する。

```bash
python scripts/prepare_gate_study.py --config configs/experiment_10_mps.json --protocol research/experiment_10_mps_20261103/protocol.json --output runs/new-design
python scripts/audit_gate_fit.py --design-dir runs/new-design --output runs/new-design/arithmetic_audit.json
```

今回のローカル記録は `runs/exp10-mps-forecast-design-20261103-001`、準備済み run は
`runs/exp10-mps-gate-20261103-001`。これらのディレクトリを再利用・上書きしない。
[pre_generation_receipt.json](pre_generation_receipt.json)に同一性と未生成の状態を記録した。

次は盲検観察の担当方法を別記録で確定し、protocol・fit lock・run lock・観察担当の
ハッシュを study lock に結びつけてから生成する。設計・予測を読んだコンテキストを
そのまま盲検観察に使わない。観察者の種類と版、割当・文脈分離の方式を記録する。
新規エージェントの起動、意味ラベルの作成、評価出力の生成はまだ行っていない。

完了後は固定予測を `pobs score` で採点し、
`scripts/summarize_gate_study.py` で元ラベルからスコアを再計算して、主比較・全補助比較・
問題別の差・観察者不一致・事前指定した進行条件を報告する。
改善が小さい場合や条件を満たさない場合も、そのまま結果を残して終了する。
観察が欠ければ主判定は行わない。

## 実装検証との区別

[MLX/MPS の短い確認](../../validation/adapters_20261103/summary.json)は既知2題・2条件・
各1生成、上限16 tokenで、研究評価ではない。両 engine は同じ重み・prompt token IDs・
seed を使うが、sampling 実装と数値計算が異なる。出力一致や処理速度の一般的な比較を
主張しない。MPS 上の新規評価は、元の CPU 実験の完全な再現とも呼ばない。
