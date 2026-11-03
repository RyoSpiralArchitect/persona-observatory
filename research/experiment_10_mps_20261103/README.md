# 実験10 MPS：採点完了・進行条件未達

**新規評価1,152生成とA/B観察・採点を完了。主比較は改善せず、進行条件は4項目とも未達。**
Joint−EvidenceのD Brier差はA/B平均 **+0.000749**。Settingが最良だったが補助所見とする。
[結果報告](REPORT_JA.md)と[機械可読結果](results_20261103.json)に全比較を残した。
予測器・問題・ルーブリック・比較と進行基準・担当方法を生成前に固定した。
生成は472.286秒で完了し、1,149件が終了トークン、3件が96 token上限で停止した。
実際の生成は計39,627 tokenだった。[生成記録](generation_receipt_20261103.json)を参照。

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
| 観察 | 別文脈の検査agent A/B。今回の条件・予測を非開示にした全文読解。初期環境情報への曝露あり |

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

[観察担当](observer_assignment_20261103.json)を新規記録で確定し、protocol・fit lock・
run lock・観察担当と環境・集計スクリプトを[study lock](study_lock_20261103.json)に
結びつけてから、一度だけ生成した。元の protocol の担当待ちという記述は履歴として
変更していない。後続記録は担当決定だけを補い、予測や判定基準を変更していない。

各agentには同じ1,152件を別々の順序で渡した。完全に同一の話者・質問・回答の組だけを
まとめると1,076件であり、それぞれを全文読んで判定した。
[資料のハッシュ](observer_packets_20261103.json)と、手入力した各54バッチを保存した。
二者は同じ実行モデルの別コンテキストで、人間の観察者や異なる基盤モデルではない。
正確な提供モデルのビルド識別子はツールから得られず、固定できたとは主張しない。
資料へのアクセス制限は指示によるものでOS隔離ではなく、質問文から証拠構造が見える
という限界もある。A/Bは同じ出力の別測定であり、生成数を倍に数えない。

`fork_turns=none`でも、環境指示・記憶要約・旧実験の判定を含むプロジェクト境界は
開始時に与えられていた。両者がこの曝露を申告し、[別記録](observer_exposure_notice_20261103.json)
と観察メモに保存した。完全に研究履歴を知らない観察とは呼ばない。

[両観察を固定](observation_receipt_20261103.json)した後、固定予測を `pobs score` で採点し、
`scripts/summarize_gate_study.py` で元ラベルからスコアを再計算して、主比較・全補助比較・
問題別の差・観察者不一致・事前指定した進行条件を報告した。
このrunは進行条件未達として終了し、追加生成や結果に合わせた再ラベルは行わない。
[公開証拠](evidence_20261103.zip)はモデルなしで再検証できる。コマンドと公開用コピーの
限定的な伏字については[報告書](REPORT_JA.md#証拠と再計算)を参照。

## 実装検証との区別

[MLX/MPS の短い確認](../../validation/adapters_20261103/summary.json)は既知2題・2条件・
各1生成、上限16 tokenで、研究評価ではない。両 engine は同じ重み・prompt token IDs・
seed を使うが、sampling 実装と数値計算が異なる。出力一致や処理速度の一般的な比較を
主張しない。MPS 上の新規評価は、元の CPU 実験の完全な再現とも呼ばない。
