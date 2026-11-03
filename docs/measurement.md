# 生成・観察・予測評価をつなぐ

このプロジェクトの出発点は、内部状態を予測の特徴として使わず、生成群から
条件ごとの行動プロファイルを作り、まだ見ていない生成に対して評価すること。
「人格」という名称の妥当性を前提にせず、観察できる区別とその予測価値を記録する。

## 完了した実験と新しい run

実験9の結果は [report](../research/experiment_09/REPORT_JA.md)、設計は
[protocol](../research/experiment_09/PROTOCOL.md)に収録している。
元のコード、ラベル、予測、ロックは `archives/` の完全なアーカイブに含まれる。
新しい portable runner の出力は別の run に保存する。
アーカイブの展開と元の完全監査の手順は
[archives/README.md](../archives/README.md)を参照する。
`python scripts/verify_archive.py` はモデルを必要としないバイト単位の整合性確認で、
元の実験の数値を独立再計算する監査とは範囲が異なる。

`prepare` によるプロンプト・seed・設定の固定と、予測器の確率を新しい評価出力より
先に固定することは、別の段階である。生成の計画を固定しただけで、予測性能の
前向き検証を行ったことにはならない。追加試験では予測器の学習・選択・予測固定の
段階を設計に含める。

## 観察者へ渡す情報

基本単位は、質問に対して生成された全文と、その質問で指定された回答者。
観察者には次の情報が必要である。

- 識別用 `sample_id`
- 回答者を指定する `requested_answerer`
- 質問 `question`
- 全生成文 `completion`
- 終了理由 `stop_reason`

指示条件、証拠の層、seed、事前予測、他の観察者のコードを読む前に、
各観察者が割り当てられた export と共通のルーブリックを読む。
意味判断を行う人・モデルの種類、モデル版、コンテキストの分離方法、
コード付けに使った指示も記録する。

```bash
pobs export-observer --run-dir runs/my-experiment-001 --observer observer_a
pobs export-observer --run-dir runs/my-experiment-001 --observer observer_b
```

export を作る操作自体は、コード付けや人間による評価の実施を意味しない。
別々の観察を行い、それぞれの結果を保存する。

各 export は `runs/<run>/observers/<observer>/` に置かれる。
観察者は `rubric.md`、`discovery_blind.jsonl`、`validation_blind.jsonl` を読み、
`labels.template.jsonl` を基に、全項目を埋めた `labels.jsonl` を作る。
`observations.md` に入っている案内文は、自分の実際の観察メモへ置き換える。

```bash
pobs validate-observer --run-dir runs/my-experiment-001 --observer observer_a
pobs validate-observer --run-dir runs/my-experiment-001 --observer observer_a --lock
```

各観察者の全レコードが揃ってからロックする。A/B の両枝を予測で宣言した場合は、
両方を別々に観察・検証・固定する。ロック後にラベルを直して採点結果を更新しない。

## 実験9で固定したコード

意味判断の正本は [rubric.md](../research/experiment_09/rubric.md)。
次の表は項目の案内であり、境界例や整合性条件は正本を参照する。

| コード | 観察するもの |
|---|---|
| `answerer_answer_mode` | 回答者本人の判断／その人の判断の伝聞／両方／どちらもなし |
| `answerer_judgment` | 質問対象への支持／否定／保留／未解決の混在／判断なし |
| `answerer_stance` | 明確に決める／実質的に留保する／混在／判断なし |
| `limit_owners` | 明示された証拠・知識の限界を、誰が述べたか |
| `other_answer` | 回答者以外の人物の、質問対象についての判断があるか |
| `character_description` | 性格・役割・一般的な傾向の説明があるか |

`sample_id` と短い説明 `note` を加えた八つのフィールドを保存する。
文章の正しさではなく、文章中に表明された判断をコードする。
質問文に書いてある証拠の弱さを、生成文が限界を述べたこととして加算しない。

直接判断 D は、mode が `direct` / `both` で、judgment が `none` 以外のとき1。
このとき最終5分類はその judgment。その他は `no_direct` とする。
明示的な限界 E は、D=1 かつ `limit_owners` に `answerer_direct` がある場合だけ1。
したがって E は D を含意する。

生成が長さ上限に達したときも、見えている部分だけをコードし、続きを補わない。
同一の `(requested_answerer, question, completion)` には、同じ六コードを付ける。
説明文の違いは記録してよい。重複した文章も元の生成記録から削除せず、
生成件数とユニークな文章数を区別して報告する。

## 予測を採点するとき

実験9の5分類順は `support, reject, withhold, mixed, no_direct`。
確率はこの順と正規化を明記する。主指標・補助指標・比較・集計単位を評価出力の
生成前に決める。

portable CLI は、外部で作った予測を生成前に固定して採点する機能を持つ。
予測器の fitting や交差検証は自動で行わない。
まず使用する config を作り、評価点の ID を取得する。

```bash
pobs design --config configs/my_experiment.json --output targets.jsonl
```

`targets.jsonl` の sample ID を用いて、別途開発データから推定した確率を
`predictions.jsonl` に保存する。各行は次の四フィールドだけを持つ。

```json
{"sample_id":"ID_FROM_TARGETS","observer":"observer_a","predictor":"Global","category_probabilities":[0.4,0.2,0.1,0.1,0.2]}
```

上の行は形式の説明用で、実験の予測ではない。
実ファイルでは `ID_FROM_TARGETS` を実際の ID に置き換え、すべての
sample × observer × predictor の組を一度ずつ含める。
5確率は有限かつ0より大きく1より小さく、合計1（許容誤差 `1e-9`）とする。
ゼロを後から補正したり、欠けた行を採点時に埋めたりしない。

```bash
pobs prepare --config configs/my_experiment.json --run-dir runs/my-experiment-001 --model-path models/smollm2-360m-instruct --forecasts predictions.jsonl
pobs generate --run-dir runs/my-experiment-001
pobs verify-run --run-dir runs/my-experiment-001
```

`purpose=prospective` では生成前の外部予測が必須。
`smoke_replay` / `exploratory` では予測を省略して生成できるが、その場合は
生成後に新たな予測を追加して前向き予測として採点することはできない。
新しい評価出力が必要な場合は、予測を固定した別の run を作る。

予測で宣言した全観察者のラベルを固定した後に採点する。

```bash
pobs score --run-dir runs/my-experiment-001
```

CLI は run・予測・観察のロックを確認してから `scores.json` と
`score_lock.json` を新規に書く。結果は各出力の損失と、観察者・予測器ごとの平均。
この関数は候補選択、モデル fitting、有意差検定、母集団の信頼区間を追加しない。

- 5分類 Brier：5つの二乗誤差の和。範囲は0〜2。
- D の Brier：二値の二乗誤差。範囲は0〜1。
- NLL：実測コードに割り当てた確率の負の自然対数。
- NLL の分解：D の NLL と、D=1 の場合の判断内訳 NLL の寄与を足す。

portable scorer の D 確率は `1 - p_no_direct` で定義する。
二値 NLL は `p_no_direct` に対する `log` / `log1p` から計算し、補数が浮動小数点で
1に丸められるほど小さい正確率でも、確率の clipping を使わず扱う。
5分類 NLL の分解は上の数学的関係であり、現在の portable scorer が書く主な
損失列は `category_brier`、`category_nll`、`gate_brier`、`gate_nll` の四つ。

ラベルを予測に合わせて修正しない。完成した観察と説明をハッシュ付きで固定し、
不一致があるときは、まず各枝での結果と不一致例を併記する。
観察者の一致率と、予測器の性能は別の測定である。

実験9の A/B は同一基盤モデルの別コンテキストで、独立した二種類のモデルや
人間評価ではない。同じ288生成を二通りに測っており、生成数を576と数えない。
新しい試験でも、モデル出力数、質問数、条件数、観察者数を分けて書く。

## プロファイルの解釈

プロファイルは、問題群・指示・モデル版・生成条件・観察基準に依存する。
良い予測は、内部機構や持続的な人格の一意な同定を意味しない。
また、この runner は隠れ状態や内部推論軌道を観測する装置ではない。
それらと比較する追加研究では、介入・readout・対応づけを別途設計する。
