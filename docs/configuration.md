# 設定と新しい run

`configs/smoke_cpu.json` を例に、config が指す質問群・指示群・モデル manifest を
JSON で指定する。相対パスは **config ファイルの場所**を基準に解決する。
入力は `prepare` で run 内へコピーされ、その後の元ファイル変更から分離される。

## Config

| キー | 内容 |
|---|---|
| `schema_version` | `persona-observatory-study-v1` |
| `experiment_id` | 実験名。英数字、`_`、`-`、`.` の識別子 |
| `purpose` | `smoke_replay`、`exploratory`、`prospective` |
| `seed_namespace` | 質問と反復から seed を作る名前空間 |
| `replicates` | 質問・指示の組合せごとの正の生成数 |
| `question_bank` | 質問リスト JSON のパス |
| `settings` | 指示リスト JSON のパス |
| `model_manifest` | モデル ID、完全な commit SHA、ファイルごとの size/SHA-256 |
| `runtime` | 下記の実行設定 |
| `system_template` | 任意。`{answerer}` を使える system 文面 |
| `wrapper_template` | 任意。`{answerer}` を使える指示の見出し |
| `notes` | 任意の設計メモ |

`purpose=prospective` は生成前の外部確率予測を要求する。
このラベルだけで、問題の新規性や研究計画の十分性が自動認定されることはない。
開発・評価の役割、既知問題との重複、fitting の出自は研究者が別途記録する。

質問の必須フィールドは `question_id`, `question`, `topic`, `evidence_stratum`。
指示の必須フィールドは `setting_id`, `requested_answerer`, `policy_text`。
ほかのメタデータは保存できるが、prompt に使うのは明示した質問・回答者・指示だけ。

## Runtime

| キー | 値・初期値 |
|---|---|
| `engine` | `transformers`（初期値）、`mlx` |
| `device` | `cpu`, `mps`, `cuda`, `cuda:N`, `auto`。初期値 `auto` |
| `dtype` | `float32`, `float16`, `bfloat16`, `auto`。`auto` は float32 |
| `max_new_tokens` | 正整数、初期値96 |
| `temperature` | 非負。0はgreedy、初期値0.8 |
| `top_p` | 0より大きく1以下、初期値0.95 |
| `cpu_threads` | 正整数、初期値4 |
| `batch_size` | 正整数、初期値1 |

Transformers の `auto` は準備時に CUDA→MPS→CPU の順で利用可能なものを選び、その選択を固定する。
明示した device が利用できない場合は停止する。dtype が実機で使えるかは
`doctor` と smoke で確認する。

MLX の device は `metal`, `cpu`, `auto` で、`auto` は Metal→CPU の順。
現行 adapter は `batch_size=1` のみ対応し、`cpu_threads` は未使用の初期値4に限定する。
MLX は元の非量子化 Llama 系重みを読み込む。
[engine ごとの環境と sampling の違い](adapters.md)を参照する。

`seed_namespace + question_id + replicate_id` から SHA-256 で63-bit seed を作る。
同じ質問・反復には、異なる指示条件でも同じ stream を割り当てる。
`experiment_id` は sample ID にも入る。
異なる backend を対応づける比較では、実験名・名前空間・質問・反復を揃え、
run のディレクトリ名を変える。同じ seed は同じ出力を保証しない。

## モデル

同梱 manifest は SmolLM2-360M-Instruct の既知の六ファイルを固定する。
`fetch-model` は公開 HTTPS の固定 revision からそれらだけを取得し、size と
SHA-256 を検証する。認証トークンを暗黙には使用しない。

別のモデルでは、利用可能な Transformers causal LM を専用ディレクトリへ用意し、
モデル ID・完全な revision・全ファイルの size/SHA-256 を持つ manifest を新規作成する。
重み、tokenizer、config、必要なテンプレートを含む実際のファイル構成を固定する。
gated model の利用手続きは別途行う。remote code の実行や量子化の自動導入はしない。

ディレクトリは manifest のファイル一覧と一致させる。Hugging Face の `.cache`
メタデータは対象外だが、重みや tokenizer の未記録ファイルは許可しない。
元と異なるファイルや失敗した download がある場合は、それを調べて保存してから
別ディレクトリを用意する。

## Run の境界

`prepare` は新規ディレクトリにだけ書く。入力のコピー、tokenized plan、
ソースのコピー、環境 preflight を `run_lock.json` にまとめる。
`generate` は `attempt.json` を作ってから生成し、各出力を逐次保存する。
完了時に `completion.json`、例外時に `failure.json` を残す。

一度始めた run を再開・上書きしない。設定・ソース・環境を変えたら新しい run を
準備する。生成中に停止した出力は採点に使えず、完全な出力を持つ run だけを
`export-observer` と `score` に渡せる。

生成前に固定した外部予測、観察、採点の具体的な形式は
[measurement.md](measurement.md) に記載している。
