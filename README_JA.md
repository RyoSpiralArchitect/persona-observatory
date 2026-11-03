# Persona Observatory

**Output-based persona research, with preserved evidence and local experiments on CPU, Apple MPS, and NVIDIA CUDA.**

LLM の内部状態を予測特徴に使わず、条件のもとで生成される出力を観察して、
その傾向を次の出力の予測に使える表現へまとめる研究用リポジトリ。
出力の意味を読む観察者と、確率を推定・評価するコードを分ける。

現時点の材料は、完了した実験9、その出自を含む実験8の記録、そして追加実験用の
小さな Python CLI。過去の結果は固定した記録として残し、新しい run は別に作る。
この段階で内部の人格や推論軌道を復元した、という主張はしていない。

**License: [Apache-2.0](LICENSE)** · [Mac](docs/mac.md) · [furnace / CUDA](docs/furnace.md) ·
[観察と採点](docs/measurement.md) · [設定](docs/configuration.md) · [次の実験案](docs/next_experiment.md)

## 研究の現在地

実験9では、過去336出力を開発用に使い、新しい24題・288出力について、
生成前に固定した予測を評価した。直接判断 D の有無と、D がある場合の判断内訳を
分けて、条件ごとの差をどれだけ全体分布へ寄せるかを比較した。

| 固定候補 | 五分類 Brier | 五分類 NLL |
|---|---:|---:|
| G: 全体分布 | 0.658847 | 1.194051 |
| F: 共通の縮約 | 0.655841 | 1.208360 |
| H: D と内訳で別々の縮約 | 0.657402 | 1.213900 |
| H_gate: D だけ条件別、内訳は全体へ | **0.651359** | **1.181487** |

同じ生成に対する観察 A/B の平均で、いずれも低い方がよい。
**H が G と F の両方を上回るという主基準は未達。** H_gate は固定した7候補中で
最良だったが、この順位を見て独立した主比較の成功へ読み替えない。
A/B は同じ基盤モデルの別コンテキストによる観察で、独立した生成が二倍になった
わけではない。

- [実験9の全文](research/experiment_09/REPORT_JA.md)
- [方法と解釈範囲](research/experiment_09/METHODS_JA.md)
- [既存研究記録の一覧](docs/research_history.md)
- [元のアーカイブと検証方法](archives/README.md)

根拠の再計算監査は元アーカイブ内にある。新しい archive checker はファイルの
バイト一致を確認するもので、数値再計算監査と役割が異なる。

## 最初の動作確認

Python 3.12 を推奨。クローンまたは展開したリポジトリの直下で実行する。
以下は **Linux CPU** の例。Mac の MPS と furnace の CUDA は上の専用手順で torch を
選んでから、同じ CLI を使う。モデル重みや仮想環境は git に入れない。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[inference,dev]'

pobs doctor --device cpu --dtype float32
pobs fetch-model --config configs/smoke_cpu.json --destination models/smollm2-360m-instruct
pobs prepare --config configs/smoke_cpu.json --run-dir runs/cpu-smoke-001 --model-path models/smollm2-360m-instruct
pobs generate --run-dir runs/cpu-smoke-001
pobs verify-run --run-dir runs/cpu-smoke-001
pobs export-observer --run-dir runs/cpu-smoke-001 --observer observer_a
```

`torch==2.14.1` は公式配布にあるセットアップ例。ここで実行確認した CPU は
`torch==2.15.0+cpu` / `transformers==4.57.6` である。異なる版・backend の結果は
別の環境として記録する。[PyTorch の配布手順](https://pytorch.org/get-started/previous-versions/)
も参照。`inference` extra は torch を固定せず、先に選んだ CPU/MPS/CUDA 用の
インストールを尊重する。

smoke は **既に観察した2題 × 2条件 × 1生成、上限16 token**。
`configs/replay_v9_{cpu,mps,cuda}.json` は既知の24題を使う288出力の replay 用。
どちらも新しい科学的な holdout 評価ではなく、元の実験9を上書きしない。
CPU/MPS/CUDA 間の同じ seed による出力一致も保証しない。

## CLI が保存するもの

| 操作 | 内容 |
|---|---|
| `doctor` | 実際の device / dtype、torch・Transformers、利用可能性。重みはロードしない |
| `fetch-model` | 完全な revision と各ファイルの SHA-256 を固定した公開モデル取得 |
| `design` | 推論前にサンプル ID、質問、条件、乱数系列を出力。外部で予測を作る際に使う |
| `prepare` | 入力、chat template、prompt token IDs、seed、コードのコピー、任意の外部予測を固定 |
| `generate` | その run に対する一回の生成試行。全文・token IDs・終了理由・実行環境を保存 |
| `verify-run` | 入出力の整合、終了状態、固定時点と生成時点の順序を確認 |
| `export-observer` | 条件名や seed を見せない、意味的コーディング用の作業場所を作成 |
| `validate-observer` | 全コードの形式・話者整合・重複一致を検証。`--lock` で確定 |
| `score` | 生成前に固定済みの外部予測だけを、確定済み観察コードで採点 |

CPU の RNG を各サンプルに一つずつ割り当て、同じ質問・反復の系列を条件間で
対応させる。forward は指定した backend、nucleus sampling は CPU float32。
推論はローカルの Transformers causal LM を使い、`trust_remote_code=False`。
別の [MLX adapter](docs/adapters.md)も選べる。同じ固定版の重みを読み込み、
NumPy の独立した乱数系列で sampling する。量子化、複数 GPU 分割、学習、内部状態 hook は含めていない。

`prepare` 後にコードや設定、モデル、指定 backend が変わった場合は新しい run を
作る。試行開始後の自動再試行・黙った resume・既存結果の上書きは行わない。
失敗記録と途中までの出力を残す。

`purpose=prospective` には `prepare --forecasts predictions.jsonl` が必須。
予測器の学習・選択・開発データの出自は、研究計画側で別途固定する。
この CLI 自体は新しい予測器を fit しない。

## コードと記録の配置

| 場所 | 用途 |
|---|---|
| `src/persona_observatory/` | 新しい portable runner、観察検証、採点 |
| `configs/`, `examples/`, `manifests/` | 短い確認・既知問題 replay の設定、入力、モデル識別 |
| `archives/` | バイト固定の実験9 ZIP。中に元の実験8記録も含む |
| `research/experiment_09/` | 元アーカイブと照合できる、読みやすい報告・図・集計のコピー |
| `docs/` | Mac / furnace、設定、観察、次回の設計案 |
| `validation/` | この実装の検証記録。研究上の新しい効果の証拠とは区別 |
| `tests/` | 乱数、データ境界、観察・採点、アーカイブ整合の検査 |
| `runs/`, `models/` | 各マシンのローカル実行・モデル。git 対象外 |

## 軽量な検証

Torch やモデルを取得せずに、core と保存済みアーカイブを確認できる。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_archive.py
pobs --help
```

Torch を使う backend tests は、Torch がない環境では skip される。
CI は Ubuntu/macOS 上のこの core 検証を対象とする。CI の macOS job が成功しても
MPS 推論を検証したことにはならない。CPU の実モデル確認は `validation/` に記録。
Mac MPS と MLX Metal は同じモデルで各4出力の短い動作確認を完了した。
[検証記録](validation/adapters_20261103/summary.json)には失敗した試行も残している。
furnace CUDA の実機生成は未確認。

## 次の研究

[次回の設計案](docs/next_experiment.md)では、判断内訳を全体分布に固定し、
D の条件づけを全体・証拠種類・指示条件・両者の組合せで比較する。
初期案は未実行の設計メモとして保持する。
[MPS での具体化](research/experiment_10_mps_20261103/README.md)では、
新規96題・1,152生成への予測を旧データだけから固定した。盲検観察担当の確定を待ち、
新規評価生成と採点はまだ行っていない。

追加実験では、新しい問題群とデータの役割を決め、予測と採点基準を生成前に固定し、
観察者へ条件情報を渡さず、元の結果と新しい結果を区別して保存する。
研究上の見通しは [CONTRIBUTING.md](CONTRIBUTING.md) にまとめた。

## ライセンスと外部資源

このプロジェクトの原著コード・作成した評価材料・研究記録は Apache License 2.0。
外部のモデル・依存ライブラリ・引用資料には、それぞれの条件が適用される。
モデル重みとライブラリ本体はリポジトリに同梱していない。
[NOTICE](NOTICE) と [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照。
