# Mac での追加実験

Apple Silicon の Mac で、まず短い生成を確認し、その後に別の run として
追加実験を行う手順。リポジトリ直下でコマンドを実行する。

MPS は PyTorch が提供する Metal 向けバックエンドである。
この移植では MPS の実機実行はまだ確認していない。CPU での確認と
Mac での実行結果は、別々の環境記録として扱う。
参照：[PyTorch MPS documentation](https://docs.pytorch.org/docs/2.15/notes/mps.html)。

## 1. Python 環境を作る

Python 3.12、Apple Silicon 向けのネイティブ Python、macOS 15 以降を
この手順の想定とする。OS の最低要件は選んだ PyTorch リリースにも依存する。
`python3.12` がない場合は、先に Python を用意する。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1
python -m pip install -e '.[inference,dev]'
```

`2.14.1` は公式配布ページに記載された macOS 用の固定例であり、
最新リリースという意味ではない。この例の Mac 実機互換性は、以下の
`doctor` と短い生成で確認する。
[公式のバージョン別インストール手順](https://pytorch.org/get-started/previous-versions/)
で別の版を選ぶ場合は、その版を環境記録に残す。

実験9の `torch==2.15.0+cpu` は、当時の Linux CPU 実行を記録した値。
Mac 環境へその requirements をそのまま適用せず、torch をプラットフォームに
合わせて先に入れる。`torchvision` と `torchaudio` はこのテキスト生成器には不要。

```bash
pobs doctor --device mps --dtype float32
```

MPS が使えないときは、Python のアーキテクチャ、macOS、PyTorch のビルドを
確認する。CPU で続ける場合は、`smoke_cpu.json` を使う別の run を準備する。
要求した MPS の run を、黙って CPU の結果に置き換えない。

## 2. 固定したモデルを取得する

```bash
pobs fetch-model --config configs/smoke_mps.json --destination models/smollm2-360m-instruct
```

モデルは `HuggingFaceTB/SmolLM2-360M-Instruct`、リビジョンは
`a10cc1512eabd3dde888204e902eca88bddb4951`。
取得した重みは git 管理の対象に含めない。モデル取得にはネット接続が必要で、
生成時はローカルのモデルファイルを使う。

## 3. 短い動作確認を行う

```bash
pobs prepare --config configs/smoke_mps.json --run-dir runs/mac-smoke-001 --model-path models/smollm2-360m-instruct
pobs generate --run-dir runs/mac-smoke-001
pobs verify-run --run-dir runs/mac-smoke-001
pobs export-observer --run-dir runs/mac-smoke-001 --observer observer_a
```

同じ run 名を再利用しない。エラーになった試行もそのまま保存し、修正後は
新しい run 名で準備する。生成の停止理由と、長さ上限に達した出力も確認する。
この smoke は操作確認用であり、新しい研究上の評価データではない。

## 4. 実験用の設定を作る

```bash
cp configs/smoke_mps.json configs/my_mac_experiment.json
```

コピー側の設定と問題群を編集し、目的・指示・質問・反復数・乱数系列・
生成上限を決めてから、新しい run を `prepare` する。
同じ設定を CPU と MPS で比較する場合も、出力先は別にする。

最初は `float32` で動作を確認する。精度形式や生成設定を変更した実行は、
新しい条件として記録する。Apple MPS では `float64` は使えない。
同じ seed でも CPU、MPS、CUDA 間の生成一致は保証されない。
参照：[MPS の型制約](https://docs.pytorch.org/docs/2.15/notes/mps.html)、
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)。

追加の予測試験を始めるときは、[次の実験案](next_experiment.md)と
[観察・測定の手順](measurement.md)を先に読む。
予測器の比較には、生成計画の固定に加えて、評価出力を見る前の予測固定が必要。

## 確認結果を残す

```bash
python -m pip freeze > runs/mac-smoke-001/packages.txt
sw_vers > runs/mac-smoke-001/macos.txt
```

自動保存される backend・dtype・生成記録と合わせて、再実行時の条件を追える
ようにする。モデル重み、仮想環境、run のローカルパスは Mac 内で管理し、
研究に採用する出力・設計・観察コードは別途選んで記録する。
