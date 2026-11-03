# furnace での追加実験

このページは NVIDIA GPU を持つ Linux / WSL2 マシンでの実行手順。
furnace の現在の OS、GPU、ドライバーは、このリポジトリ作成時には
直接確認していない。CUDA 実機での動作結果も未収集なので、最初に環境を
確認し、短い smoke を通してから実験用の run を作る。

## 1. GPU とドライバーを確認する

```bash
nvidia-smi
```

GPU 名とドライバーを記録する。RTX 5090 などの Blackwell を使う場合は、
そのアーキテクチャを含む PyTorch バイナリを選ぶ。
下の固定例は PyTorch 2.14.1 / CUDA 13.0 用で、公式配布ページに記載されている。
PyTorch の案内では CUDA 13.0 の利用に Linux 580.65.06 以降、
Windows 580.88 以降のドライバーを挙げている。
参照：[公式の配布コマンド](https://pytorch.org/get-started/previous-versions/)、
[CUDA 13.0 と Blackwell の案内](https://pytorch.org/blog/pytorch-2-12-release-blog/)。

別の wheel を選ぶ場合は、その CUDA 版の要件を確認する。
必要なドライバー変更は OS / NVIDIA の正式な手順で行う。
このプロジェクトのインストール処理はドライバーを変更しない。

## 2. Python と CUDA 用 torch を入れる

リポジトリ直下で実行する。Python 3.12 を想定する。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[inference,dev]'
pobs doctor --device cuda --dtype float32
```

この CUDA 手順は公式配布に基づくセットアップ例であり、本リポジトリで実機検証
済みの環境という意味ではない。使用した版と backend は各 run に記録する。
`doctor` で CUDA が利用可能なことを確認し、次の短い生成でモデルの実行を確かめる。
アーキテクチャ不一致や driver エラーがある状態で本試行を始めない。

`archives/` の実験9は Linux CPU の過去記録である。
その `torch==2.15.0+cpu` を CUDA 用環境に入れると GPU 用設定にはならないため、
上のように torch を別に選んでからプロジェクトを入れる。

## 3. モデル取得と短い生成

```bash
pobs fetch-model --config configs/smoke_cuda.json --destination models/smollm2-360m-instruct
pobs prepare --config configs/smoke_cuda.json --run-dir runs/furnace-smoke-001 --model-path models/smollm2-360m-instruct
pobs generate --run-dir runs/furnace-smoke-001
pobs verify-run --run-dir runs/furnace-smoke-001
pobs export-observer --run-dir runs/furnace-smoke-001 --observer observer_a
```

モデルの識別子とコミットは config で固定される。
この短い run は生成・保存・観察用 export の確認用であり、実験10の評価ではない。
実験用の問題群、予測、評価基準は別に固定する。

```bash
python -m pip freeze > runs/furnace-smoke-001/packages.txt
nvidia-smi > runs/furnace-smoke-001/nvidia-smi.txt
```

生成の実測時間、終了理由、出力文字列、実際の device / dtype を確認する。
GPU の使用率や温度を見たい場合は、別ターミナルで `nvidia-smi` を使う。

## 4. 大きな実験へ進めるとき

smoke の config を別名でコピーし、質問・指示・反復数・生成上限を決める。
モデルや dtype を変えた試行は、別の run として準備する。
ハードウェア差と persona 条件差を混同しないよう、比較する条件を一つの
backend 内で揃えた後、別 backend の結果を並べる。

同じ seed を割り当てても、CPU と CUDA の演算結果や生成列は一致するとは
限らない。元の実験9の数値と新しい GPU 実行の数値を、同じ実行の再現値として
置き換えない。
参照：[PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)。

別モデルを使うときは、完全なリビジョン、モデルファイル、chat template、
コンテキスト長と生成上限を確認する。この runner はローカルの Transformers
causal LM を対象とし、量子化、複数 GPU 分割、MLX、学習処理を自動では追加しない。
モデルを変更したという記録も、追加実験の設計の一部にする。

## Windows ネイティブを使う場合

CUDA を利用できる Windows Python を選び、環境作成・有効化だけを次のように
置き換える。以降の `python -m pip` と `pobs` のコマンドは同じ形式で使う。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[inference,dev]"
pobs doctor --device cuda --dtype float32
```

Windows ネイティブでの一連の実行も未検証。パスやファイルロックの問題が出た
場合は、出力を保存したうえで別 run を準備する。
既存の失敗記録や完了した run を消して再実行しない。
