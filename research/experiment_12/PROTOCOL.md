# 実験12 protocol(ドラフト)

**状態:ドラフト。未凍結。2026-11-06。**

凍結前の残り:両モデルの開発生成・盲検 A/B ラベル・fit・両 run の
forecast lock。この文書が機械可読 lock を兼ねない。

E10/E11 の凍結成果物・判定は変更しない。E12 は別の主仮説を持つ新実験。

## 主張の範囲

問い:既知の指示条件を知ることが、証拠種類だけの場合より、新しい題材の
**5カテゴリ応答分布**(support/reject/withhold/mixed/no_direct)を予測
する助けになるか。E10/E11 で測った「直接判断の有無 D」は no_direct
カテゴリとして分布の一部に含まれる。人格の持続、判断の正しさ、内部機構
は測定対象にしない。

## 主比較と判定

- **主比較**:Joint − Evidence の 5カテゴリ Brier paired delta(左−右、
  負が改善)、質問加重・枝別平均
- **判定(閾値合流型)**:delta ≤ −0.005 で SUPPORT、それ以外は
  NOT_SUPPORTED。E10 の「符号 SUPPORT + 別立て進行判定」方式では微小差
  でも SUPPORT と出る問題があったため、閾値を主判定に組み込む
- **補助比較(事前指定)**:Joint − Setting(証拠層情報が設定情報を超えて
  改善するかの識別対比)、Setting − Global、Joint − Global、
  Evidence − Global。補助の成功は主比較の不成立を救済しない
- **E10 比較用導出指標**:各予測の p(no_direct) から導出する gate Brier
  の同対比群を補助として報告する
- 欠けた質問・観察枝は分母から落とさず、完全性確認が通らなければ
  `INCOMPLETE`

## 予測器と選択規則

- Global / Evidence / Setting / Joint の4候補。予測対象は5カテゴリ分布
  そのもの。`h_base[k] = (N_k + 0.5)/(N + 2.5)`
- セル分布 `h_c[k] = (count_c[k] + rho·h_base[k])/(n_c + rho)`。
  空セルと inf は `h_c = h_base`。Global は常に h_base
- rho 候補 {0.25, 1, 4, inf}。**category Brier で質問ブロック CV 選択**
  (E10 は gate Brier で選択していた — 構成チャネルを選択が見ない問題の
  修正)。1e-12 以内の同点は強い縮約優先
- rho は予測器ごとに1つ、全セル・A/B 共通。A/B のカウントは別々
- 実装:`src/persona_observatory/forecasting.py`(mode="category")、
  `scripts/fit_forecasts.py --mode category`

## 介入(設定)

`research/experiment_12/settings.json`:E10 の6設定(neutral/cautious/
confident × paraphrase 0/1)に、non-answer を正当化する2 family を追加:

- deferral × 2: 証拠不十分なら明示的辞退を認める指示
- gatekeeper × 2: 証拠が十分な場合のみ答える役割

計10設定 × 2 replicate。requested_answerer は Morgan で統一。

## データ役割

| モデル | 開発 | 評価 |
|---|---|---|
| SmolLM2-360M-Instruct | 52題×10設定×2 = 1,040生成,cpu/float32 | 48題×10設定×2 = 960生成,cpu/float32 |
| Llama-3.2-3B-Instruct | 同 1,040,cuda/bf16 | 同 960,cuda/bf16 |

質問は両モデル共通(開発=E10 の52題、評価=E10 確定48題)。sample_id は
両モデルで一致する設計。

## 観察

- 盲検エクスポートを48件バッチで独立コンテキストに分割して意味ラベル。
  A/B は別測定(同基盤モデル別コンテキスト)。
- ラベルは rubric の8フィールド。コードは表明判断に従い、データの誤読は
  note に記録する。
- `validate-observer --lock` で固定してから fit・採点に進む。

## 時系列

1. 設定・予測器・主比較・rubric・題の確定(本書)+ 機械 freeze
2. 両モデル開発生成 → 盲検 A/B ラベル → ロック
3. dev pool 構築(入力ハッシュ束縛)→ category モード fit → 両モデル
   forecasts を1つの freeze manifest に束縛
4. `pobs prepare --forecasts` → 各1回生成 → 盲検観察 → `pobs score`
5. `e12_report` で判定
