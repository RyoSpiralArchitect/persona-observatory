# 実験10/11 protocol（ドラフト）

**状態：ドラフト。未凍結。2026-11-06。**

凍結前の残り：新題48題の人による題材重複点検、Llama 開発出力の盲検 A/B
ラベル、両 run の forecast lock。この文書が機械可読 lock を兼ねない。

## 主張の範囲

問い：既知の指示条件を知ることが、証拠種類だけの場合より、新しい題材の
直接判断 D を予測する助けになるか。D は指定回答者による利用可能な判断の
有無。人格の持続、判断の正しさ、内部機構は測定対象にしない。

## 主比較と判定

- **主比較**：Joint − Evidence の D 二値 Brier 差（左−右、負が改善）
- **補助比較（事前指定）**：Setting − Global、Joint − Setting、Joint − Global、
  Evidence − Global。補助比較の成功は主比較の不成立を救済しない
- **記述的支持**：delta < −1e-12（有限自作問題群での符号判定。有意差ではない）
- **次段階への進行基準（提案値）**：delta ≤ −0.005、A/B 各枝で負、
  Joint−Global と Joint−Evidence NLL が悪化しない
- 欠けた質問・観察枝は分母から落とさず、完全性確認が通らなければ `INCOMPLETE`

## 予測器と選択規則（実装済み）

- Global / Evidence / Setting / Joint の4候補。判断内訳 h は全候補で共通の
  プール分布、D の確率のみグループ別縮約
- `g_D = (N_D + 0.5)/(N + 1)`、`h_k = (N_k + 0.5)/(N_D + 2)`、
  `r_c = (d_c + rho·n_c·g_D)/(n_c·(1 + rho))`
- rho 候補 {0.25, 1, 4, inf}。空セルと inf は `r = g_D`
- 質問ブロック 13-fold（feasibility 記録の fold 表と一致確認済み）で
  D Brier を最小化。1e-12 以内の同点は強い縮約（inf, 4, 1, 0.25）優先
- rho は予測器ごとに1つ、全セル・A/B 共通。A/B のカウントは別々
- 実装：`src/persona_observatory/forecasting.py`、選択実行
  `scripts/fit_forecasts.py`、抽出 `scripts/extract_dev_pool.py`

## データ役割

| モデル | 開発 | 評価 |
|---|---|---|
| SmolLM2-360M | v9 開発336 + 旧評価288（開発へ再割当、ラベル流用） | 新題48×6条件×2=576 |
| Llama-3.2-3B-Instruct | 同じ52題をこのモデルで再生成（624生成、開発専用） | 同一新題・同一样本IDの576 |

- 開発と評価で題材を共有しない。A/B は同一生成の二つの測定であり、
  独立した1,152観察ではない
- 評価 sample_id は両モデルで同一（experiment_id・seed_namespace・題・
  条件・反復を揃えた `design` 出力に由来）

## モデル同一性と環境

| run | モデル manifest | backend | dtype |
|---|---|---|---|
| E10 評価 | `manifests/smollm2-360m-instruct.json` | CPU | float32 |
| E10 開発 | 同上（アーカイブ由来、frozen runtime） | — | — |
| E11 評価 | `manifests/llama-3.2-3b-instruct.json` (rev 0cb88a4f) | CUDA | bfloat16 |
| E11 開発 | 同上 | CUDA | bfloat16 |

- 両評価 run の forecasts を同一凍結イベントとして扱い、**両方が揃うまで
  どちらの評価生成も開始しない**。一方の結果を見て他方を変更しない
- ソフトウェア：transformers==4.57.6（5.x では `prepare` が失敗することを
  smoke で確認済み）。backend 差とモデル差の分離は補助 run の記述報告のみ

## 観察

- v9 同様、同じ基盤モデルの別コンテキスト A/B。独立人間評者ではない留保継続
- 条件名・seed・証拠層・予測・過去順位・モデル identity は渡さない。
  文体からの推測可能性は盲検範囲の限界として明記
- Llama 開発624生成にも盲検 A/B ラベルが必要（fit 前の未完了作業）

## 完了済みの記録（2026-11-06 時点）

- Llama-3.2-3B-Instruct manifest 固定・CUDA smoke（runs/llama32-smoke-001）
- 開発池抽出：`research/experiment_10/development/`（1,248行・52題・fold 表は
  feasibility 検証値と一致）
- 新題ドラフト：`question_bank_draft.json`（48題・4層×12・過去60題と
  トピック非重複を機械確認。意味的重複の人による点検は未実施）
- design 出力 `design_eval.jsonl`（576 samples）
- SmolLM2 側 fit・rho 選択・予測 `predictions_smollm2.jsonl`（4,608行）：
  global=inf, evidence=4, setting=1, joint=1。CV の最良値は選択用であり
  独立した成功証拠ではない
- Llama 開発生成 run：furnace `runs/llama32-devpool-001`（実行中/完了時に
  verify-run で確認）

## 未完了

- 新題の人による題材重複点検と question_bank 確定（確定時に draft を改名し
  ハッシュを固定）
- Llama 開発出力の盲検ラベル → Llama 版 fit・予測
- 両 predictions の forecast lock → prepare(prospective) → 各1回の生成
- 盲検観察 → 採点 → 報告
