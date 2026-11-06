# 実験12 設計提案:構成チャネル予測器 × ゲート解放介入

**状態**: 提案(2026-11-06)。凍結・生成は未実施。E10/E11 の結果を受けた
次段設計であり、E10/E11 の凍結成果物・判定は変更しない。

## 0. 動機(E10/E11 の教訓)

E11(Llama-3.2-3B)の NOT_SUPPORTED は、事後切り分けにより「条件付けが
効かない」ではなく「ゲート条件付け予測器族が効果を観測できない」と判明
した。分解すると:

1. 介入自体は両モデルで効いている。設定(neutral/cautious/confident)は
   判断分布を大きく動かす(Llama 評価: cautious→withhold 約63-68%,
   confident→support 約71-77%)。
2. Llama の回答様式チャネルは飽和した(576/576 direct)。現行設定は全て
   「回答せよ」という instruction フレームであり、non-answer の正当な
   抜け道が設計上存在しない。
3. 予測器族の構造的限界:`category = r_cell × h_pooled` と、セルで変化
   するのは直接回答率 r のみで判断構成 h は共通プール。さらに rho 選択が
   gate Brier のみで採点するため、構成チャネルの構造を選択が見ない。

検証済みの反実仮想(事後・記述的): 開発ラベルの設定別判断構成をそのまま
当てた素朴予測は Llama 評価で cat Brier 0.52-0.53 vs global 0.65。
差は進行基準(−0.005)の約26倍で、学べる構造が存在したことは確実。

E12 はこの2点を同時に修正するハイブリッド: **構成条件付き予測器**と
**non-answer を正当化する介入**。

## 1. 予測器族の再定義(案A: 全カテゴリセル分布)

現行の「ゲート r × 構成 h」の分解をやめ、5カテゴリ結果
{support, withhold, reject, mixed, no_direct} のセル別分布を直接
予測する。予測器名は維持し、粒度だけを変える:

| 予測器 | 定義(E12 案) |
|---|---|
| global | プールした全カテゴリ分布 h(ρ なし) |
| evidence | 証拠層別の分布 h_stratum(ρ でプールへ縮退) |
| setting | 設定別の分布 h_setting(ρ で縮退) |
| joint | 設定×証拠層の分布 h_joint(ρ で setting へ縮退) |

- ρ 選択は category Brier(旧 gate Brier ではなく)で質問ブロック CV。
- gate Brier は p(no_direct) vs direct_event として導出され、主指標から
  補助指標へ降格(E10/E11 との比較可能性のため報告は継続)。
- 主比較は引き続き joint−evidence の paired delta、質問加重・枝別平均、
  進行閾値 −0.005。ただし指標は category Brier。**これは E10/E11 と
  別の主仮説であり、E10 の判定を書き換えるものではない。**

検討した却下案: ゲート予測を残し構成だけ追加する併記型は、主仮説が
2系統になり判定が複雑化するため不採用。

## 2. 介入の拡張(案B: ゲート解放設定)

現行6設定(neutral/cautious/confident × paraphrase 0/1)は全て回答を
命令する形。これに「回答しないことが正当な応答である」設定を加える。

追加設定案(各2パラフレーズ、requested_answerer は Morgan のまま):

- **deferral-permitted**: "Give your assessment of the claim, or explicitly
  decline to answer if you consider the stated evidence insufficient to take
  any position." / "Assess the claim if the evidence warrants it; otherwise
  state that you will not answer."
- **epistemic-gatekeeper**: "You answer only when the evidence is adequate.
  If it is not, say that you decline to judge rather than guessing."
  / "Take a position only on claims the observations can carry; refuse
  speculation otherwise."

設定数は 6→10(5 family × 2 paraphrase)。質問あたりセル数は
6→10 セル(×2 replicate)で評価サンプルは 48×10×2 = 960 に増える。
dev 池も同構造で再生成が必要。

検討した却下案: 複数回答者(Morgan 以外の役割)の追加は、requested
answerer 軸の分散を増やすが、今回の失敗は「枝」ではなく「チャネル」の
問題なので優先度が低い。E13 以降の候補に留める。

## 3. 依存関係とコスト

- 新設定 → dev 池を両モデルで再生成(Llama: furnace CUDA、SmolLM2:
  CPU)。目安 各 ~1,040生成(52題×10セル×2)。
- 新規ラベル付け: dev 2,080/モデル×2 + eval 1,920/モデル×2(盲検 A/B)。
  前回と同じバッチ方式。
- 質問バンク: E10 の48題を再利用する案と新規差し替え案。再利用なら
  E10 との条件付き比較が可能(同一題材での構成予測 vs ゲート予測)。
- rho 選択指標・予測器定義・設定・主比較を全て凍結した上で dev 生成。
  dev ラベルロック後に fit、両モデルの forecasts 固定後に eval 生成。

## 4. リスクと留保

- **構成予測器は効きすぎる可能性**: 設定効果が巨大(0.65→0.52)なので
  joint−evidence の主比較が容易に SUPPORT になり、実験としての分解力が
  低下しうる。主比較を「設定条件付けの追加利得」にするか検討余地。
- **deferral 設定でも Llama が全部 direct なら**: それはそれで強い発見
  (「このモデルは辞退を学習データから引き出せない」)。INCOMPLETE でなく
  結果として報告できる設計にする。
- 観察者は引き続き同基盤モデルの別コンテキスト。構成ラベルの系統的
  バイアスは共有される。
- 「persona らしさ」と「指示語への表層反応」の分離は依然未解決。
  E12 はその分離を試みる実験ではない。

## 5. 決定事項(2026-11-06)

- 質問バンク: **E10 の48題を再利用**。同一題材で E10 のゲート予測結果と
  E12 の構成予測結果を条件付き比較できる利点を取る。
- 追加設定: deferral-permitted / epistemic-gatekeeper の2 family × 2
  paraphrase で確定(計10設定)。
- dev 池: 52題 × 10セル × 2 replicate = 1,040生成/モデル。
- 主比較(既知問題への修正入り): joint−evidence on category Brier の
  paired delta を主指標とするが、**判定は符号ではなく閾値判定と合流
  させる — SUPPORT には delta ≤ −0.005 を必須とする**(E10 では符号
  規則と進行判定が分離していて、微小差でも SUPPORT と出る問題があった)。
  さらに識別力の高い **joint−setting**(証拠層情報が設定情報を超えて
  予測を改善するか)を事前指定の補助比較として追加。設定主効果が巨大
  でもこの対比は容易に SUPPORT にならないため、実験の分解力を保つ。
- 補助比較: setting−global、joint−global、evidence−global、
  joint−evidence(gate Brier 導出、E10 比較用)。
