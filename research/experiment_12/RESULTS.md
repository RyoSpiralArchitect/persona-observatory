# 実験12 結果報告(2026-11-06)

**状態**: 完了・採点済み。run: `exp12-eval-smollm2-002`(ローカル CPU float32)と
`exp12-eval-llama32-002`(furnace CUDA bf16)。両方 forecasts_locked、
観察 A/B ロック済み。分析は凍結済み `e12_report`。

## 主結果

主比較 Joint − Evidence(5カテゴリ Brier、質問加重 paired delta、
閾値合流型 −0.005):

| モデル | delta | 判定 | 枝別(a/b) |
|---|---|---|---|
| Llama-3.2-3B-Instruct | **−0.101** | SUPPORT | −0.107 / −0.096 |
| SmolLM2-360M-Instruct | **−0.013** | SUPPORT | −0.017 / −0.008 |

補助対比(category Brier):

| 対比 | llama32 | smollm2 |
|---|---|---|
| joint − setting | −0.167 | +0.005 |
| setting − global | −0.077 | −0.018 |
| joint − global | −0.245 | −0.013 |
| evidence − global | −0.144 | 0.000 |

gate(直接回答率)対比は両者で実質退化(Llama eval: 960/960 direct)。

## 解釈

E11 の「Llama で効かない」は予測器族の構造的限界だったことが確定した。
5カテゴリ分布をセル別に学ぶ予測器では、Llama 側の効果が最大になった。

**なぜ Llama の構成チャネルが太いか(開発池ラベルの事後分析)**

- Llama は証拠層×指示 family の両方に条件付けられた応答方針を示す:
  cautious×limited→withhold 94%、cautious×supportive→support 79%、
  deferral/gatekeeper セルでも層別に判断が分かれる。つまり「指示に応じた
  証拠評価ポリシー」として整合的で、予測可能な構造が大きい。
- SmolLM2 は判断構成が平たく、変動の多くが `no_direct`(指示/ルールの
  エコー、前提言い換え)側に住む。つまり変動は「証拠評価」ではなく
  「応答モードの不安定さ」が主成分で、構成予測の利得は小さい。
- 辞退系設定でも Llama は 960/960 全件 direct —「答えない」という
  様式は引き出せず、辞退は必ず withhold(判断保留の直接応答)として
  現れる。ゲート飽和は介入の強さの問題ではない。

## 留保

- 観察者は同基盤モデル別コンテキストであり、意味ラベルの系統バイアスを
  共有しうる(E9 からの既知の限界、未解決)。
- 「設定に応じた応答方針」が persona に相当するのか、指示語への表層
  反応なのかは未分離。E12 はその分離を主張しない。
- 閾値合流型判定は E10 の符号規則+進行判定より厳しい基準。E10/E11 の
  判定・成果物は不変。

## 工程記録

- eval config の seed_namespace 不一致(モデル間 sample_id 非共有)を
  prepare 時エラーで検出。共有化して v2 で design/fit/freeze を再発行、
  v1 は `*_v1_superseded` 保全。smollm2 dev 001 はソース変更で
  INCOMPLETE 終了(1032/1040)、002 完走。全試行・lock 保全。
- 観察量: dev 1,040×2枝×2モデル、eval 960×2枝×2モデル、計 8,000ラベル。
  重複トリプル競合は全件調和済み。
