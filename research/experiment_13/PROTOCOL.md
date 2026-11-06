# 実験13 プロトコル(事後凍結)

**注意**: 本書は実行後の記録的凍結。設計・生成・ラベルは本書発行前に
実施済みであり、pre-registration の強度は E10-12 より弱い。
ただし介入設計・ラベル手順・分離基準(JS divergence、閾値なし記述的
判定)は `docs/experiment_13_design_20261106.md` に事前記録済み。

- 介入: E12 の10設定 + lexswap_cautious/confident, gloss_cautious,
  role_alex 各2パラフレーズ(計18設定)
- 質問: E10/E12 の48題 eval、52題 dev 再利用
- 主比較: 対照 family vs 正本 family の5カテゴリ構成分布 JS divergence
  (閾値なし・記述的解釈のみ)
- 測定コード: `scripts/js_divergence.py`(本凍結で収録)
