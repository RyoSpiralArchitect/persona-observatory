# 出力から作る「概念」は、何を指しているのか

今回の概念の候補は、ある既知の指示条件と証拠種類のもとで、次の出力がどの観察
カテゴリーに入りやすいかを表す確率的なプロファイルである。入力条件そのもの、
モデルが語る人物像、実際に出てくる回答の分類、内部の計算状態は、区別して扱う。

## 観察を予測する区別としての概念

ShaliziとCrutchfieldのmemoryless transductionの構成は、同じ条件付き出力分布を
与える入力を同じクラスへまとめる。出力に関係する情報を保ちながら入力を記述
する、という理論的な基盤になる。[1]

ただし、この実験は完全な分布の等しさを確かめたわけではない。有限の問題集合で、
出力全文をさらに五カテゴリーへ符号化し、そこから推定した予測を比較している。
理論上の最小十分な状態や、ニューラルネットの因果機構を発見したとは言わない。
また、information bottleneckの最適化アルゴリズムを実装した実験でもない。

測定したものを式で書くなら、出力Yそのものの完全な分布ではなく、
観察者と基準を含む符号化φを通した

    P(φ(Y, question, requested_answerer) | known instruction setting, evidence type)

である。観察基準が変われば、保持すべき区別も変わりうる。A/Bの比較は、その
測定手続きの揺れを部分的に可視化する。同じ基盤モデルと基準を共有しているため、
一致が高くても、その外側で同じ区別が成立するとは限らない。

## 生成された説明と、説明が生まれた過程

Turpinらは、GPT-3.5とClaude 1.0の特定の設定で、入力の偏りが回答を動かしても、
生成された説明がその影響を述べず、回答を理由づける例を示した。[2]
これは全ての自己報告が偽という証明ではないが、説明文を実際の内部処理の透明な
記録として扱わない理由になる。

この実験も、「何を語ったか」は観察できる一方、「その語りのとおりに内部で
考えたか」は測定していない。出力に表れた慎重さや断定を符号化し、次の回答を
予測することと、内部の推論軌道を同定することは別の主張になる。

## 二段階の座標を持つ実例

固定予測のうち、観察者A・neutral wording0・limitedのHは次のように書ける。

    direct-answer probability: 0.601139481
    conditional judgment distribution:
      support:  0.676949213
      reject:   0.212446352
      withhold: 0.104345494
      mixed:    0.006258941
    selected strengths: rho_D=4, rho_C=1

この短い符号には、「直接の判断が現れる見込み」と「現れた場合の判断の構成」が
入っている。モデル名・版、既知の条件、観察基準、適用範囲も一緒に保持して初めて、
何を予測する符号なのかが明確になる。値は新規生成を見る前に固定されており、
最終的な価値は新しい問題の予測で評価する。

これを持続する人格の座標と呼ぶには、履歴を伴う状態遷移、条件変更への応答、
別の言い回しやモデルへの移植など、追加の検証が必要になる。今回の独立した
問題提示だけでは、その時間的な性質は測れていない。

## 一次資料

[1] Shalizi, C. R. and Crutchfield, J. P. (2002; preprint 2000),
Information Bottlenecks, Causal States, and Statistical Relevance Bases:
How to Represent Relevant Information in Memoryless Transduction, section IV.
https://arxiv.org/pdf/nlin/0006025

[2] Turpin, M., Michael, J., Perez, E., and Bowman, S. R. (2023),
Language Models Don't Always Say What They Think: Unfaithful Explanations in
Chain-of-Thought Prompting.
https://papers.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf

ここでの「概念として使う」という位置づけは本実験への応用的な解釈であり、
上記論文が今回のモデルやデータに対してそれを実証したという意味ではない。
