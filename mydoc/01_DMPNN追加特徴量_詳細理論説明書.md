# DMPNN追加特徴量対応 - 詳細理論説明書

## 1. 概要

本文書は、Directed Message Passing Neural Network (D-MPNN) モデルにおいて、分子グラフ構造の特徴量に加えて、追加のスカラー特徴量およびベクトル特徴量を組み込むための理論的基盤を提供します。これにより、分子の物理化学的性質や実験条件などの追加情報を活用し、予測精度の向上を図ります。

## 2. 背景と動機

### 2.1 従来のDMPNNの限界

従来のD-MPNNモデルは、分子グラフの構造情報（原子特徴量、結合特徴量）のみを入力として使用します：

- **原子特徴量**: 原子番号、形式電荷、混成軌道などの133次元ベクトル
- **結合特徴量**: 結合タイプ、芳香族性、立体配置などの14次元ベクトル
- **グローバル特徴量**: Morgan指紋やRDKit記述子などの分子全体の特徴量

しかし、実際の化学・薬学的予測タスクでは、以下のような追加情報が重要となる場合があります：

1. **実験条件**: 温度、圧力、pH、溶媒特性
2. **物理化学的性質**: 分子量、LogP、極性表面積
3. **生物学的文脈**: 細胞種、投与量、時間依存性
4. **測定条件**: 測定装置、プロトコル、バッチ効果

### 2.2 追加特徴量の必要性

追加のスカラーおよびベクトル特徴量を統合することで、以下の利点が得られます：

- **予測精度の向上**: 構造情報だけでは捉えきれない文脈情報を利用
- **多様なタスクへの対応**: 条件依存的な予測タスクへの拡張
- **解釈可能性の向上**: どの追加特徴が予測に寄与しているかの分析が可能

## 3. 数学的定式化

### 3.1 基本的なD-MPNN

#### 3.1.1 記号の定義

分子グラフを $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ とします：

- $\mathcal{V}$: 原子の集合、$|\mathcal{V}| = N_{\text{atoms}}$
- $\mathcal{E}$: 結合の集合（有向グラフとして扱う）、$|\mathcal{E}| = N_{\text{bonds}}$
- $v \in \mathcal{V}$: 原子
- $e = (u, v) \in \mathcal{E}$: 原子 $u$ から原子 $v$ への有向結合

#### 3.1.2 初期特徴量

各原子 $v$ について：
$$
\mathbf{x}_v \in \mathbb{R}^{d_{\text{atom}}}
$$

各結合 $e = (u, v)$ について：
$$
\mathbf{e}_{uv} \in \mathbb{R}^{d_{\text{bond}}}
$$

ここで、デフォルト設定では：
- $d_{\text{atom}} = 133$
- $d_{\text{bond}} = 14$

#### 3.1.3 結合の初期隠れ状態

結合 $e = (u, v)$ の初期隠れ状態は、始点原子の特徴量と結合特徴量の連結として定義されます：

$$
\mathbf{h}_{uv}^{(0)} = \tau\left(\mathbf{W}_i \left[\mathbf{x}_u \oplus \mathbf{e}_{uv}\right]\right)
$$

ここで：
- $\oplus$: ベクトルの連結演算
- $\mathbf{W}_i \in \mathbb{R}^{d_h \times (d_{\text{atom}} + d_{\text{bond}})}$: 重み行列
- $\tau$: 活性化関数（ReLU等）
- $d_h$: 隠れ状態の次元数（デフォルト: 300）

### 3.2 メッセージパッシング

#### 3.2.1 メッセージの計算

深さ $t = 1, 2, \ldots, T$ における結合 $(u, v)$ へのメッセージは：

$$
\mathbf{m}_{uv}^{(t)} = \sum_{w \in \mathcal{N}(u) \setminus \{v\}} \mathbf{h}_{wu}^{(t-1)}
$$

ここで：
- $\mathcal{N}(u)$: 原子 $u$ の隣接原子集合
- $\setminus \{v\}$: 逆方向の結合を除外（メッセージの重複を防ぐ）

#### 3.2.2 隠れ状態の更新

結合の隠れ状態は、初期隠れ状態とメッセージを用いて更新されます：

$$
\mathbf{h}_{uv}^{(t)} = \tau\left(\mathbf{W}_h \left[\mathbf{h}_{uv}^{(0)} \oplus \mathbf{m}_{uv}^{(t)}\right]\right)
$$

ここで：
- $\mathbf{W}_h \in \mathbb{R}^{d_h \times (d_h + d_h)}$: 重み行列

### 3.3 原子表現の集約

#### 3.3.1 原子ごとのメッセージ集約

各原子 $v$ について、入ってくる結合からのメッセージを集約します：

$$
\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{h}_{uv}^{(T)}
$$

#### 3.3.2 原子の最終隠れ状態

原子 $v$ の最終的な隠れ状態は：

$$
\mathbf{h}_v = \tau\left(\mathbf{W}_o \left[\mathbf{x}_v \oplus \mathbf{m}_v\right]\right)
$$

ここで：
- $\mathbf{W}_o \in \mathbb{R}^{d_h \times (d_{\text{atom}} + d_h)}$: 重み行列

### 3.4 分子レベルの表現

#### 3.4.1 原子表現の集約方法

分子全体の表現は、原子の隠れ状態を集約して得られます。集約方法には以下の選択肢があります：

##### (1) 平均プーリング（Mean）

$$
\mathbf{h}_{\mathcal{G}}^{\text{struct}} = \frac{1}{N_{\text{atoms}}} \sum_{v \in \mathcal{V}} \mathbf{h}_v
$$

##### (2) 総和プーリング（Sum）

$$
\mathbf{h}_{\mathcal{G}}^{\text{struct}} = \sum_{v \in \mathcal{V}} \mathbf{h}_v
$$

##### (3) 正規化プーリング（Norm）

$$
\mathbf{h}_{\mathcal{G}}^{\text{struct}} = \frac{1}{Z} \sum_{v \in \mathcal{V}} \mathbf{h}_v
$$

ここで、$Z$ は正規化定数（デフォルト: 100）

#### 3.4.2 グローバル特徴量の連結

既存のグローバル特徴量 $\mathbf{f}_{\text{global}} \in \mathbb{R}^{d_{\text{global}}}$ を連結します：

$$
\mathbf{h}_{\mathcal{G}} = \mathbf{h}_{\mathcal{G}}^{\text{struct}} \oplus \mathbf{f}_{\text{global}}
$$

結果として、$\mathbf{h}_{\mathcal{G}} \in \mathbb{R}^{d_h + d_{\text{global}}}$

## 4. 追加特徴量の統合理論

### 4.1 追加特徴量の分類

追加特徴量を以下の2つのカテゴリに分類します：

#### 4.1.1 スカラー特徴量

スカラー特徴量は、単一の数値で表現される特徴です：

$$
\mathbf{s} = [s_1, s_2, \ldots, s_{n_s}]^T \in \mathbb{R}^{n_s}
$$

例：
- 温度: $T \in \mathbb{R}$
- pH値: $\text{pH} \in \mathbb{R}$
- 投与量: $\text{dose} \in \mathbb{R}$
- 測定時間: $t \in \mathbb{R}$

#### 4.1.2 ベクトル特徴量

ベクトル特徴量は、複数の要素を持つベクトルで表現される特徴です：

$$
\mathbf{v} = [v_1, v_2, \ldots, v_{n_v}]^T \in \mathbb{R}^{n_v}
$$

例：
- 溶媒の物性ベクトル: $\mathbf{v}_{\text{solvent}} \in \mathbb{R}^{10}$
- タンパク質の埋め込み表現: $\mathbf{v}_{\text{protein}} \in \mathbb{R}^{512}$
- 細胞の遺伝子発現プロファイル: $\mathbf{v}_{\text{cell}} \in \mathbb{R}^{1000}$
- 複数の実験条件のワンホットエンコーディング: $\mathbf{v}_{\text{condition}} \in \mathbb{R}^{k}$

### 4.2 追加特徴量の統合アプローチ

追加特徴量を統合する主要なアプローチとして、以下の3つの手法を定式化します：

#### 4.2.1 後期連結方式（Late Concatenation）

この方式では、分子グラフの表現を計算した後、Feed-Forward Network (FFN) の入力層で追加特徴量を連結します。

##### 統合された分子表現

$$
\mathbf{h}_{\mathcal{G}}^{\text{augmented}} = \mathbf{h}_{\mathcal{G}} \oplus \mathbf{s} \oplus \mathbf{v}
$$

ここで：
- $\mathbf{h}_{\mathcal{G}} \in \mathbb{R}^{d_h + d_{\text{global}}}$: 従来の分子表現
- $\mathbf{s} \in \mathbb{R}^{n_s}$: スカラー特徴量ベクトル
- $\mathbf{v} \in \mathbb{R}^{n_v}$: ベクトル特徴量

結果として：
$$
\mathbf{h}_{\mathcal{G}}^{\text{augmented}} \in \mathbb{R}^{d_h + d_{\text{global}} + n_s + n_v}
$$

##### FFNへの入力

$$
d_{\text{FFN input}} = d_h + d_{\text{global}} + n_s + n_v
$$

この方式の利点：
- **実装が簡単**: 既存のエンコーダ部分を変更せずに追加可能
- **計算効率が良い**: 追加の計算コストが最小限
- **柔軟性が高い**: 様々なタイプの追加特徴量に対応可能

#### 4.2.2 早期統合方式（Early Integration）

この方式では、メッセージパッシング前の段階で追加特徴量を組み込みます。

##### 原子特徴量への統合

追加特徴量を各原子の初期特徴量にブロードキャストして連結します：

$$
\mathbf{x}_v^{\text{augmented}} = \mathbf{x}_v \oplus \mathbf{s} \oplus \mathbf{v}, \quad \forall v \in \mathcal{V}
$$

##### 結合の初期隠れ状態の再定義

$$
\mathbf{h}_{uv}^{(0)} = \tau\left(\mathbf{W}_i^{\text{aug}} \left[\mathbf{x}_u^{\text{augmented}} \oplus \mathbf{e}_{uv}\right]\right)
$$

ここで：
$$
\mathbf{W}_i^{\text{aug}} \in \mathbb{R}^{d_h \times (d_{\text{atom}} + n_s + n_v + d_{\text{bond}})}
$$

この方式の利点：
- **情報の伝播**: 追加特徴量がメッセージパッシング全体に影響
- **表現力の向上**: 構造情報と追加情報が深く統合される

この方式の課題：
- **計算コストの増加**: すべてのメッセージパッシングステップで追加特徴量を処理
- **過学習のリスク**: パラメータ数が増加

#### 4.2.3 アテンション統合方式（Attention-based Integration）

この方式では、アテンション機構を用いて追加特徴量と分子表現を動的に統合します。

##### アテンションスコアの計算

追加特徴量の統合重みをアテンション機構で学習します：

$$
\alpha_{\text{struct}} = \frac{\exp(\mathbf{w}_{\text{struct}}^T \mathbf{h}_{\mathcal{G}}^{\text{struct}})}{\exp(\mathbf{w}_{\text{struct}}^T \mathbf{h}_{\mathcal{G}}^{\text{struct}}) + \exp(\mathbf{w}_{\text{add}}^T [\mathbf{s} \oplus \mathbf{v}])}
$$

$$
\alpha_{\text{add}} = 1 - \alpha_{\text{struct}}
$$

ここで：
- $\mathbf{w}_{\text{struct}} \in \mathbb{R}^{d_h}$: 構造特徴用のアテンション重みベクトル
- $\mathbf{w}_{\text{add}} \in \mathbb{R}^{n_s + n_v}$: 追加特徴用のアテンション重みベクトル

##### 統合された表現

まず、追加特徴量を変換します：

$$
\mathbf{h}_{\text{add}} = \mathbf{W}_{\text{add}} [\mathbf{s} \oplus \mathbf{v}]
$$

ここで、$\mathbf{W}_{\text{add}} \in \mathbb{R}^{d_h \times (n_s + n_v)}$ は変換行列です。

次に、アテンション重み付き和を計算します：

$$
\mathbf{h}_{\mathcal{G}}^{\text{attention}} = \alpha_{\text{struct}} \cdot \mathbf{h}_{\mathcal{G}}^{\text{struct}} + \alpha_{\text{add}} \cdot \mathbf{h}_{\text{add}}
$$

最後に、グローバル特徴量を連結します：

$$
\mathbf{h}_{\mathcal{G}}^{\text{augmented}} = \mathbf{h}_{\mathcal{G}}^{\text{attention}} \oplus \mathbf{f}_{\text{global}}
$$

この方式の利点：
- **適応的統合**: サンプルごとに最適な特徴量の重み付けを学習
- **解釈可能性**: どの特徴量が重要かを定量的に評価可能

### 4.3 Feed-Forward Network（FFN）

#### 4.3.1 基本構造

FFNは、統合された分子表現から最終的な予測を行います：

$$
\mathbf{z}^{(0)} = \mathbf{h}_{\mathcal{G}}^{\text{augmented}}
$$

各層 $l = 1, 2, \ldots, L$ について：

$$
\mathbf{z}^{(l)} = \tau\left(\mathbf{W}_l \mathbf{z}^{(l-1)} + \mathbf{b}_l\right)
$$

ここで：
- $\mathbf{W}_l \in \mathbb{R}^{d_{\text{FFN}} \times d_{\text{input}}^{(l)}}$: 第 $l$ 層の重み行列
- $\mathbf{b}_l \in \mathbb{R}^{d_{\text{FFN}}}$: バイアスベクトル
- $\tau$: 活性化関数（ReLU等）
- $d_{\text{FFN}}$: FFNの隠れ層の次元数（デフォルト: 300）

#### 4.3.2 ドロップアウトの適用

過学習を防ぐため、各層にドロップアウトを適用します：

$$
\mathbf{z}_{\text{dropout}}^{(l)} = \text{Dropout}(\mathbf{z}^{(l)}, p)
$$

ここで、$p$ はドロップアウト確率（デフォルト: 0.0）

#### 4.3.3 出力層

##### 回帰タスク

$$
\hat{\mathbf{y}} = \mathbf{W}_{\text{out}} \mathbf{z}^{(L)} + \mathbf{b}_{\text{out}}
$$

ここで：
- $\mathbf{W}_{\text{out}} \in \mathbb{R}^{n_{\text{tasks}} \times d_{\text{FFN}}}$: 出力重み行列
- $\mathbf{b}_{\text{out}} \in \mathbb{R}^{n_{\text{tasks}}}$: 出力バイアス
- $n_{\text{tasks}}$: 予測タスク数

##### 分類タスク

ロジット：
$$
\mathbf{logits} = \mathbf{W}_{\text{out}} \mathbf{z}^{(L)} + \mathbf{b}_{\text{out}}
$$

ここで、$\mathbf{W}_{\text{out}} \in \mathbb{R}^{(n_{\text{tasks}} \times n_{\text{classes}}) \times d_{\text{FFN}}}$

ソフトマックス変換：
$$
\hat{\mathbf{y}}_c = \text{softmax}(\mathbf{logits})_c = \frac{\exp(\mathbf{logits}_c)}{\sum_{c'=1}^{n_{\text{classes}}} \exp(\mathbf{logits}_{c'})}
$$

## 5. 損失関数

### 5.1 回帰タスク

L2損失（平均二乗誤差）を使用します：

$$
\mathcal{L}_{\text{regression}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{n_{\text{tasks}}} w_i^{(j)} \left(\hat{y}_i^{(j)} - y_i^{(j)}\right)^2
$$

ここで：
- $N$: バッチサイズ
- $\hat{y}_i^{(j)}$: 第 $i$ サンプルの第 $j$ タスクの予測値
- $y_i^{(j)}$: 第 $i$ サンプルの第 $j$ タスクの真の値
- $w_i^{(j)}$: サンプル重み（デフォルト: 1）

### 5.2 分類タスク

スパースソフトマックス交差エントロピー損失を使用します：

$$
\mathcal{L}_{\text{classification}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{n_{\text{tasks}}} w_i^{(j)} \log(\hat{y}_i^{(j, y_i)})
$$

ここで：
- $y_i$: 第 $i$ サンプルの真のクラスラベル
- $\hat{y}_i^{(j, y_i)}$: 第 $i$ サンプルの第 $j$ タスクにおける真のクラスの予測確率

## 6. 正規化と正規化手法

### 6.1 特徴量の正規化

追加特徴量は、スケールが大きく異なる可能性があるため、正規化が重要です。

#### 6.1.1 標準化（Z-score normalization）

$$
s_k^{\text{norm}} = \frac{s_k - \mu_{s_k}}{\sigma_{s_k}}
$$

$$
v_{k,l}^{\text{norm}} = \frac{v_{k,l} - \mu_{v_{k,l}}}{\sigma_{v_{k,l}}}
$$

ここで：
- $\mu_{s_k}$: スカラー特徴量 $k$ の訓練データにおける平均
- $\sigma_{s_k}$: スカラー特徴量 $k$ の訓練データにおける標準偏差
- $\mu_{v_{k,l}}$: ベクトル特徴量 $k$ の第 $l$ 要素の平均
- $\sigma_{v_{k,l}}$: ベクトル特徴量 $k$ の第 $l$ 要素の標準偏差

#### 6.1.2 最小-最大正規化（Min-Max normalization）

$$
s_k^{\text{norm}} = \frac{s_k - \min(s_k)}{\max(s_k) - \min(s_k)}
$$

### 6.2 重みの正則化

過学習を防ぐため、L2正則化を適用します：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_{l} \|\mathbf{W}_l\|_F^2
$$

ここで：
- $\mathcal{L}_{\text{task}}$: タスク固有の損失（回帰または分類）
- $\lambda$: 正則化係数
- $\|\mathbf{W}_l\|_F^2$: フロベニウスノルムの二乗

## 7. バッチ処理における実装上の考慮事項

### 7.1 可変長グラフのバッチ化

分子グラフは原子数や結合数が異なるため、バッチ処理では特別な処理が必要です。

#### 7.1.1 グラフのバッチ化

バッチ内の $K$ 個の分子グラフ $\{\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_K\}$ を1つの大きなグラフに統合します：

$$
\mathcal{G}_{\text{batch}} = \bigoplus_{k=1}^{K} \mathcal{G}_k
$$

ここで、$\bigoplus$ はグラフの非連結和（disjoint union）を表します。

#### 7.1.2 原子インデックスのオフセット

バッチ化されたグラフにおいて、各分子の原子インデックスをオフセットする必要があります：

分子 $k$ の原子 $v$ のバッチ内インデックス：
$$
v_{\text{batch}}^{(k)} = v + \sum_{i=1}^{k-1} N_{\text{atoms}}^{(i)}
$$

#### 7.1.3 結合マッピングの調整

結合のマッピングも同様にオフセットが必要です：

$$
\text{mapping}_{\text{batch}}^{(k)} = \text{mapping}^{(k)} + \sum_{i=1}^{k-1} N_{\text{bonds}}^{(i)}
$$

ここで、無効なインデックス（-1）はそのままにします。

### 7.2 追加特徴量のバッチ化

#### 7.2.1 スカラーおよびベクトル特徴量

追加特徴量は分子ごとに定義されるため、単純にスタックできます：

$$
\mathbf{S}_{\text{batch}} = \begin{bmatrix} \mathbf{s}_1^T \\ \mathbf{s}_2^T \\ \vdots \\ \mathbf{s}_K^T \end{bmatrix} \in \mathbb{R}^{K \times n_s}
$$

$$
\mathbf{V}_{\text{batch}} = \begin{bmatrix} \mathbf{v}_1^T \\ \mathbf{v}_2^T \\ \vdots \\ \mathbf{v}_K^T \end{bmatrix} \in \mathbb{R}^{K \times n_v}
$$

### 7.3 出力のアンバッチ化

バッチ処理後、各分子の予測結果を抽出する必要があります。分子ごとの原子数情報を用いて、分子単位で予測結果を分割します。

## 8. 数値安定性と最適化

### 8.1 勾配クリッピング

勾配爆発を防ぐため、勾配クリッピングを適用します：

$$
\mathbf{g}_{\text{clipped}} = \begin{cases} 
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\
\theta \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{otherwise}
\end{cases}
$$

ここで、$\theta$ はクリッピング閾値です。

### 8.2 学習率スケジューリング

学習の安定化のため、学習率を動的に調整します：

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{\text{decay}} \rfloor}
$$

ここで：
- $\eta_0$: 初期学習率
- $\gamma$: 減衰率（例: 0.9）
- $T_{\text{decay}}$: 減衰間隔（エポック数）

## 9. 理論的保証と収束性

### 9.1 メッセージパッシングの収束

有限深度 $T$ において、メッセージパッシングは各結合の $T$-hop 近傍の情報を集約します。これは以下の定理によって保証されます：

**定理1**: 深さ $T$ のメッセージパッシングにより、各原子は自身から $T$ ホップ以内の部分構造の情報を含む表現を獲得する。

証明のスケッチ：
- $t=0$: 各結合は1-hop の情報（自身の始点原子と結合）を持つ
- $t=1$: 各結合は2-hop の情報を持つ（メッセージ経由）
- 帰納的に、$t=T$ で $T$-hop の情報を持つ

### 9.2 追加特徴量の寄与

追加特徴量の寄与は、以下の分解定理によって理解できます：

**定理2**: 後期連結方式において、予測 $\hat{y}$ は構造情報による寄与と追加特徴量による寄与に分解できる（線形近似の下で）。

$$
\hat{y} \approx \hat{y}_{\text{struct}} + \hat{y}_{\text{add}}
$$

ここで：
- $\hat{y}_{\text{struct}}$: 構造情報のみによる予測
- $\hat{y}_{\text{add}}$: 追加特徴量による補正項

## 10. まとめ

本理論説明書では、D-MPNNモデルに追加のスカラーおよびベクトル特徴量を統合するための数学的基盤を詳細に記述しました。主要な貢献は以下の通りです：

1. **3つの統合アプローチの定式化**:
   - 後期連結方式（実装が簡単で効率的）
   - 早期統合方式（情報の深い統合）
   - アテンション統合方式（適応的で解釈可能）

2. **完全な数式の提供**:
   - メッセージパッシングから最終予測まで、全ての計算ステップを数式で記述
   - バッチ処理における実装上の考慮事項を含む

3. **理論的正当性**:
   - メッセージパッシングの収束性
   - 追加特徴量の寄与の分解

この理論的基盤に基づき、詳細仕様書および詳細設計書において、具体的な実装方法を記述します。
