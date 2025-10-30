# DMPNN追加特徴量対応 - 詳細仕様書

## 1. 概要

本文書は、DMPNNModelおよびDMPNNFeaturizerクラスに追加のスカラー特徴量およびベクトル特徴量を組み込むための詳細仕様を定義します。この仕様に基づき、実装者は具体的なコード変更を行うことができます。

## 2. 要件定義

### 2.1 機能要件

#### FR-1: スカラー特徴量の入力

- **FR-1.1**: DMPNNFeaturizerは、各分子に対して任意の数のスカラー特徴量を受け付けること
- **FR-1.2**: スカラー特徴量は、numpy配列またはリストとして提供できること
- **FR-1.3**: スカラー特徴量は、特徴量名と値のマッピングとして提供できること

#### FR-2: ベクトル特徴量の入力

- **FR-2.1**: DMPNNFeaturizerは、各分子に対して任意次元のベクトル特徴量を受け付けること
- **FR-2.2**: 複数種類のベクトル特徴量を同時に使用できること
- **FR-2.3**: ベクトル特徴量は、numpy配列として提供できること

#### FR-3: 特徴量の統合

- **FR-3.1**: 追加特徴量は、分子グラフ表現に統合されること
- **FR-3.2**: 3つの統合モード（後期連結、早期統合、アテンション）をサポートすること
- **FR-3.3**: デフォルトの統合モードは後期連結方式とすること

#### FR-4: 互換性

- **FR-4.1**: 追加特徴量を使用しない場合、既存のコードと完全に互換性を持つこと
- **FR-4.2**: 既存のグローバル特徴量（Morgan指紋等）との併用が可能であること

#### FR-5: バッチ処理

- **FR-5.1**: 追加特徴量を含むデータのバッチ処理が正しく動作すること
- **FR-5.2**: 可変長の追加特徴量をサポートすること（パディング処理を含む）

### 2.2 非機能要件

#### NFR-1: パフォーマンス

- **NFR-1.1**: 追加特徴量の統合による計算オーバーヘッドは、従来の処理時間の20%以内であること
- **NFR-1.2**: メモリ使用量の増加は、追加特徴量のサイズに比例すること

#### NFR-2: 保守性

- **NFR-2.1**: 既存のコードへの変更は最小限に抑えること
- **NFR-2.2**: 明確なドキュメンテーションとコメントを提供すること

#### NFR-3: 拡張性

- **NFR-3.1**: 新しい統合方式の追加が容易であること
- **NFR-3.2**: 新しい特徴量タイプの追加が容易であること

## 3. インターフェース仕様

### 3.1 DMPNNFeaturizerクラスの拡張

#### 3.1.1 コンストラクタの拡張

```python
class DMPNNFeaturizer(MolecularFeaturizer):
    def __init__(self,
                 features_generators: Optional[List[str]] = None,
                 is_adding_hs: bool = False,
                 use_original_atom_ranks: bool = False,
                 additional_scalar_features_dim: int = 0,
                 additional_vector_features_dim: int = 0):
        """
        Parameters
        ----------
        features_generators: List[str], default None
            既存のグローバル特徴量生成器のリスト
        is_adding_hs: bool, default False
            水素原子を追加するかどうか
        use_original_atom_ranks: bool, default False
            元の原子マッピングを使用するかどうか
        additional_scalar_features_dim: int, default 0
            追加スカラー特徴量の次元数
            0の場合、追加スカラー特徴量は使用されない
        additional_vector_features_dim: int, default 0
            追加ベクトル特徴量の次元数
            0の場合、追加ベクトル特徴量は使用されない
        """
```

**仕様詳細**:
- `additional_scalar_features_dim`: 
  - 型: `int`
  - 範囲: `0 <= additional_scalar_features_dim <= 1000`
  - デフォルト: `0`
  - 意味: 各分子に追加されるスカラー特徴量の数
  
- `additional_vector_features_dim`:
  - 型: `int`
  - 範囲: `0 <= additional_vector_features_dim <= 10000`
  - デフォルト: `0`
  - 意味: 追加ベクトル特徴量の合計次元数

#### 3.1.2 _featurizeメソッドの拡張

```python
def _featurize(self, 
               datapoint: RDKitMol,
               additional_scalar_features: Optional[np.ndarray] = None,
               additional_vector_features: Optional[np.ndarray] = None,
               **kwargs) -> GraphData:
    """
    Parameters
    ----------
    datapoint: RDKitMol
        RDKit分子オブジェクト
    additional_scalar_features: Optional[np.ndarray], default None
        追加スカラー特徴量
        shape: (n_scalar,) where n_scalar == additional_scalar_features_dim
    additional_vector_features: Optional[np.ndarray], default None
        追加ベクトル特徴量
        shape: (n_vector,) where n_vector == additional_vector_features_dim
    
    Returns
    -------
    graph: GraphData
        拡張された分子グラフデータ
        以下の追加属性を含む:
        - additional_scalar_features: np.ndarray, shape (n_scalar,)
        - additional_vector_features: np.ndarray, shape (n_vector,)
    """
```

**仕様詳細**:
- **入力検証**:
  - `additional_scalar_features`が提供された場合、その形状は`(additional_scalar_features_dim,)`でなければならない
  - `additional_vector_features`が提供された場合、その形状は`(additional_vector_features_dim,)`でなければならない
  - 次元が一致しない場合、`ValueError`を発生させる

- **デフォルト値の処理**:
  - `additional_scalar_features`が`None`で`additional_scalar_features_dim > 0`の場合、ゼロベクトルを使用
  - `additional_vector_features`が`None`で`additional_vector_features_dim > 0`の場合、ゼロベクトルを使用

#### 3.1.3 GraphDataオブジェクトの拡張

GraphDataクラスに以下の属性を追加します：

```python
class GraphData:
    # 既存の属性
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    
    # 新規追加の属性
    additional_scalar_features: Optional[np.ndarray] = None
    additional_vector_features: Optional[np.ndarray] = None
```

**仕様詳細**:
- `additional_scalar_features`:
  - 型: `Optional[np.ndarray]`
  - 形状: `(n_scalar,)`
  - データ型: `float32` または `float64`
  - デフォルト: `None`
  
- `additional_vector_features`:
  - 型: `Optional[np.ndarray]`
  - 形状: `(n_vector,)`
  - データ型: `float32` または `float64`
  - デフォルト: `None`

### 3.2 _MapperDMPNNクラスの拡張

#### 3.2.1 コンストラクタの拡張

```python
class _MapperDMPNN:
    def __init__(self, graph: GraphData):
        """
        Parameters
        ----------
        graph: GraphData
            拡張されたGraphDataオブジェクト
            additional_scalar_featuresとadditional_vector_features属性を含む可能性がある
        """
```

**仕様詳細**:
- `graph.additional_scalar_features`が存在する場合、それを`self.additional_scalar_features`として保存
- `graph.additional_vector_features`が存在する場合、それを`self.additional_vector_features`として保存
- 存在しない場合は、空の配列として初期化

#### 3.2.2 valuesプロパティの拡張

```python
@property
def values(self) -> Sequence[np.ndarray]:
    """
    Returns
    -------
    mappings: Sequence[np.ndarray]
        以下の順序で返される:
        - atom_features
        - f_ini_atoms_bonds
        - atom_to_incoming_bonds
        - mapping
        - global_features
        - additional_scalar_features
        - additional_vector_features
    """
    return (self.atom_features, 
            self.f_ini_atoms_bonds, 
            self.atom_to_incoming_bonds, 
            self.mapping, 
            self.global_features,
            self.additional_scalar_features,
            self.additional_vector_features)
```

### 3.3 DMPNNModelクラスの拡張

#### 3.3.1 コンストラクタの拡張

```python
class DMPNNModel(TorchModel):
    def __init__(self,
                 mode: str = 'regression',
                 n_classes: int = 3,
                 n_tasks: int = 1,
                 batch_size: int = 1,
                 global_features_size: int = 0,
                 use_default_fdim: bool = True,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 enc_hidden: int = 300,
                 depth: int = 3,
                 bias: bool = False,
                 enc_activation: str = 'relu',
                 enc_dropout_p: float = 0.0,
                 aggregation: str = 'mean',
                 aggregation_norm: Union[int, float] = 100,
                 ffn_hidden: int = 300,
                 ffn_activation: str = 'relu',
                 ffn_layers: int = 3,
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 additional_scalar_features_size: int = 0,
                 additional_vector_features_size: int = 0,
                 integration_mode: str = 'late_concat',
                 **kwargs):
        """
        Parameters
        ----------
        # ... 既存のパラメータ ...
        
        additional_scalar_features_size: int, default 0
            追加スカラー特徴量のサイズ
            0の場合、追加スカラー特徴量は使用されない
        additional_vector_features_size: int, default 0
            追加ベクトル特徴量のサイズ
            0の場合、追加ベクトル特徴量は使用されない
        integration_mode: str, default 'late_concat'
            追加特徴量の統合モード
            選択肢: 'late_concat', 'early_integration', 'attention'
        """
```

**仕様詳細**:
- `additional_scalar_features_size`:
  - 型: `int`
  - 範囲: `0 <= additional_scalar_features_size <= 1000`
  - デフォルト: `0`
  
- `additional_vector_features_size`:
  - 型: `int`
  - 範囲: `0 <= additional_vector_features_size <= 10000`
  - デフォルト: `0`
  
- `integration_mode`:
  - 型: `str`
  - 選択肢: `['late_concat', 'early_integration', 'attention']`
  - デフォルト: `'late_concat'`
  - 説明:
    - `'late_concat'`: FFN入力層で追加特徴量を連結
    - `'early_integration'`: メッセージパッシング前に追加特徴量を統合
    - `'attention'`: アテンション機構で追加特徴量を統合

#### 3.3.2 _to_pyg_graphメソッドの拡張

```python
def _to_pyg_graph(self, values: Sequence[np.ndarray]) -> _ModData:
    """
    Parameters
    ----------
    values: Sequence[np.ndarray]
        _MapperDMPNNから返された値のシーケンス
        以下を含む:
        - atom_features
        - f_ini_atoms_bonds
        - atom_to_incoming_bonds
        - mapping
        - global_features
        - additional_scalar_features
        - additional_vector_features
    
    Returns
    -------
    _ModData
        PyTorch Geometric用の拡張データオブジェクト
        以下の追加属性を含む:
        - additional_scalar_features: torch.Tensor
        - additional_vector_features: torch.Tensor
    """
```

**仕様詳細**:
- 7番目の要素（`additional_scalar_features`）と8番目の要素（`additional_vector_features`）を抽出
- PyTorch Tensorに変換し、適切なデバイスに配置
- `_ModData`オブジェクトに追加属性として設定

### 3.4 DMPNNクラスの拡張

#### 3.4.1 コンストラクタの拡張

```python
class DMPNN(nn.Module):
    def __init__(self,
                 mode: str = 'regression',
                 n_classes: int = 3,
                 n_tasks: int = 1,
                 global_features_size: int = 0,
                 use_default_fdim: bool = True,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 enc_hidden: int = 300,
                 depth: int = 3,
                 bias: bool = False,
                 enc_activation: str = 'relu',
                 enc_dropout_p: float = 0.0,
                 aggregation: str = 'mean',
                 aggregation_norm: Union[int, float] = 100,
                 ffn_hidden: int = 300,
                 ffn_activation: str = 'relu',
                 ffn_layers: int = 3,
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 additional_scalar_features_size: int = 0,
                 additional_vector_features_size: int = 0,
                 integration_mode: str = 'late_concat'):
        """
        Parameters
        ----------
        # ... 既存のパラメータ ...
        
        additional_scalar_features_size: int, default 0
            追加スカラー特徴量のサイズ
        additional_vector_features_size: int, default 0
            追加ベクトル特徴量のサイズ
        integration_mode: str, default 'late_concat'
            追加特徴量の統合モード
        """
```

**仕様詳細**:
- FFN入力サイズの計算:
  ```python
  additional_features_size = additional_scalar_features_size + additional_vector_features_size
  
  if integration_mode == 'late_concat':
      ffn_input = enc_hidden + global_features_size + additional_features_size
  elif integration_mode == 'early_integration':
      ffn_input = enc_hidden + global_features_size
  elif integration_mode == 'attention':
      ffn_input = enc_hidden + global_features_size
  ```

#### 3.4.2 forwardメソッドの拡張

```python
def forward(self, pyg_batch: Batch) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """
    Parameters
    ----------
    pyg_batch: Batch
        以下のテンソルを含むPyTorch Geometricバッチ:
        - atom_features
        - f_ini_atoms_bonds
        - atom_to_incoming_bonds
        - mapping
        - global_features
        - additional_scalar_features (オプション)
        - additional_vector_features (オプション)
    
    Returns
    -------
    output: Union[torch.Tensor, Sequence[torch.Tensor]]
        予測結果
    """
```

**仕様詳細**:

##### 後期連結方式（late_concat）

```python
# エンコーダの出力を取得
encodings = self.encoder(atom_features, f_ini_atoms_bonds, 
                        atom_to_incoming_bonds, mapping, 
                        global_features, molecules_unbatch_key)

# 追加特徴量を連結
if self.additional_scalar_features_size > 0:
    additional_scalar = pyg_batch['additional_scalar_features']
    encodings = torch.cat([encodings, additional_scalar], dim=1)

if self.additional_vector_features_size > 0:
    additional_vector = pyg_batch['additional_vector_features']
    encodings = torch.cat([encodings, additional_vector], dim=1)

# FFNに入力
output = self.ffn(encodings)
```

##### 早期統合方式（early_integration）

```python
# 追加特徴量を原子特徴量にブロードキャスト
if self.additional_scalar_features_size > 0 or self.additional_vector_features_size > 0:
    # 各分子の原子数を取得
    molecules_unbatch_key = torch.diff(pyg_batch._slice_dict['atom_features']).tolist()
    
    # 追加特徴量を結合
    additional_features = []
    if self.additional_scalar_features_size > 0:
        additional_features.append(pyg_batch['additional_scalar_features'])
    if self.additional_vector_features_size > 0:
        additional_features.append(pyg_batch['additional_vector_features'])
    
    additional_features = torch.cat(additional_features, dim=1)  # shape: (batch_size, n_add)
    
    # 各分子の原子数に応じてブロードキャスト
    augmented_features = []
    for i, n_atoms in enumerate(molecules_unbatch_key):
        # 分子iの追加特徴量をn_atoms回繰り返す
        mol_add_features = additional_features[i:i+1].expand(n_atoms, -1)
        augmented_features.append(mol_add_features)
    
    augmented_features = torch.cat(augmented_features, dim=0)  # shape: (total_atoms, n_add)
    
    # 原子特徴量に連結
    atom_features_augmented = torch.cat([atom_features, augmented_features], dim=1)
    
    # 拡張されたエンコーダを使用
    encodings = self.encoder_augmented(atom_features_augmented, f_ini_atoms_bonds, 
                                       atom_to_incoming_bonds, mapping, 
                                       global_features, molecules_unbatch_key)
else:
    encodings = self.encoder(atom_features, f_ini_atoms_bonds, 
                            atom_to_incoming_bonds, mapping, 
                            global_features, molecules_unbatch_key)

output = self.ffn(encodings)
```

##### アテンション統合方式（attention）

```python
# 構造情報のエンコーディング
encodings_struct = self.encoder(atom_features, f_ini_atoms_bonds, 
                                atom_to_incoming_bonds, mapping, 
                                global_features, molecules_unbatch_key)

# 追加特徴量の処理
if self.additional_scalar_features_size > 0 or self.additional_vector_features_size > 0:
    additional_features = []
    if self.additional_scalar_features_size > 0:
        additional_features.append(pyg_batch['additional_scalar_features'])
    if self.additional_vector_features_size > 0:
        additional_features.append(pyg_batch['additional_vector_features'])
    
    additional_features = torch.cat(additional_features, dim=1)
    
    # 追加特徴量を変換
    h_add = self.W_add(additional_features)  # shape: (batch_size, enc_hidden)
    
    # アテンションスコアを計算
    score_struct = torch.sum(self.w_struct * encodings_struct, dim=1, keepdim=True)
    score_add = torch.sum(self.w_add * h_add, dim=1, keepdim=True)
    
    # ソフトマックスで正規化
    scores = torch.cat([score_struct, score_add], dim=1)
    attention_weights = torch.softmax(scores, dim=1)
    
    alpha_struct = attention_weights[:, 0:1]
    alpha_add = attention_weights[:, 1:2]
    
    # アテンション重み付き和
    encodings = alpha_struct * encodings_struct + alpha_add * h_add
else:
    encodings = encodings_struct

output = self.ffn(encodings)
```

## 4. データフロー仕様

### 4.1 特徴化フェーズ

```
入力: 分子（SMILES/RDKitMol）+ 追加特徴量
  ↓
DMPNNFeaturizer._featurize()
  ↓
GraphData（拡張版）
  - node_features
  - edge_features
  - edge_index
  - global_features
  - additional_scalar_features ← 新規
  - additional_vector_features ← 新規
```

### 4.2 マッピングフェーズ

```
GraphData（拡張版）
  ↓
_MapperDMPNN.__init__()
  ↓
_MapperDMPNN.values
  - atom_features
  - f_ini_atoms_bonds
  - atom_to_incoming_bonds
  - mapping
  - global_features
  - additional_scalar_features ← 新規
  - additional_vector_features ← 新規
```

### 4.3 バッチ化フェーズ

```
List[GraphData]
  ↓
DMPNNModel.default_generator()
  ↓
List[_ModData]
  ↓
Batch.from_data_list()
  ↓
Batch（拡張版）
  - すべての分子の特徴量がバッチ化される
  - additional_scalar_features: (batch_size, n_scalar)
  - additional_vector_features: (batch_size, n_vector)
```

### 4.4 推論フェーズ

```
Batch（拡張版）
  ↓
DMPNN.forward()
  ├─ DMPNNEncoderLayer（構造情報のエンコード）
  │   └─ encodings_struct: (batch_size, enc_hidden + global_features_size)
  │
  ├─ 統合処理（integration_modeに応じて）
  │   ├─ late_concat: 追加特徴量を連結
  │   ├─ early_integration: 早期に統合
  │   └─ attention: アテンション機構で統合
  │
  └─ PositionwiseFeedForward
      └─ 予測結果: (batch_size, n_tasks) or (batch_size, n_tasks, n_classes)
```

## 5. エラー処理仕様

### 5.1 入力検証エラー

#### E-001: 次元不一致エラー

**条件**: 提供された追加特徴量の次元が、初期化時に指定された次元と一致しない

**エラーメッセージ**:
```
ValueError: additional_scalar_features dimension mismatch. 
Expected {expected_dim}, but got {actual_dim}.
```

**処理**: 
- 特徴化時に検証を行い、次元が一致しない場合は即座にエラーを発生させる
- エラーメッセージには期待される次元と実際の次元を明記する

#### E-002: データ型エラー

**条件**: 追加特徴量が数値型でない、またはnumpy配列に変換できない

**エラーメッセージ**:
```
TypeError: additional_scalar_features must be a numeric array. 
Got {type} instead.
```

**処理**:
- 入力データの型チェックを行う
- 可能であればnumpy配列に自動変換を試みる
- 変換できない場合はエラーを発生させる

#### E-003: 統合モード不正エラー

**条件**: 無効な`integration_mode`が指定された

**エラーメッセージ**:
```
ValueError: Invalid integration_mode '{mode}'. 
Must be one of ['late_concat', 'early_integration', 'attention'].
```

**処理**:
- DMPNNModelの初期化時に`integration_mode`を検証
- 無効な値の場合は即座にエラーを発生させる

### 5.2 実行時エラー

#### E-004: バッチサイズ不一致エラー

**条件**: バッチ内の分子数と追加特徴量の数が一致しない

**エラーメッセージ**:
```
RuntimeError: Batch size mismatch. 
Number of molecules: {n_molecules}, 
Number of additional feature vectors: {n_features}.
```

**処理**:
- バッチ化時にサイズの整合性を確認
- 不一致がある場合はエラーを発生させる

#### E-005: NaN/Inf値エラー

**条件**: 追加特徴量にNaNまたはInf値が含まれる

**警告メッセージ**:
```
Warning: NaN or Inf values detected in additional features at index {index}. 
These will be replaced with zeros.
```

**処理**:
- 警告を発し、NaN/Inf値をゼロで置換
- オプションでエラーとして扱うことも可能

## 6. パラメータ範囲と制約

### 6.1 次元数の制約

| パラメータ | 最小値 | 最大値 | 推奨範囲 |
|-----------|--------|--------|----------|
| additional_scalar_features_dim | 0 | 1000 | 1-50 |
| additional_vector_features_dim | 0 | 10000 | 10-1000 |
| 合計追加特徴量次元 | 0 | 10000 | 10-1000 |

**理由**:
- 最大値は、メモリとパフォーマンスの観点から設定
- 推奨範囲は、実用的な使用例に基づく

### 6.2 統合モードと特徴量サイズの制約

| 統合モード | 追加特徴量の最大合計サイズ | 備考 |
|-----------|---------------------------|------|
| late_concat | 10000 | メモリ効率が良い |
| early_integration | 500 | 計算コストが高いため制限 |
| attention | 2000 | アテンション計算のコスト |

### 6.3 バッチサイズの考慮

追加特徴量を使用する場合のバッチサイズ推奨値:

```
推奨バッチサイズ = max(1, min(デフォルトバッチサイズ, メモリ制限 / (分子サイズ + 追加特徴量サイズ)))
```

## 7. パフォーマンス仕様

### 7.1 計算複雑度

#### 後期連結方式

- メッセージパッシング: $O(E \cdot d_h \cdot T)$（変化なし）
- FFN: $O(B \cdot (d_h + d_g + d_a) \cdot d_{\text{FFN}} \cdot L)$
  - $B$: バッチサイズ
  - $d_a = n_s + n_v$: 追加特徴量の次元
  - 増加率: $(d_h + d_g + d_a) / (d_h + d_g)$

#### 早期統合方式

- 原子特徴量の拡張: $O(N \cdot d_a)$
  - $N$: 原子数
- メッセージパッシング: $O(E \cdot (d_h + d_a) \cdot T)$
- 増加率: $(d_h + d_a) / d_h$

#### アテンション統合方式

- 追加特徴量の変換: $O(B \cdot d_a \cdot d_h)$
- アテンションスコア計算: $O(B \cdot d_h)$
- 重み付き和: $O(B \cdot d_h)$

### 7.2 メモリ使用量

追加のメモリ使用量:

- **GraphDataオブジェクト**: $O(B \cdot d_a)$
- **バッチテンソル**: $O(B \cdot d_a)$
- **勾配**: $O(B \cdot d_a)$
- **合計**: $O(3 \cdot B \cdot d_a)$

## 8. 互換性とバージョン管理

### 8.1 後方互換性

- 追加特徴量を使用しない場合（`additional_scalar_features_size = 0` および `additional_vector_features_size = 0`）、既存のコードと完全に互換
- 既存のモデルの保存・読み込み機能は影響を受けない
- 既存のテストは全て合格すること

### 8.2 保存・読み込み仕様

#### モデルの保存

モデルの状態辞書に以下の追加情報を含める:

```python
{
    'model_state_dict': model.state_dict(),
    'additional_scalar_features_size': additional_scalar_features_size,
    'additional_vector_features_size': additional_vector_features_size,
    'integration_mode': integration_mode,
    # ... 既存のメタデータ ...
}
```

#### モデルの読み込み

```python
checkpoint = torch.load(path)
model = DMPNNModel(
    additional_scalar_features_size=checkpoint['additional_scalar_features_size'],
    additional_vector_features_size=checkpoint['additional_vector_features_size'],
    integration_mode=checkpoint['integration_mode'],
    # ... 他のパラメータ ...
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## 9. テスト要件

### 9.1 単体テスト

#### T-001: 特徴化テスト
- スカラー特徴量のみを使用した特徴化
- ベクトル特徴量のみを使用した特徴化
- スカラーとベクトル両方を使用した特徴化
- 追加特徴量なしの特徴化（後方互換性）

#### T-002: 次元検証テスト
- 正しい次元の追加特徴量を入力した場合の成功
- 誤った次元の追加特徴量を入力した場合のエラー
- 境界値テスト（0次元、最大次元）

#### T-003: バッチ処理テスト
- 単一分子のバッチ
- 複数分子のバッチ
- 可変長分子のバッチ
- 追加特徴量のバッチ化

#### T-004: 統合モードテスト
- 後期連結方式の動作確認
- 早期統合方式の動作確認
- アテンション統合方式の動作確認

### 9.2 統合テスト

#### T-005: エンドツーエンドテスト
- 特徴化からモデル推論までの完全なパイプライン
- 学習と予測の一貫性確認

#### T-006: パフォーマンステスト
- 追加特徴量使用時の処理時間測定
- メモリ使用量の測定
- スループットの比較

### 9.3 回帰テスト

#### T-007: 後方互換性テスト
- 既存のコードが正常に動作することを確認
- 追加特徴量を使用しない場合の結果が変わらないことを確認

## 10. ドキュメンテーション要件

### 10.1 コードドキュメント

- 全ての新規メソッドと拡張メソッドにdocstringを追加
- パラメータの型、範囲、デフォルト値を明記
- 使用例を含める

### 10.2 ユーザーガイド

以下の内容を含むユーザーガイドを作成:
- 追加特徴量の準備方法
- 各統合モードの使い分け
- パフォーマンスチューニングのヒント
- トラブルシューティングガイド

### 10.3 サンプルコード

以下のサンプルコードを提供:
- 基本的な使用例
- 各統合モードの使用例
- カスタム特徴量の作成例
- 大規模データセットでの使用例

## 11. まとめ

本仕様書では、DMPNNModelおよびDMPNNFeaturizerに追加特徴量を統合するための詳細な仕様を定義しました。主要な内容は以下の通りです:

1. **明確なインターフェース定義**: 新規パラメータ、メソッド、戻り値の詳細仕様
2. **3つの統合モード**: 後期連結、早期統合、アテンション統合の詳細仕様
3. **エラー処理**: 想定されるエラーケースとその処理方法
4. **パフォーマンス仕様**: 計算複雑度とメモリ使用量の分析
5. **互換性保証**: 後方互換性の維持とバージョン管理
6. **テスト要件**: 包括的なテスト項目の定義

この仕様に基づき、詳細設計書において具体的な実装方法を記述します。
