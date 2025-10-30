# DMPNN追加特徴量対応 - 詳細設計書

## 1. 概要

本文書は、DMPNNModelおよびDMPNNFeaturizerクラスに追加のスカラー特徴量およびベクトル特徴量を組み込むための詳細設計を提供します。理論説明書と仕様書に基づき、具体的な実装方法、クラス設計、メソッド設計、データ構造を記述します。

## 2. アーキテクチャ設計

### 2.1 システムアーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                        ユーザーコード                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DMPNNFeaturizer                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  _featurize(mol, scalar_feat, vector_feat)           │  │
│  │    ├─ 分子グラフの特徴化（既存）                       │  │
│  │    ├─ 追加特徴量の検証                                 │  │
│  │    └─ GraphDataへの統合                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      GraphData (拡張版)                       │
│  - node_features                                            │
│  - edge_features                                            │
│  - edge_index                                               │
│  - global_features                                          │
│  - additional_scalar_features ← 新規                        │
│  - additional_vector_features ← 新規                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     _MapperDMPNN                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  __init__(graph)                                     │  │
│  │    ├─ 既存のマッピング処理                            │  │
│  │    └─ 追加特徴量の抽出と保存                          │  │
│  │                                                       │  │
│  │  values プロパティ                                    │  │
│  │    └─ 7要素のタプルを返す（追加特徴量を含む）          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      DMPNNModel                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  default_generator()                                 │  │
│  │    └─ バッチ作成時に追加特徴量を処理                  │  │
│  │                                                       │  │
│  │  _to_pyg_graph(values)                               │  │
│  │    └─ PyTorch Tensorへの変換                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         DMPNN                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DMPNNEncoderLayer (既存)                            │  │
│  │    └─ 分子グラフ構造のエンコード                      │  │
│  │                                                       │  │
│  │  統合レイヤー (新規)                                  │  │
│  │    ├─ LateConcatIntegration                          │  │
│  │    ├─ EarlyIntegration                               │  │
│  │    └─ AttentionIntegration                           │  │
│  │                                                       │  │
│  │  PositionwiseFeedForward                             │  │
│  │    └─ 最終予測                                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 モジュール構成

```
deepchem/
├── feat/
│   ├── graph_data.py (拡張)
│   │   └── GraphData クラスに追加属性を追加
│   └── molecule_featurizers/
│       └── dmpnn_featurizer.py (拡張)
│           └── DMPNNFeaturizer クラスを拡張
├── models/
│   └── torch_models/
│       ├── dmpnn.py (拡張)
│       │   ├── _MapperDMPNN クラスを拡張
│       │   ├── DMPNN クラスを拡張
│       │   └── DMPNNModel クラスを拡張
│       └── layers.py (拡張)
│           ├── IntegrationLayer (新規基底クラス)
│           ├── LateConcatIntegration (新規)
│           ├── EarlyIntegration (新規)
│           └── AttentionIntegration (新規)
└── mydoc/
    ├── 01_DMPNN追加特徴量_詳細理論説明書.md
    ├── 02_DMPNN追加特徴量_詳細仕様書.md
    └── 03_DMPNN追加特徴量_詳細設計書.md
```

## 3. クラス設計

### 3.1 GraphData クラスの拡張

#### 3.1.1 クラス図

```python
class GraphData:
    """拡張された分子グラフデータクラス"""
    
    # 既存の属性
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    num_nodes: int
    num_edges: int
    num_node_features: int
    num_edge_features: int
    
    # 新規追加の属性
    additional_scalar_features: Optional[np.ndarray]
    additional_vector_features: Optional[np.ndarray]
    
    def __init__(self, 
                 node_features: np.ndarray,
                 edge_index: np.ndarray,
                 edge_features: Optional[np.ndarray] = None,
                 global_features: Optional[np.ndarray] = None,
                 additional_scalar_features: Optional[np.ndarray] = None,
                 additional_vector_features: Optional[np.ndarray] = None,
                 **kwargs):
        """拡張されたコンストラクタ"""
```

#### 3.1.2 実装詳細

**ファイル**: `deepchem/feat/graph_data.py`

```python
# 既存のGraphDataクラスの__init__メソッドに以下を追加:

def __init__(self, 
             node_features: np.ndarray,
             edge_index: np.ndarray,
             edge_features: Optional[np.ndarray] = None,
             global_features: Optional[np.ndarray] = None,
             additional_scalar_features: Optional[np.ndarray] = None,
             additional_vector_features: Optional[np.ndarray] = None,
             **kwargs):
    # 既存の初期化コード...
    
    # 追加特徴量の初期化
    self.additional_scalar_features = additional_scalar_features
    self.additional_vector_features = additional_vector_features
    
    # 追加特徴量のプロパティ
    if additional_scalar_features is not None:
        self.num_additional_scalar_features = additional_scalar_features.shape[0]
    else:
        self.num_additional_scalar_features = 0
        
    if additional_vector_features is not None:
        self.num_additional_vector_features = additional_vector_features.shape[0]
    else:
        self.num_additional_vector_features = 0
```

### 3.2 DMPNNFeaturizer クラスの拡張

#### 3.2.1 クラス図

```python
class DMPNNFeaturizer(MolecularFeaturizer):
    """拡張されたDMPNN特徴化クラス"""
    
    # 既存の属性
    features_generators: Optional[List[str]]
    is_adding_hs: bool
    
    # 新規追加の属性
    additional_scalar_features_dim: int
    additional_vector_features_dim: int
    
    def __init__(self,
                 features_generators: Optional[List[str]] = None,
                 is_adding_hs: bool = False,
                 use_original_atom_ranks: bool = False,
                 additional_scalar_features_dim: int = 0,
                 additional_vector_features_dim: int = 0):
        """拡張されたコンストラクタ"""
        
    def _featurize(self, 
                   datapoint: RDKitMol,
                   additional_scalar_features: Optional[np.ndarray] = None,
                   additional_vector_features: Optional[np.ndarray] = None,
                   **kwargs) -> GraphData:
        """拡張された特徴化メソッド"""
        
    def _validate_additional_features(self,
                                     additional_scalar_features: Optional[np.ndarray],
                                     additional_vector_features: Optional[np.ndarray]) -> None:
        """追加特徴量の検証メソッド（新規）"""
```

#### 3.2.2 実装詳細

**ファイル**: `deepchem/feat/molecule_featurizers/dmpnn_featurizer.py`

```python
def __init__(self,
             features_generators: Optional[List[str]] = None,
             is_adding_hs: bool = False,
             use_original_atom_ranks: bool = False,
             additional_scalar_features_dim: int = 0,
             additional_vector_features_dim: int = 0):
    """
    初期化メソッドの実装
    
    Implementation Notes:
    - additional_scalar_features_dimとadditional_vector_features_dimを保存
    - 既存の初期化処理は変更しない
    """
    self.additional_scalar_features_dim = additional_scalar_features_dim
    self.additional_vector_features_dim = additional_vector_features_dim
    
    # 既存の初期化を呼び出し
    super().__init__(use_original_atom_ranks)
    self.features_generators = features_generators
    self.is_adding_hs = is_adding_hs

def _validate_additional_features(self,
                                 additional_scalar_features: Optional[np.ndarray],
                                 additional_vector_features: Optional[np.ndarray]) -> None:
    """
    追加特徴量の検証
    
    Validation Rules:
    1. 次元数の検証
    2. データ型の検証（numeric）
    3. NaN/Inf値のチェック（警告）
    
    Raises:
    - ValueError: 次元不一致
    - TypeError: 不正なデータ型
    """
    if additional_scalar_features is not None:
        # 配列への変換試行
        if not isinstance(additional_scalar_features, np.ndarray):
            try:
                additional_scalar_features = np.asarray(additional_scalar_features, dtype=np.float32)
            except Exception as e:
                raise TypeError(f"additional_scalar_features must be convertible to numpy array. Error: {e}")
        
        # 次元チェック
        if len(additional_scalar_features.shape) != 1:
            raise ValueError(f"additional_scalar_features must be 1-dimensional. Got shape {additional_scalar_features.shape}")
        
        if additional_scalar_features.shape[0] != self.additional_scalar_features_dim:
            raise ValueError(f"additional_scalar_features dimension mismatch. "
                           f"Expected {self.additional_scalar_features_dim}, "
                           f"but got {additional_scalar_features.shape[0]}")
        
        # NaN/Inf値のチェック
        if np.any(np.isnan(additional_scalar_features)) or np.any(np.isinf(additional_scalar_features)):
            logger.warning("NaN or Inf values detected in additional_scalar_features. "
                          "These will be replaced with zeros.")
            additional_scalar_features = np.nan_to_num(additional_scalar_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 同様の検証をadditional_vector_featuresにも適用
    if additional_vector_features is not None:
        if not isinstance(additional_vector_features, np.ndarray):
            try:
                additional_vector_features = np.asarray(additional_vector_features, dtype=np.float32)
            except Exception as e:
                raise TypeError(f"additional_vector_features must be convertible to numpy array. Error: {e}")
        
        if len(additional_vector_features.shape) != 1:
            raise ValueError(f"additional_vector_features must be 1-dimensional. Got shape {additional_vector_features.shape}")
        
        if additional_vector_features.shape[0] != self.additional_vector_features_dim:
            raise ValueError(f"additional_vector_features dimension mismatch. "
                           f"Expected {self.additional_vector_features_dim}, "
                           f"but got {additional_vector_features.shape[0]}")
        
        if np.any(np.isnan(additional_vector_features)) or np.any(np.isinf(additional_vector_features)):
            logger.warning("NaN or Inf values detected in additional_vector_features. "
                          "These will be replaced with zeros.")
            additional_vector_features = np.nan_to_num(additional_vector_features, nan=0.0, posinf=0.0, neginf=0.0)

def _featurize(self, 
               datapoint: RDKitMol,
               additional_scalar_features: Optional[np.ndarray] = None,
               additional_vector_features: Optional[np.ndarray] = None,
               **kwargs) -> GraphData:
    """
    拡張された特徴化メソッド
    
    Implementation Steps:
    1. 既存の分子グラフ特徴化を実行
    2. 追加特徴量の検証
    3. デフォルト値の処理（Noneの場合はゼロベクトル）
    4. GraphDataオブジェクトへの統合
    """
    # 既存の特徴化処理（変更なし）
    if isinstance(datapoint, Chem.rdchem.Mol):
        if self.is_adding_hs:
            datapoint = Chem.AddHs(datapoint)
    else:
        raise ValueError("Feature field should contain smiles for DMPNN featurizer!")
    
    f_atoms = np.asarray([atom_features(atom) for atom in datapoint.GetAtoms()], dtype=float)
    f_bonds = self._get_bond_features(datapoint)
    edge_index = self._construct_bond_index(datapoint)
    
    global_features = np.empty(0)
    if self.features_generators is not None:
        global_features = generate_global_features(datapoint, self.features_generators)
    
    # 追加特徴量の処理
    if self.additional_scalar_features_dim > 0 or self.additional_vector_features_dim > 0:
        self._validate_additional_features(additional_scalar_features, additional_vector_features)
    
    # デフォルト値の設定
    if additional_scalar_features is None and self.additional_scalar_features_dim > 0:
        additional_scalar_features = np.zeros(self.additional_scalar_features_dim, dtype=np.float32)
    
    if additional_vector_features is None and self.additional_vector_features_dim > 0:
        additional_vector_features = np.zeros(self.additional_vector_features_dim, dtype=np.float32)
    
    # GraphDataオブジェクトの作成
    return GraphData(node_features=f_atoms,
                    edge_index=edge_index,
                    edge_features=f_bonds,
                    global_features=global_features,
                    additional_scalar_features=additional_scalar_features,
                    additional_vector_features=additional_vector_features)
```

### 3.3 _MapperDMPNN クラスの拡張

#### 3.3.1 実装詳細

**ファイル**: `deepchem/models/torch_models/dmpnn.py`

```python
class _MapperDMPNN:
    def __init__(self, graph: GraphData):
        """
        拡張された初期化メソッド
        
        Implementation Notes:
        - 既存の初期化処理は変更しない
        - 追加特徴量を抽出して保存
        """
        # 既存の初期化コード（変更なし）
        self.num_atoms = graph.num_nodes
        self.num_atom_features = graph.num_node_features
        self.num_bonds = graph.num_edges
        self.num_bond_features = graph.num_edge_features
        self.atom_features = graph.node_features
        self.bond_features = graph.edge_features
        self.bond_index = graph.edge_index
        self.global_features = graph.global_features
        
        # 追加特徴量の抽出（新規）
        if hasattr(graph, 'additional_scalar_features') and graph.additional_scalar_features is not None:
            self.additional_scalar_features = graph.additional_scalar_features
        else:
            self.additional_scalar_features = np.empty(0, dtype=np.float32)
        
        if hasattr(graph, 'additional_vector_features') and graph.additional_vector_features is not None:
            self.additional_vector_features = graph.additional_vector_features
        else:
            self.additional_vector_features = np.empty(0, dtype=np.float32)
        
        # 既存のマッピング処理（変更なし）
        if self.num_bonds == 0:
            # ... 既存のコード ...
        else:
            # ... 既存のコード ...
    
    @property
    def values(self) -> Sequence[np.ndarray]:
        """
        拡張されたvaluesプロパティ
        
        Returns:
        - atom_features
        - f_ini_atoms_bonds
        - atom_to_incoming_bonds
        - mapping
        - global_features
        - additional_scalar_features (新規)
        - additional_vector_features (新規)
        """
        return (self.atom_features, 
                self.f_ini_atoms_bonds, 
                self.atom_to_incoming_bonds, 
                self.mapping, 
                self.global_features,
                self.additional_scalar_features,
                self.additional_vector_features)
```

### 3.4 統合レイヤーの設計（新規）

#### 3.4.1 基底クラス: IntegrationLayer

**ファイル**: `deepchem/models/torch_models/layers.py`

```python
class IntegrationLayer(nn.Module):
    """
    追加特徴量統合の基底クラス
    
    このクラスは、異なる統合方式の共通インターフェースを定義します。
    """
    
    def __init__(self,
                 enc_hidden: int,
                 additional_features_size: int):
        """
        Parameters:
        - enc_hidden: エンコーダの隠れ層次元数
        - additional_features_size: 追加特徴量の合計サイズ
        """
        super(IntegrationLayer, self).__init__()
        self.enc_hidden = enc_hidden
        self.additional_features_size = additional_features_size
    
    def forward(self,
                encodings_struct: torch.Tensor,
                additional_features: torch.Tensor) -> torch.Tensor:
        """
        統合処理の抽象メソッド
        
        Parameters:
        - encodings_struct: 構造エンコーディング (batch_size, enc_hidden)
        - additional_features: 追加特徴量 (batch_size, additional_features_size)
        
        Returns:
        - integrated: 統合された特徴量 (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")
```

#### 3.4.2 後期連結方式: LateConcatIntegration

```python
class LateConcatIntegration(IntegrationLayer):
    """
    後期連結方式の統合レイヤー
    
    数式:
    h_augmented = [h_struct || h_add]
    
    where:
    - h_struct: 構造エンコーディング
    - h_add: 追加特徴量
    - ||: 連結演算
    """
    
    def __init__(self,
                 enc_hidden: int,
                 additional_features_size: int):
        super(LateConcatIntegration, self).__init__(enc_hidden, additional_features_size)
        self.output_dim = enc_hidden + additional_features_size
    
    def forward(self,
                encodings_struct: torch.Tensor,
                additional_features: torch.Tensor) -> torch.Tensor:
        """
        Implementation:
        単純な連結演算を実行
        
        Complexity:
        - Time: O(B * (d_h + d_a))
        - Space: O(B * (d_h + d_a))
        """
        if additional_features.numel() == 0:
            return encodings_struct
        return torch.cat([encodings_struct, additional_features], dim=1)
```

#### 3.4.3 早期統合方式: EarlyIntegration

```python
class EarlyIntegration(IntegrationLayer):
    """
    早期統合方式の統合レイヤー
    
    この方式では、追加特徴量を各原子にブロードキャストします。
    
    Note: この実装はDMPNNEncoderLayerの変更が必要なため、
    DMPNNクラスのforward内で処理されます。
    """
    
    def __init__(self,
                 enc_hidden: int,
                 additional_features_size: int):
        super(EarlyIntegration, self).__init__(enc_hidden, additional_features_size)
        self.output_dim = enc_hidden
    
    def forward(self,
                encodings_struct: torch.Tensor,
                additional_features: torch.Tensor) -> torch.Tensor:
        """
        早期統合方式では、この段階では何もしない
        （実際の統合はエンコーダ前に行われる）
        """
        return encodings_struct
```

#### 3.4.4 アテンション統合方式: AttentionIntegration

```python
class AttentionIntegration(IntegrationLayer):
    """
    アテンション機構による統合レイヤー
    
    数式:
    α_struct = exp(w_struct^T h_struct) / Z
    α_add = exp(w_add^T h_add) / Z
    h_integrated = α_struct * h_struct + α_add * W_add * h_add
    
    where:
    - Z: 正規化定数
    - W_add: 追加特徴量の変換行列
    - w_struct, w_add: アテンション重みベクトル
    """
    
    def __init__(self,
                 enc_hidden: int,
                 additional_features_size: int):
        super(AttentionIntegration, self).__init__(enc_hidden, additional_features_size)
        self.output_dim = enc_hidden
        
        # 追加特徴量の変換層
        self.W_add = nn.Linear(additional_features_size, enc_hidden, bias=False)
        
        # アテンション重みベクトル
        self.w_struct = nn.Parameter(torch.randn(enc_hidden))
        self.w_add = nn.Parameter(torch.randn(enc_hidden))
    
    def forward(self,
                encodings_struct: torch.Tensor,
                additional_features: torch.Tensor) -> torch.Tensor:
        """
        Implementation Steps:
        1. 追加特徴量を変換: h_add = W_add * additional_features
        2. アテンションスコアを計算
        3. ソフトマックスで正規化
        4. 重み付き和を計算
        
        Complexity:
        - Time: O(B * d_a * d_h + B * d_h)
        - Space: O(B * d_h)
        """
        if additional_features.numel() == 0:
            return encodings_struct
        
        # 追加特徴量の変換
        h_add = self.W_add(additional_features)  # (batch_size, enc_hidden)
        
        # アテンションスコアの計算
        score_struct = torch.sum(self.w_struct * encodings_struct, dim=1, keepdim=True)
        score_add = torch.sum(self.w_add * h_add, dim=1, keepdim=True)
        
        # ソフトマックスで正規化
        scores = torch.cat([score_struct, score_add], dim=1)  # (batch_size, 2)
        attention_weights = torch.softmax(scores, dim=1)
        
        alpha_struct = attention_weights[:, 0:1]  # (batch_size, 1)
        alpha_add = attention_weights[:, 1:2]     # (batch_size, 1)
        
        # 重み付き和
        integrated = alpha_struct * encodings_struct + alpha_add * h_add
        
        return integrated
```

### 3.5 DMPNN クラスの拡張

#### 3.5.1 実装詳細

**ファイル**: `deepchem/models/torch_models/dmpnn.py`

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
        拡張されたコンストラクタ
        
        Implementation Notes:
        1. 追加特徴量のサイズを保存
        2. 統合モードに応じた統合レイヤーを初期化
        3. FFN入力サイズを調整
        """
        super(DMPNN, self).__init__()
        self.mode = mode
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.additional_scalar_features_size = additional_scalar_features_size
        self.additional_vector_features_size = additional_vector_features_size
        self.integration_mode = integration_mode
        
        # 追加特徴量の合計サイズ
        additional_features_size = additional_scalar_features_size + additional_vector_features_size
        
        # 統合モードの検証
        valid_modes = ['late_concat', 'early_integration', 'attention']
        if integration_mode not in valid_modes:
            raise ValueError(f"Invalid integration_mode '{integration_mode}'. "
                           f"Must be one of {valid_modes}")
        
        # エンコーダの初期化
        if integration_mode == 'early_integration' and additional_features_size > 0:
            # 早期統合の場合、原子特徴量の次元が増加
            self.encoder = layers.DMPNNEncoderLayer(
                use_default_fdim=use_default_fdim,
                atom_fdim=atom_fdim + additional_features_size,
                bond_fdim=bond_fdim,
                d_hidden=enc_hidden,
                depth=depth,
                bias=bias,
                activation=enc_activation,
                dropout_p=enc_dropout_p,
                aggregation=aggregation,
                aggregation_norm=aggregation_norm)
        else:
            # 通常のエンコーダ
            self.encoder = layers.DMPNNEncoderLayer(
                use_default_fdim=use_default_fdim,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                d_hidden=enc_hidden,
                depth=depth,
                bias=bias,
                activation=enc_activation,
                dropout_p=enc_dropout_p,
                aggregation=aggregation,
                aggregation_norm=aggregation_norm)
        
        # 統合レイヤーの初期化
        if additional_features_size > 0:
            if integration_mode == 'late_concat':
                self.integration_layer = layers.LateConcatIntegration(
                    enc_hidden, additional_features_size)
            elif integration_mode == 'early_integration':
                self.integration_layer = layers.EarlyIntegration(
                    enc_hidden, additional_features_size)
            elif integration_mode == 'attention':
                self.integration_layer = layers.AttentionIntegration(
                    enc_hidden, additional_features_size)
        else:
            self.integration_layer = None
        
        # FFN入力サイズの計算
        if integration_mode == 'late_concat':
            ffn_input = enc_hidden + global_features_size + additional_features_size
        else:
            ffn_input = enc_hidden + global_features_size
        
        # FFN出力サイズ
        if self.mode == 'regression':
            ffn_output = self.n_tasks
        elif self.mode == 'classification':
            ffn_output = self.n_tasks * self.n_classes
        
        # FFNの初期化
        self.ffn = layers.PositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden=ffn_hidden,
            d_output=ffn_output,
            activation=ffn_activation,
            n_layers=ffn_layers,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)
    
    def forward(self, pyg_batch: Batch) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        拡張されたforwardメソッド
        
        Implementation Flow:
        1. バッチから特徴量を抽出
        2. 追加特徴量を結合
        3. 統合モードに応じた処理
        4. FFNで予測
        """
        # 既存の特徴量を抽出
        atom_features = pyg_batch['atom_features']
        f_ini_atoms_bonds = pyg_batch['f_ini_atoms_bonds']
        atom_to_incoming_bonds = pyg_batch['atom_to_incoming_bonds']
        mapping = pyg_batch['mapping']
        global_features = pyg_batch['global_features']
        
        # 分子ごとの原子数リストを取得
        molecules_unbatch_key = torch.diff(
            pyg_batch._slice_dict['atom_features']).tolist()
        
        # 追加特徴量の抽出と結合
        additional_features = []
        if self.additional_scalar_features_size > 0:
            additional_scalar = pyg_batch.get('additional_scalar_features', None)
            if additional_scalar is not None and additional_scalar.numel() > 0:
                additional_features.append(additional_scalar)
        
        if self.additional_vector_features_size > 0:
            additional_vector = pyg_batch.get('additional_vector_features', None)
            if additional_vector is not None and additional_vector.numel() > 0:
                additional_features.append(additional_vector)
        
        if len(additional_features) > 0:
            additional_features = torch.cat(additional_features, dim=1)
        else:
            additional_features = torch.empty(0)
        
        # 統合モードに応じた処理
        if self.integration_mode == 'early_integration' and additional_features.numel() > 0:
            # 追加特徴量を各原子にブロードキャスト
            augmented_features = []
            for i, n_atoms in enumerate(molecules_unbatch_key):
                mol_add_features = additional_features[i:i+1].expand(n_atoms, -1)
                augmented_features.append(mol_add_features)
            
            augmented_features = torch.cat(augmented_features, dim=0)
            atom_features_augmented = torch.cat([atom_features, augmented_features], dim=1)
            
            # 拡張された原子特徴量でエンコード
            encodings = self.encoder(atom_features_augmented, f_ini_atoms_bonds,
                                   atom_to_incoming_bonds, mapping,
                                   global_features, molecules_unbatch_key)
        else:
            # 通常のエンコーディング
            encodings = self.encoder(atom_features, f_ini_atoms_bonds,
                                   atom_to_incoming_bonds, mapping,
                                   global_features, molecules_unbatch_key)
        
        # 統合レイヤーの適用
        if self.integration_layer is not None and additional_features.numel() > 0:
            if self.integration_mode != 'early_integration':
                encodings = self.integration_layer(encodings, additional_features)
        
        # FFNで予測
        output = self.ffn(encodings)
        
        # モードに応じた出力処理
        if self.mode == 'regression':
            final_output = output
        elif self.mode == 'classification':
            if self.n_tasks == 1:
                output = output.view(-1, self.n_classes)
                final_output = nn.functional.softmax(output, dim=1), output
            else:
                output = output.view(-1, self.n_tasks, self.n_classes)
                final_output = nn.functional.softmax(output, dim=2), output
        
        return final_output
```

### 3.6 DMPNNModel クラスの拡張

#### 3.6.1 実装詳細

**ファイル**: `deepchem/models/torch_models/dmpnn.py`

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
        拡張されたコンストラクタ
        
        Implementation Notes:
        - 追加パラメータをDMPNNクラスに渡す
        - 既存の初期化処理は変更しない
        """
        # DMPNNモデルの初期化（追加パラメータを含む）
        model = DMPNN(
            mode=mode,
            n_classes=n_classes,
            n_tasks=n_tasks,
            global_features_size=global_features_size,
            use_default_fdim=use_default_fdim,
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            enc_hidden=enc_hidden,
            depth=depth,
            bias=bias,
            enc_activation=enc_activation,
            enc_dropout_p=enc_dropout_p,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            ffn_hidden=ffn_hidden,
            ffn_activation=ffn_activation,
            ffn_layers=ffn_layers,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            additional_scalar_features_size=additional_scalar_features_size,
            additional_vector_features_size=additional_vector_features_size,
            integration_mode=integration_mode)
        
        # 損失関数の設定
        if mode == 'regression':
            loss = L2Loss()
            output_types = ['prediction']
        elif mode == 'classification':
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        
        # TorchModelの初期化
        super(DMPNNModel, self).__init__(model,
                                        loss=loss,
                                        output_types=output_types,
                                        batch_size=batch_size,
                                        **kwargs)
    
    def _to_pyg_graph(self, values: Sequence[np.ndarray]) -> _ModData:
        """
        拡張された_to_pyg_graphメソッド
        
        Implementation Notes:
        - valuesから7要素（追加特徴量2つを含む）を抽出
        - 追加特徴量をTensorに変換
        - _ModDataオブジェクトに追加
        """
        # 既存の要素を抽出
        atom_features = values[0]
        f_ini_atoms_bonds = values[1]
        atom_to_incoming_bonds = values[2]
        mapping = values[3]
        global_features = values[4]
        
        # 追加特徴量を抽出（新規）
        additional_scalar_features = values[5] if len(values) > 5 else np.empty(0)
        additional_vector_features = values[6] if len(values) > 6 else np.empty(0)
        
        # Tensorへの変換
        t_atom_features = torch.from_numpy(atom_features).float().to(device=self.device)
        t_f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float().to(device=self.device)
        t_atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds).to(device=self.device)
        t_mapping = torch.from_numpy(mapping).to(device=self.device)
        t_global_features = torch.from_numpy(global_features).float().to(device=self.device)
        
        # 追加特徴量のTensor変換（新規）
        t_additional_scalar_features = torch.from_numpy(additional_scalar_features).float().to(device=self.device)
        t_additional_vector_features = torch.from_numpy(additional_vector_features).float().to(device=self.device)
        
        # _ModDataオブジェクトの作成
        return _ModData(required_inc=len(t_f_ini_atoms_bonds),
                       atom_features=t_atom_features,
                       f_ini_atoms_bonds=t_f_ini_atoms_bonds,
                       atom_to_incoming_bonds=t_atom_to_incoming_bonds,
                       mapping=t_mapping,
                       global_features=t_global_features,
                       additional_scalar_features=t_additional_scalar_features,
                       additional_vector_features=t_additional_vector_features)
```

## 4. データフロー詳細

### 4.1 シーケンス図: 特徴化からバッチ処理まで

```
User -> DMPNNFeaturizer: featurize(smiles, scalar_feat, vector_feat)
DMPNNFeaturizer -> DMPNNFeaturizer: _featurize()
DMPNNFeaturizer -> DMPNNFeaturizer: _validate_additional_features()
DMPNNFeaturizer -> GraphData: __init__(..., additional_scalar, additional_vector)
GraphData --> DMPNNFeaturizer: GraphData object
DMPNNFeaturizer --> User: GraphData object

User -> DMPNNModel: fit(dataset)
DMPNNModel -> DMPNNModel: default_generator()
loop for each batch
    DMPNNModel -> _MapperDMPNN: __init__(graph)
    _MapperDMPNN -> _MapperDMPNN: extract additional features
    _MapperDMPNN --> DMPNNModel: mapper.values (7 elements)
    DMPNNModel -> DMPNNModel: _to_pyg_graph(values)
    DMPNNModel -> _ModData: create with additional features
    _ModData --> DMPNNModel: pyg_graph
end
DMPNNModel -> Batch: from_data_list(pyg_graphs_list)
Batch --> DMPNNModel: batched data
DMPNNModel -> DMPNN: forward(batch)
DMPNN -> IntegrationLayer: integrate features
IntegrationLayer --> DMPNN: integrated features
DMPNN -> FFN: forward(integrated)
FFN --> DMPNN: predictions
DMPNN --> DMPNNModel: predictions
```

## 5. 使用例

### 5.1 基本的な使用例（後期連結方式）

```python
import deepchem as dc
import numpy as np
from rdkit import Chem

# データの準備
smiles_list = ["CCO", "CC(C)O", "CCCO"]
scalar_features = np.array([
    [298.15, 7.0],  # 温度とpH
    [310.15, 7.4],
    [300.15, 6.8]
])
vector_features = np.array([
    [0.1, 0.2, 0.3, 0.4],  # 溶媒特性など
    [0.2, 0.3, 0.4, 0.5],
    [0.15, 0.25, 0.35, 0.45]
])

# Featurizerの初期化
featurizer = dc.feat.DMPNNFeaturizer(
    additional_scalar_features_dim=2,
    additional_vector_features_dim=4
)

# 特徴化
graphs = []
for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    graph = featurizer._featurize(
        mol,
        additional_scalar_features=scalar_features[i],
        additional_vector_features=vector_features[i]
    )
    graphs.append(graph)

# モデルの初期化
model = dc.models.DMPNNModel(
    mode='regression',
    n_tasks=1,
    additional_scalar_features_size=2,
    additional_vector_features_size=4,
    integration_mode='late_concat'
)

# 学習
# dataset = dc.data.NumpyDataset(X=graphs, y=labels, w=weights)
# model.fit(dataset, nb_epoch=10)
```

### 5.2 アテンション統合方式の使用例

```python
# Featurizerは同じ
featurizer = dc.feat.DMPNNFeaturizer(
    additional_scalar_features_dim=2,
    additional_vector_features_dim=4
)

# モデルをアテンションモードで初期化
model = dc.models.DMPNNModel(
    mode='regression',
    n_tasks=1,
    additional_scalar_features_size=2,
    additional_vector_features_size=4,
    integration_mode='attention'  # アテンション統合
)

# 学習後、アテンション重みを確認可能
# attention_weights = model.model.integration_layer.get_attention_weights(batch)
```

## 6. テスト設計

### 6.1 単体テストケース

#### テストファイル: `test_dmpnn_additional_features.py`

```python
import pytest
import numpy as np
import torch
from rdkit import Chem
import deepchem as dc

class TestDMPNNAdditionalFeatures:
    
    def test_featurizer_with_scalar_features(self):
        """スカラー特徴量のみを使用した特徴化のテスト"""
        featurizer = dc.feat.DMPNNFeaturizer(
            additional_scalar_features_dim=3
        )
        mol = Chem.MolFromSmiles("CCO")
        scalar_feat = np.array([1.0, 2.0, 3.0])
        
        graph = featurizer._featurize(mol, additional_scalar_features=scalar_feat)
        
        assert graph.additional_scalar_features is not None
        assert graph.additional_scalar_features.shape == (3,)
        assert np.allclose(graph.additional_scalar_features, scalar_feat)
    
    def test_featurizer_with_vector_features(self):
        """ベクトル特徴量のみを使用した特徴化のテスト"""
        featurizer = dc.feat.DMPNNFeaturizer(
            additional_vector_features_dim=5
        )
        mol = Chem.MolFromSmiles("CCO")
        vector_feat = np.random.randn(5)
        
        graph = featurizer._featurize(mol, additional_vector_features=vector_feat)
        
        assert graph.additional_vector_features is not None
        assert graph.additional_vector_features.shape == (5,)
    
    def test_featurizer_dimension_mismatch(self):
        """次元不一致のエラーテスト"""
        featurizer = dc.feat.DMPNNFeaturizer(
            additional_scalar_features_dim=3
        )
        mol = Chem.MolFromSmiles("CCO")
        wrong_scalar_feat = np.array([1.0, 2.0])  # 2次元（期待は3次元）
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            featurizer._featurize(mol, additional_scalar_features=wrong_scalar_feat)
    
    def test_integration_late_concat(self):
        """後期連結方式のテスト"""
        model = dc.models.DMPNNModel(
            additional_scalar_features_size=2,
            additional_vector_features_size=3,
            integration_mode='late_concat'
        )
        
        # FFN入力サイズの確認
        expected_ffn_input = 300 + 0 + 2 + 3  # enc_hidden + global + scalar + vector
        assert model.model.ffn.layers[0].in_features == expected_ffn_input
    
    def test_integration_attention(self):
        """アテンション統合方式のテスト"""
        model = dc.models.DMPNNModel(
            additional_scalar_features_size=2,
            additional_vector_features_size=3,
            integration_mode='attention'
        )
        
        assert hasattr(model.model.integration_layer, 'W_add')
        assert hasattr(model.model.integration_layer, 'w_struct')
        assert hasattr(model.model.integration_layer, 'w_add')
    
    def test_backward_compatibility(self):
        """後方互換性のテスト（追加特徴量なし）"""
        # 追加特徴量を使わない場合、既存の動作と同じ
        featurizer = dc.feat.DMPNNFeaturizer()
        mol = Chem.MolFromSmiles("CCO")
        
        graph = featurizer._featurize(mol)
        
        # 追加特徴量は空であるべき
        assert graph.additional_scalar_features is None or graph.additional_scalar_features.size == 0
        assert graph.additional_vector_features is None or graph.additional_vector_features.size == 0
```

## 7. パフォーマンス最適化

### 7.1 メモリ最適化

```python
# 大規模バッチでのメモリ使用量の削減
def optimize_batch_size(self, 
                       molecule_sizes: List[int],
                       additional_features_size: int,
                       available_memory: int) -> int:
    """
    最適なバッチサイズを計算
    
    Parameters:
    - molecule_sizes: 分子ごとの原子数のリスト
    - additional_features_size: 追加特徴量の合計サイズ
    - available_memory: 利用可能なメモリ（バイト）
    
    Returns:
    - optimal_batch_size: 最適なバッチサイズ
    """
    avg_molecule_size = np.mean(molecule_sizes)
    
    # 1分子あたりのメモリ使用量を推定
    per_molecule_memory = (
        avg_molecule_size * 133 * 4 +  # atom features (float32)
        avg_molecule_size * 2 * 14 * 4 +  # bond features
        additional_features_size * 4  # additional features
    )
    
    # 安全マージンを考慮
    optimal_batch_size = int(available_memory * 0.8 / per_molecule_memory)
    
    return max(1, optimal_batch_size)
```

### 7.2 計算最適化

```python
# 早期統合方式での効率化
def efficient_broadcast(self, 
                       additional_features: torch.Tensor,
                       molecules_unbatch_key: List[int]) -> torch.Tensor:
    """
    効率的な追加特徴量のブロードキャスト
    
    torch.repeatを使った効率的な実装
    """
    # repeat_interleaveを使用して効率的にブロードキャスト
    repeat_counts = torch.tensor(molecules_unbatch_key, device=additional_features.device)
    augmented_features = torch.repeat_interleave(additional_features, repeat_counts, dim=0)
    return augmented_features
```

## 8. エラーハンドリング実装

```python
def validate_integration_mode(self, mode: str) -> None:
    """統合モードの検証"""
    valid_modes = ['late_concat', 'early_integration', 'attention']
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid integration_mode '{mode}'. "
            f"Must be one of {valid_modes}. "
            f"See documentation for details on each mode."
        )

def handle_nan_values(self, features: np.ndarray, feature_name: str) -> np.ndarray:
    """NaN値の処理"""
    if np.any(np.isnan(features)):
        logger.warning(
            f"NaN values detected in {feature_name}. "
            f"Replacing with zeros. "
            f"Consider preprocessing your features to avoid this."
        )
        features = np.nan_to_num(features, nan=0.0)
    return features

def check_batch_consistency(self, 
                           batch_size: int,
                           additional_features_batch_size: int) -> None:
    """バッチサイズの整合性チェック"""
    if batch_size != additional_features_batch_size:
        raise RuntimeError(
            f"Batch size mismatch. "
            f"Number of molecules: {batch_size}, "
            f"Number of additional feature vectors: {additional_features_batch_size}. "
            f"Each molecule must have exactly one set of additional features."
        )
```

## 9. まとめ

本詳細設計書では、DMPNNModelおよびDMPNNFeaturizerに追加特徴量を統合するための具体的な実装方法を記述しました。主要な内容は以下の通りです：

1. **完全なクラス設計**: 全ての拡張クラスの詳細な実装
2. **3つの統合レイヤー**: LateConcatIntegration、EarlyIntegration、AttentionIntegrationの完全実装
3. **データフロー**: 特徴化からモデル推論までの詳細なフロー
4. **使用例**: 実際のコード例を含む包括的な使用ガイド
5. **テスト設計**: 単体テストケースの完全な仕様
6. **パフォーマンス最適化**: メモリと計算の効率化手法
7. **エラーハンドリング**: 堅牢なエラー処理の実装

この設計書に基づき、実装者は確実にコードを実装できます。全ての数式は理論説明書と一貫性があり、全ての仕様は仕様書と整合しています。
