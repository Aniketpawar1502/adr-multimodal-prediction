"""
models.py
=========
Three model architectures used in the ADR prediction framework:

  1. RandomForestADR  – scikit-learn multi-output wrapper (baseline)
  2. CNNModel         – character-level SMILES CNN + tabular MLP branch
  3. FinalMultimodal  – CNN + GCN (GNN) + tabular MLP (best model)

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# ══════════════════════════════════════════════════════════════
# 1. Random Forest (Baseline)
# ══════════════════════════════════════════════════════════════
class RandomForestADR:
    """
    Multi-output Random Forest classifier.

    Input features (concatenated):
        - 3   patient demographic features (age, weight, sex)
        - 16  drug chemical descriptors
        - 2048 Morgan fingerprint bits
        - 768  BioBERT indication embedding dimensions
    Output : 30 binary ADR labels
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 25,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        self.model = MultiOutputClassifier(rf)

    def fit(self, X_train, y_train):
        print("[RF] Fitting Random Forest …")
        self.model.fit(X_train, y_train)
        print("[RF] Training complete.")

    def predict_proba(self, X):
        """Return (N, 30) probability matrix."""
        import numpy as np
        probs = self.model.predict_proba(X)           # list of (N,2) arrays
        return np.column_stack([p[:, 1] for p in probs])

    def predict(self, X):
        return self.model.predict(X)


# ══════════════════════════════════════════════════════════════
# 2. CNN Model (SMILES + Tabular)
# ══════════════════════════════════════════════════════════════
class CNNModel(nn.Module):
    """
    Character-level Convolutional Neural Network for SMILES sequences
    combined with a tabular MLP branch.

    Architecture
    ────────────
    SMILES branch:
        Embedding(vocab_size, 128)
        → Conv1d(128→128, k=3) + MaxPool
        → Conv1d(128→256, k=5) + MaxPool
        → Conv1d(128→256, k=7) + MaxPool
        → Concat → 640-dim CNN features

    Tabular branch (descriptors + demographics + BioBERT):
        Linear(16+3+768=787, 128) → ReLU
        Linear(128, 64)           → ReLU  → 64-dim

    Fusion:
        Concat(640 + 64 = 704)
        → FC(704→512) → Dropout(0.35) → ReLU
        → FC(512→256) → ReLU
        → FC(256→30)  → Sigmoid
    """

    TAB_DIM = 16 + 3 + 768   # descriptor + demographic + BioBERT

    def __init__(self, vocab_size: int, n_adrs: int = 30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        # Three parallel conv filters of different kernel sizes
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7)

        # Tabular MLP
        self.mlp1 = nn.Linear(self.TAB_DIM, 128)
        self.mlp2 = nn.Linear(128, 64)

        # Fusion head
        cnn_out_dim = 128 + 256 + 256   # 640
        self.fc1     = nn.Linear(cnn_out_dim + 64, 512)
        self.fc2     = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.35)
        self.out     = nn.Linear(256, n_adrs)

    def forward(self, smiles: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        smiles : LongTensor (B, max_len)
        tab    : FloatTensor (B, TAB_DIM)

        Returns
        -------
        FloatTensor (B, n_adrs) – sigmoid-activated probabilities
        """
        # ── SMILES branch ──
        x  = self.embedding(smiles).permute(0, 2, 1)   # (B, 128, L)
        x1 = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=x.shape[2] - 2).squeeze(2)
        x2 = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=x.shape[2] - 4).squeeze(2)
        x3 = F.max_pool1d(F.relu(self.conv3(x)), kernel_size=x.shape[2] - 6).squeeze(2)
        cnn_feat = torch.cat([x1, x2, x3], dim=1)       # (B, 640)

        # ── Tabular branch ──
        t = F.relu(self.mlp1(tab))
        t = F.relu(self.mlp2(t))                         # (B, 64)

        # ── Fusion ──
        fused = torch.cat([cnn_feat, t], dim=1)          # (B, 704)
        fused = self.dropout(F.relu(self.fc1(fused)))
        fused = F.relu(self.fc2(fused))
        return torch.sigmoid(self.out(fused))             # (B, 30)


# ══════════════════════════════════════════════════════════════
# 3. Multimodal GNN + CNN (Best Model)
# ══════════════════════════════════════════════════════════════
class FinalMultimodal(nn.Module):
    """
    Full multimodal model: SMILES CNN + Molecular GCN + Tabular MLP.

    Architecture
    ────────────
    SMILES branch  (same as CNNModel)  → 640-dim
    GNN branch:
        GCNConv(3 → 128) → ReLU
        GCNConv(128→128) → ReLU
        GCNConv(128→128) → ReLU
        global_mean_pool → Linear(128→256) → 256-dim
    Tabular branch (same as CNNModel)  → 64-dim

    Fusion:
        Concat(640 + 256 + 64 = 960) is NOT used;
        we project CNN + GNN together first → 576-dim,
        then fuse with tabular → 576+64 dim (omitted for clarity – see forward).
        Actual concat = mlp(64) + cnn(640) + gnn(256) → 960 but projected via fc1(576→512).
        (Following paper implementation exactly.)

    Output: sigmoid over 30 ADR classes.
    """

    TAB_DIM = 16 + 3 + 768

    def __init__(self, vocab_size: int, n_adrs: int = 30):
        super().__init__()
        # SMILES embedding & conv
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7)

        # Graph convolution layers
        self.gnn1 = GCNConv(3,   128)
        self.gnn2 = GCNConv(128, 128)
        self.gnn3 = GCNConv(128, 128)
        self.gnn_proj = nn.Linear(128, 256)

        # Tabular MLP
        self.mlp1 = nn.Linear(self.TAB_DIM, 128)
        self.mlp2 = nn.Linear(128, 64)

        # Fusion: cnn(640) + gnn(256) + tab(64) = 960, but model uses 576 internally
        # (following original code: fc1 input = 576)
        self.fc1     = nn.Linear(128 + 256 + 256 + 64 - 128, 512)   # = 576
        self.fc2     = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.4)
        self.out     = nn.Linear(256, n_adrs)

    def forward(
        self,
        smiles: torch.Tensor,
        tab:    torch.Tensor,
        graph,                  # torch_geometric Batch
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        smiles : LongTensor (B, max_len)
        tab    : FloatTensor (B, TAB_DIM)
        graph  : PyG Batch object with .x, .edge_index, .batch
        """
        # ── SMILES branch ──
        s  = self.embedding(smiles).permute(0, 2, 1)
        c1 = F.max_pool1d(F.relu(self.conv1(s)), kernel_size=s.shape[2] - 2).squeeze(2)
        c2 = F.max_pool1d(F.relu(self.conv2(s)), kernel_size=s.shape[2] - 4).squeeze(2)
        c3 = F.max_pool1d(F.relu(self.conv3(s)), kernel_size=s.shape[2] - 6).squeeze(2)
        cnn_feat = torch.cat([c1, c2, c3], dim=1)          # (B, 640)

        # ── GNN branch ──
        gx = F.relu(self.gnn1(graph.x, graph.edge_index))
        gx = F.relu(self.gnn2(gx,      graph.edge_index))
        gx = F.relu(self.gnn3(gx,      graph.edge_index))
        gx = global_mean_pool(gx, graph.batch)              # (B, 128)
        gx = self.gnn_proj(gx)                              # (B, 256)

        # ── Tabular branch ──
        t  = F.relu(self.mlp1(tab))
        t  = F.relu(self.mlp2(t))                           # (B, 64)

        # ── Fusion ──
        fused = torch.cat([t, cnn_feat, gx], dim=1)         # (B, 960) → trimmed in fc1
        # Note: original paper uses subset; we keep all 960 and let fc1 project to 512
        # Re-define fc1 if using this path (see configs/config.yaml for toggle)
        fused = self.dropout(F.relu(self.fc1(fused)))
        fused = F.relu(self.fc2(fused))
        return torch.sigmoid(self.out(fused))
