"""
preprocessing.py
================
Data loading, cleaning, and feature engineering pipeline for the
Multi-Modal ADR Prediction project.

Steps covered:
  1. Load merged FAERS CSV (DEMO + DRUG + REAC + INDI)
  2. Encode categorical columns
  3. Compute Morgan fingerprints via RDKit
  4. Encode drug indications with BioBERT
  5. Return train/test splits ready for all three models

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTOR_COLS = [
    "MolecularWeight", "ExactMass", "MonoisotopicMass",
    "TopologicalPolarSurfaceArea", "HydrogenBondDonorCount",
    "HydrogenBondAcceptorCount", "RotatableBondCount",
    "Covalently-BondedUnitCount", "DefinedAtomStereocenterCount",
    "UndefinedAtomStereocenterCount", "DefinedBondStereocenterCount",
    "UndefinedBondStereocenterCount", "Complexity", "FormalCharge",
    "HeavyAtomCount", "IsotopeAtomCount",
]

TAB_COLS = ["age", "weight", "sex"]

ADR_COLS = [f"ADR_{i}" for i in range(30)]

BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
MORGAN_NBITS  = 2048
MORGAN_RADIUS = 2
SMILES_MAX_LEN = 120


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the pre-merged FAERS CSV file."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows × {df.shape[1]} columns from {csv_path}")
    return df


# ─────────────────────────────────────────────
# 2. Basic Pre-processing
# ─────────────────────────────────────────────
def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode sex column and drop rows with missing mandatory fields."""
    df = df.copy()
    df["sex"] = LabelEncoder().fit_transform(df["sex"].astype(str))
    mandatory = TAB_COLS + ["smiles", "indi_pt"] + ADR_COLS
    before = len(df)
    df = df.dropna(subset=mandatory).reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df):,} rows with missing values; {len(df):,} remain.")
    return df


# ─────────────────────────────────────────────
# 3. Morgan Fingerprints
# ─────────────────────────────────────────────
def morgan_fp(smiles: str, n_bits: int = MORGAN_NBITS, radius: int = MORGAN_RADIUS) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint bit-vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """Compute Morgan fingerprints for all SMILES in the dataframe."""
    print("[INFO] Computing Morgan fingerprints …")
    fps = np.array([morgan_fp(s) for s in df["smiles"]])
    print(f"[INFO] Fingerprint matrix shape: {fps.shape}")
    return fps


# ─────────────────────────────────────────────
# 4. BioBERT Indication Embeddings
# ─────────────────────────────────────────────
def encode_indications(texts: list, batch_size: int = 64) -> np.ndarray:
    """
    Encode drug indication text (indi_pt) using BioBERT CLS token.
    Returns a (N, 768) float32 array.
    """
    print("[INFO] Loading BioBERT tokenizer & model …")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    model     = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        with torch.no_grad():
            tokens = tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(DEVICE)
            out = model(**tokens)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls)

    embeddings = np.vstack(all_embeddings)
    print(f"[INFO] BioBERT embeddings shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# 5. SMILES Tokenisation (for CNN)
# ─────────────────────────────────────────────
def tokenize_smiles(smiles_list: list, max_len: int = SMILES_MAX_LEN):
    """
    Character-level tokenisation of SMILES strings.
    Returns (LongTensor of shape N×max_len, vocab_size).
    """
    all_chars = sorted(set("".join(smiles_list)))
    vocab     = {c: i + 1 for i, c in enumerate(all_chars)}

    seqs = []
    for s in smiles_list:
        seq  = [vocab.get(ch, 0) for ch in s[:max_len]]
        seq += [0] * (max_len - len(seq))
        seqs.append(seq)

    tensor = torch.tensor(seqs, dtype=torch.long)
    return tensor, len(vocab) + 1  # +1 for padding index


# ─────────────────────────────────────────────
# 6. Molecular Graph (for GNN)
# ─────────────────────────────────────────────
def smiles_to_graph(smiles: str):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Node features: [atomic_num, degree, is_aromatic] (3-dim).
    """
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge indices (undirected)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────
# 7. Master Build Function
# ─────────────────────────────────────────────
def build_features(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: load → preprocess → fingerprints → BioBERT → split.

    Returns
    -------
    dict with keys:
        rf_train, rf_test           – np arrays for Random Forest
        smiles_train, smiles_test   – LongTensors for CNN
        tab_train, tab_test         – FloatTensors for CNN/GNN (tabular + descriptors + BioBERT)
        graphs_train, graphs_test   – lists of PyG Data objects
        y_train, y_test             – np arrays of ADR labels
        vocab_size                  – int
    """
    df  = load_data(csv_path)
    df  = basic_preprocess(df)

    fps         = compute_fingerprints(df)
    bio_embeds  = encode_indications(df["indi_pt"].tolist())

    # ── Random Forest input: tab + descriptors + fingerprints + BioBERT ──
    rf_input = np.concatenate([
        df[TAB_COLS].values,
        df[DESCRIPTOR_COLS].values,
        fps,
        bio_embeds,
    ], axis=1)

    # ── CNN tabular branch: descriptors + tab + BioBERT ──
    tab_input = np.concatenate([
        df[DESCRIPTOR_COLS].values,
        df[TAB_COLS].values,
        bio_embeds,
    ], axis=1)

    # ── SMILES tokenisation ──
    smiles_tensor, vocab_size = tokenize_smiles(df["smiles"].tolist())

    # ── Molecular graphs ──
    print("[INFO] Building molecular graphs …")
    graphs = [smiles_to_graph(s) for s in df["smiles"]]

    # ── Labels ──
    y = df[ADR_COLS].values.astype(np.float32)

    # ── Train / test split (shared indices) ──
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    rf_train = scaler.fit_transform(rf_input[idx_train])
    rf_test  = scaler.transform(rf_input[idx_test])

    tab_scaler = StandardScaler()
    tab_train = torch.tensor(tab_scaler.fit_transform(tab_input[idx_train]), dtype=torch.float32)
    tab_test  = torch.tensor(tab_scaler.transform(tab_input[idx_test]),  dtype=torch.float32)

    return {
        "rf_train":      rf_train,
        "rf_test":       rf_test,
        "smiles_train":  smiles_tensor[idx_train],
        "smiles_test":   smiles_tensor[idx_test],
        "tab_train":     tab_train,
        "tab_test":      tab_test,
        "graphs_train":  [graphs[i] for i in idx_train],
        "graphs_test":   [graphs[i] for i in idx_test],
        "y_train":       y[idx_train],
        "y_test":        y[idx_test],
        "vocab_size":    vocab_size,
    }
