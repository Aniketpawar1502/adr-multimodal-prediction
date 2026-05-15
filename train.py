"""
train.py
========
Training routines for all three ADR prediction models.

Usage
-----
    python src/train.py --model rf   --data data/processed/faers_final.csv
    python src/train.py --model cnn  --data data/processed/faers_final.csv
    python src/train.py --model gnn  --data data/processed/faers_final.csv

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch

from models import RandomForestADR, CNNModel, FinalMultimodal
from preprocessing import build_features
from evaluate import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Random Forest Training
# ─────────────────────────────────────────────
def train_rf(features: dict, save_path: str = "results/rf_model.pkl"):
    model = RandomForestADR()
    model.fit(features["rf_train"], features["y_train"])

    proba = model.predict_proba(features["rf_test"])
    metrics = compute_metrics(features["y_test"], proba)
    print("\n[RF] Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[RF] Model saved → {save_path}")
    return metrics


# ─────────────────────────────────────────────
# CNN Training
# ─────────────────────────────────────────────
def train_cnn(
    features: dict,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: str = "results/cnn_model.pt",
):
    smiles_train = features["smiles_train"]
    tab_train    = features["tab_train"]
    y_train      = torch.tensor(features["y_train"], dtype=torch.float32)

    smiles_test  = features["smiles_test"].to(DEVICE)
    tab_test     = features["tab_test"].to(DEVICE)

    dataset  = TensorDataset(smiles_train, tab_train, y_train)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model    = CNNModel(vocab_size=features["vocab_size"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for smiles_b, tab_b, y_b in loader:
            smiles_b, tab_b, y_b = smiles_b.to(DEVICE), tab_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds = model(smiles_b, tab_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                proba = model(smiles_test, tab_test).cpu().numpy()
            metrics = compute_metrics(features["y_test"], proba)
            print(
                f"[CNN] Epoch {epoch:3d}/{epochs} | "
                f"Loss {epoch_loss/len(loader):.4f} | "
                f"Macro-AUC {metrics['macro_auc']:.4f} | "
                f"MAP {metrics['map']:.4f}"
            )
            if metrics["macro_auc"] > best_auc:
                best_auc = metrics["macro_auc"]
                torch.save(model.state_dict(), save_path)
                print(f"[CNN] New best model saved → {save_path}")

    print(f"\n[CNN] Best Macro-AUC: {best_auc:.4f}")
    return best_auc


# ─────────────────────────────────────────────
# GNN + CNN Training
# ─────────────────────────────────────────────
def train_gnn(
    features: dict,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 5e-4,
    save_path: str = "results/gnn_model.pt",
):
    model     = FinalMultimodal(vocab_size=features["vocab_size"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    smiles_train = features["smiles_train"]
    tab_train    = features["tab_train"]
    graphs_train = features["graphs_train"]
    y_train      = torch.tensor(features["y_train"], dtype=torch.float32)

    smiles_test  = features["smiles_test"].to(DEVICE)
    tab_test     = features["tab_test"].to(DEVICE)

    N = len(smiles_train)
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        indices = torch.randperm(N)

        for start in range(0, N, batch_size):
            idx_b = indices[start: start + batch_size]
            s_b   = smiles_train[idx_b].to(DEVICE)
            t_b   = tab_train[idx_b].to(DEVICE)
            y_b   = y_train[idx_b].to(DEVICE)
            g_b   = Batch.from_data_list([graphs_train[i] for i in idx_b]).to(DEVICE)

            optimizer.zero_grad()
            preds = model(s_b, t_b, g_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                g_test = Batch.from_data_list(features["graphs_test"]).to(DEVICE)
                proba  = model(smiles_test, tab_test, g_test).cpu().numpy()
            metrics = compute_metrics(features["y_test"], proba)
            print(
                f"[GNN] Epoch {epoch:3d}/{epochs} | "
                f"Loss {epoch_loss:.4f} | "
                f"Macro-AUC {metrics['macro_auc']:.4f} | "
                f"MAP {metrics['map']:.4f}"
            )
            if metrics["macro_auc"] > best_auc:
                best_auc = metrics["macro_auc"]
                torch.save(model.state_dict(), save_path)
                print(f"[GNN] New best model saved → {save_path}")

    print(f"\n[GNN] Best Macro-AUC: {best_auc:.4f}")
    return best_auc


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train ADR prediction models")
    parser.add_argument("--model",      type=str, required=True, choices=["rf", "cnn", "gnn"])
    parser.add_argument("--data",       type=str, required=True, help="Path to processed CSV")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    features = build_features(args.data)

    if args.model == "rf":
        train_rf(features)
    elif args.model == "cnn":
        train_cnn(features, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.model == "gnn":
        train_gnn(features, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
