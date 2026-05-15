"""
predict.py
==========
Run inference on new drug–patient pairs using a saved model.

Usage
-----
    python src/predict.py \
        --model gnn \
        --model_path results/gnn_model.pt \
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \
        --age 55 --weight 70 --sex 1 \
        --indication "Rheumatoid arthritis"

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import argparse
import pickle

import numpy as np
import torch

from models import CNNModel, FinalMultimodal
from preprocessing import (
    morgan_fp, encode_indications, tokenize_smiles,
    smiles_to_graph, DESCRIPTOR_COLS, MORGAN_NBITS
)
from evaluate import ADR_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.5


def predict_single(args):
    """Run inference for a single SMILES + patient record."""

    smiles      = args.smiles
    age         = float(args.age)
    weight      = float(args.weight)
    sex         = float(args.sex)     # 0 = Female, 1 = Male
    indication  = args.indication

    # ── Feature engineering ──
    fp          = morgan_fp(smiles).reshape(1, -1)
    bio_embed   = encode_indications([indication])       # (1, 768)

    # Dummy descriptor values (replace with PubChem API call in production)
    descriptors = np.zeros((1, len(DESCRIPTOR_COLS)))

    tab_input = np.concatenate(
        [descriptors, np.array([[age, weight, sex]]), bio_embed], axis=1
    )   # shape (1, 16+3+768)

    if args.model in ("cnn", "gnn"):
        smiles_tensor, vocab_size = tokenize_smiles([smiles])

    if args.model == "rf":
        rf_input = np.concatenate(
            [np.array([[age, weight, sex]]), descriptors, fp, bio_embed], axis=1
        )
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)
        proba = model.predict_proba(rf_input)[0]

    elif args.model == "cnn":
        from preprocessing import tokenize_smiles
        s_tensor, vocab_size = tokenize_smiles([smiles])
        net = CNNModel(vocab_size=vocab_size).to(DEVICE)
        net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        net.eval()
        with torch.no_grad():
            proba = net(
                s_tensor.to(DEVICE),
                torch.tensor(tab_input, dtype=torch.float32).to(DEVICE),
            ).cpu().numpy()[0]

    elif args.model == "gnn":
        from torch_geometric.data import Batch
        s_tensor, vocab_size = tokenize_smiles([smiles])
        graph = smiles_to_graph(smiles)
        net   = FinalMultimodal(vocab_size=vocab_size).to(DEVICE)
        net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        net.eval()
        with torch.no_grad():
            g_batch = Batch.from_data_list([graph]).to(DEVICE)
            proba   = net(
                s_tensor.to(DEVICE),
                torch.tensor(tab_input, dtype=torch.float32).to(DEVICE),
                g_batch,
            ).cpu().numpy()[0]

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ── Display results ──
    print(f"\nPredicted ADRs (threshold = {THRESHOLD}):\n")
    print(f"{'ADR':<25} {'Probability':>12}")
    print("-" * 40)
    flagged = []
    for name, prob in zip(ADR_NAMES, proba):
        flag = "  ← PREDICTED" if prob >= THRESHOLD else ""
        if prob >= THRESHOLD:
            flagged.append(name)
        print(f"{name:<25} {prob:>12.4f}{flag}")

    print(f"\nTotal predicted ADRs: {len(flagged)}")
    return proba


def parse_args():
    parser = argparse.ArgumentParser(description="ADR Prediction Inference")
    parser.add_argument("--model",      type=str, required=True, choices=["rf", "cnn", "gnn"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--smiles",     type=str, required=True, help="SMILES string of the drug")
    parser.add_argument("--age",        type=float, default=50)
    parser.add_argument("--weight",     type=float, default=70)
    parser.add_argument("--sex",        type=float, default=1, help="0=Female, 1=Male")
    parser.add_argument("--indication", type=str, default="Pain management")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_single(args)
