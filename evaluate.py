"""
evaluate.py
===========
Evaluation metrics for multi-label ADR prediction.

Metrics computed:
  - Macro AUC (per-label averaged ROC-AUC)
  - Mean Average Precision (MAP)
  - Hamming Loss
  - Micro F1 score

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    f1_score,
)

ADR_NAMES = [
    "Nausea", "Fatigue", "Diarrhoea", "Dyspnoea", "Headache",
    "Dizziness", "Vomiting", "Abdominal pain", "Malaise", "Alopecia",
    "Pneumonia", "Pruritus", "Anxiety", "Insomnia", "Weight decreased",
    "Anaemia", "Arthralgia", "Pain in extremity", "Cough", "Fall",
    "Acute kidney injury", "Hypertension", "Weight increased", "Depression",
    "Back pain", "Constipation", "Hypotension", "Chest Pain",
    "Gait disturbance", "Somnolence",
]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true    : (N, 30) binary ground-truth labels
    y_prob    : (N, 30) predicted probabilities
    threshold : binarisation threshold (default 0.5)

    Returns
    -------
    dict with keys: macro_auc, map, hamming_loss, micro_f1
    """
    y_pred = (y_prob >= threshold).astype(int)

    macro_auc = roc_auc_score(y_true, y_prob, average="macro")
    map_score = average_precision_score(y_true, y_prob, average="macro")
    h_loss    = hamming_loss(y_true, y_pred)
    micro_f1  = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return {
        "macro_auc":    macro_auc,
        "map":          map_score,
        "hamming_loss": h_loss,
        "micro_f1":     micro_f1,
    }


def per_adr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute per-ADR AUC and return as a sorted DataFrame.

    Returns
    -------
    DataFrame with columns: ADR, AUC, Count
    """
    records = []
    for i, name in enumerate(ADR_NAMES):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        count = int(y_true[:, i].sum())
        records.append({"ADR": name, "AUC": auc, "Count": count})

    df = pd.DataFrame(records).sort_values("Count", ascending=False).reset_index(drop=True)
    return df


def print_full_report(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model"):
    """Print a formatted evaluation report."""
    metrics = compute_metrics(y_true, y_prob)
    per_adr = per_adr_auc(y_true, y_prob)

    print(f"\n{'='*55}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*55}")
    print(f"  Macro AUC    : {metrics['macro_auc']:.4f}")
    print(f"  MAP          : {metrics['map']:.4f}")
    print(f"  Hamming Loss : {metrics['hamming_loss']:.4f}")
    print(f"  Micro F1     : {metrics['micro_f1']:.4f}")
    print(f"\n  Per-ADR AUC (top 10 by count):")
    print(per_adr.head(10).to_string(index=False))
    print(f"{'='*55}\n")
