"""
utils.py
========
Shared helper utilities: logging, reproducibility seeding, and
result-saving functions.

Author : Pawar Aniket Satish (24BTM1R02), NIT Warangal
"""

import os
import random
import json
import logging
from datetime import datetime

import numpy as np
import torch


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def get_logger(name: str = "ADR", log_dir: str = "results") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt     = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh      = logging.FileHandler(log_file)
    ch      = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Saving Results
# ─────────────────────────────────────────────
def save_metrics(metrics: dict, path: str = "results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {path}")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
