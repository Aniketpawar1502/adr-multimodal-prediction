# Multi-Modal Deep Learning Framework for Adverse Drug Reaction Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/PyG-2.x-orange" />
  <img src="https://img.shields.io/badge/RDKit-2024-green" />
  <img src="https://img.shields.io/badge/BioBERT-v1.1-blueviolet" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **M.Tech Thesis | National Institute of Technology, Warangal**  
> **Author:** Pawar Aniket Satish (24BTM1R02) | **Supervisor:** Prof. M. Jerold  
> **Department:** Biotechnology | **Year:** 2025–2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Model Architecture](#3-model-architecture)
4. [Results](#4-results)
5. [Dataset](#5-dataset)
6. [Technology Stack](#6-technology-stack)
7. [Computational Resources](#7-computational-resources)
8. [Folder Structure](#8-folder-structure)
9. [Installation](#9-installation)
10. [Usage](#10-usage)
11. [Example Output](#11-example-output)
12. [Future Improvements](#12-future-improvements)
13. [Contributing](#13-contributing)
14. [License](#14-license)
15. [Author](#15-author)

---

## 1. Project Overview

Adverse Drug Reactions (ADRs) are harmful, unintended effects that occur when medicines are taken at recommended doses. They account for nearly **6% of all hospital admissions** worldwide, cause patient suffering, increase healthcare costs, and in severe cases lead to permanent organ damage or death.

Traditional ADR detection relies on clinical trials (limited by small, controlled populations) and post-marketing surveillance systems such as the **FDA Adverse Event Reporting System (FAERS)**, which suffer from under-reporting, inconsistent entries, and missing values.

This project presents a **multi-modal deep learning framework** that integrates:
- Patient-level demographic information (age, sex, weight)
- Drug chemical descriptors (16 physicochemical properties from PubChem)
- Structural drug representations — SMILES sequences (processed via CNN) and molecular graphs (processed via GNN)
- Biomedical NLP embeddings of drug indications (via BioBERT)

Three progressively sophisticated models are developed and compared: a Random Forest baseline, a CNN + Tabular model, and the final multimodal **GNN + CNN + Tabular** model — achieving a Macro AUC of **0.887** across 30 ADR categories.

---

## 2. Key Features

- **Multi-modal fusion** — combines four distinct data modalities (demographics, chemical descriptors, SMILES sequences, molecular graphs) in a single unified model.
- **30-class multi-label prediction** — simultaneously predicts 30 clinically relevant ADR types.
- **Large-scale FAERS dataset** — 10 years of FDA pharmacovigilance data (2015 Q1 – 2024 Q4).
- **BioBERT drug-indication encoding** — contextual biomedical NLP embeddings capture semantic nuance of drug indications (768-dim CLS vectors).
- **Morgan fingerprints** — 2048-bit circular fingerprints encode molecular substructure via RDKit.
- **GCN molecular graphs** — graph convolutional networks learn atom- and bond-level structural representations directly from molecular topology.
- **Reproducible training pipeline** — seeded, configurable, and modular code structure.
- **GPU-accelerated** — tested on AWS EC2 g5.4xlarge (NVIDIA A10G, 24 GB VRAM).

---

## 3. Model Architecture

### 3.1 Model 1 — Random Forest (Baseline)

A multi-output Random Forest classifier using concatenated features:

```
Input (3 demographics + 16 descriptors + 2048 Morgan FP + 768 BioBERT) → 2835-dim
  └─ MultiOutputClassifier(RandomForest, n_estimators=500, max_depth=25)
       └─ 30 binary predictions
```

### 3.2 Model 2 — CNN + Tabular

Character-level SMILES processed through parallel convolutional filters, fused with a tabular MLP branch:

```
SMILES (char-tokenised, max_len=120)
  └─ Embedding(vocab→128)
       ├─ Conv1d(128→128, k=3) + MaxPool  ─┐
       ├─ Conv1d(128→256, k=5) + MaxPool  ─┤─ Concat → 640-dim
       └─ Conv1d(128→256, k=7) + MaxPool  ─┘
                                             ↓
Tabular (16 descriptors + 3 demographics + 768 BioBERT = 787-dim)
  └─ Linear(787→128) → ReLU → Linear(128→64)
                                             ↓
Fusion: Concat(640 + 64 = 704) → FC(512) → Dropout(0.35) → FC(256) → Sigmoid(30)
```

### 3.3 Model 3 — GNN + CNN + Tabular (Best Model)

Full multimodal model adding a three-layer Graph Convolutional Network for molecular topology:

```
SMILES branch       → 640-dim   (same as Model 2)
                                             ↓
Molecular Graph (atoms as nodes, bonds as edges; node features: [atomic_num, degree, aromatic])
  └─ GCNConv(3→128) → ReLU
     GCNConv(128→128) → ReLU
     GCNConv(128→128) → ReLU
     global_mean_pool → Linear(128→256)  → 256-dim
                                             ↓
Tabular branch      → 64-dim    (same as Model 2)
                                             ↓
Fusion: Concat(640 + 256 + 64) → FC(512) → Dropout(0.4) → FC(256) → Sigmoid(30)
```

---

## 4. Results

### 4.1 Overall Performance

| Model | Macro AUC | MAP | Hamming Loss | Micro F1 |
|-------|:---------:|:---:|:------------:|:--------:|
| Random Forest (Baseline) | 0.764 | 0.647 | 0.323 | 0.786 |
| CNN + Tabular | 0.836 | 0.734 | 0.246 | 0.853 |
| **GNN + CNN + Tabular** | **0.887** | **0.805** | **0.173** | **0.894** |

The GNN + CNN + Tabular model improves Macro AUC by **+12.3 percentage points** over the Random Forest baseline and **+5.1 points** over the CNN-only model.

### 4.2 Per-ADR AUC — GNN + CNN + Tabular

| ADR | Sample Count | AUC (RF) | AUC (CNN+Tab) | AUC (GNN+CNN+Tab) |
|-----|:------------:|:--------:|:-------------:|:-----------------:|
| Nausea | 95,530 | 0.814 | 0.886 | **0.932** |
| Fatigue | 87,702 | 0.827 | 0.873 | **0.915** |
| Diarrhoea | 87,089 | 0.803 | 0.867 | **0.918** |
| Dyspnoea | 76,003 | 0.792 | 0.854 | **0.907** |
| Headache | 73,325 | 0.786 | 0.852 | **0.891** |
| Dizziness | 64,586 | 0.778 | 0.835 | **0.884** |
| Vomiting | 60,030 | 0.765 | 0.843 | **0.887** |
| Abdominal pain | 57,855 | 0.754 | 0.816 | **0.863** |
| Malaise | 50,513 | 0.761 | 0.804 | **0.856** |
| Alopecia | 50,478 | 0.746 | 0.809 | **0.852** |
| Pneumonia | 47,457 | 0.742 | 0.813 | **0.858** |
| Pruritus | 44,855 | 0.738 | 0.794 | **0.846** |
| Anxiety | 42,041 | 0.744 | 0.798 | **0.842** |
| Insomnia | 40,813 | 0.731 | 0.782 | **0.844** |
| Weight decreased | 40,042 | 0.725 | 0.791 | **0.848** |
| Anaemia | 39,658 | 0.723 | 0.784 | **0.843** |
| Arthralgia | 38,961 | 0.717 | 0.802 | **0.835** |
| Pain in extremity | 37,733 | 0.714 | 0.788 | **0.833** |
| Cough | 35,657 | 0.729 | 0.775 | **0.841** |
| Fall | 35,300 | 0.712 | 0.771 | **0.834** |
| Acute kidney injury | 34,592 | 0.718 | 0.793 | **0.837** |
| Hypertension | 33,398 | 0.722 | 0.778 | **0.845** |
| Weight increased | 30,154 | 0.706 | 0.767 | **0.823** |
| Depression | 29,800 | 0.694 | 0.758 | **0.826** |
| Back pain | 28,078 | 0.709 | 0.754 | **0.818** |
| Constipation | 26,142 | 0.701 | 0.748 | **0.814** |
| Hypotension | 25,977 | 0.697 | 0.744 | **0.806** |
| Chest Pain | 25,331 | 0.683 | 0.769 | **0.803** |
| Gait disturbance | 24,288 | 0.676 | 0.734 | **0.791** |
| Somnolence | 23,320 | 0.688 | 0.726 | **0.784** |

---

## 5. Dataset

| Property | Details |
|----------|---------|
| **Primary Source** | FDA Adverse Event Reporting System (FAERS) |
| **Time Span** | 2015 Q1 – 2024 Q4 (10 years, 40 quarterly releases) |
| **Files Used** | DEMO, DRUG, REAC, INDI (ASCII format) |
| **Chemical Data** | PubChem (CID, SMILES, 16 computed descriptors via PubChemPy) |
| **Drug Indication NLP** | MedDRA preferred terms |
| **ADR Labels** | 30 most frequent ADRs from CTCAE guidelines |
| **Final Features** | 3 demographics + 16 descriptors + 2048 Morgan FP + 768 BioBERT |

### Data Preprocessing Summary

1. Downloaded 40 quarterly FAERS ASCII releases (2015–2024); converted to CSV.
2. Merged DEMO, DRUG, REAC, INDI tables by `primaryid` and `caseid`.
3. Retained only **Primary Suspect (PS)** drugs.
4. Standardised age (to years) and weight (to kg) using `age_cod` and `wt_cod` columns.
5. Deduplicated by keeping the highest `caseversion` row per case.
6. Validated all drug names via RxNORM; fetched SMILES + 16 descriptors from PubChem.
7. Converted REAC and INDI terms to MedDRA Preferred Terms.
8. Removed rows where the ADR and the indication are the same (confound removal).
9. Removed age > 120 and weight outliers; dropped rows with any missing feature.
10. One-hot encoded 30 ADR labels; removed XlogP3 (>50% missing).

---

## 6. Technology Stack

| Category | Tool / Library | Version |
|----------|---------------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.x |
| Graph Neural Networks | PyTorch Geometric | 2.x |
| Classical ML | scikit-learn | 1.4+ |
| Cheminformatics | RDKit | 2024.03 |
| Biomedical NLP | BioBERT (dmis-lab) | v1.1 |
| NLP Framework | HuggingFace Transformers | 4.x |
| Data Processing | Pandas, NumPy | latest |
| Drug Data API | PubChemPy | 1.0.4 |
| Drug Validation | RxNORM (via requests) | — |

---

## 7. Computational Resources

All model training was performed on an **AWS EC2 g5.4xlarge** instance:

| Specification | Detail |
|--------------|--------|
| Instance Family | G5 (Accelerated Computing) |
| GPU | 1× NVIDIA A10G Tensor Core GPU |
| GPU Architecture | NVIDIA Ampere |
| GPU Memory (VRAM) | 24 GB GDDR6 |
| vCPU | 16 (AMD EPYC 7R32, ~2.8–3.3 GHz) |
| System RAM | 64 GiB |
| Local Storage | 600 GB NVMe SSD |
| Network Bandwidth | Up to 25 Gbps |
| CUDA | Supported (Tensor Cores enabled) |

---

## 8. Folder Structure

```
adr-multimodal-prediction/
│
├── src/                        # Core source code
│   ├── preprocessing.py        # Data loading, Morgan FP, BioBERT encoding
│   ├── models.py               # RF, CNN, GNN+CNN model definitions
│   ├── train.py                # Training loops with evaluation
│   ├── evaluate.py             # Metric computation (AUC, MAP, F1)
│   ├── predict.py              # Inference on new drug–patient pairs
│   └── utils.py                # Logging, seeding, helper functions
│
├── data/
│   ├── raw/                    # Raw FAERS quarterly ASCII files (not tracked)
│   └── processed/              # Merged & cleaned CSV (not tracked – see note)
│
├── results/
│   ├── figures/                # AUC curves, bar charts, architecture diagrams
│   └── metrics/                # JSON files with saved evaluation metrics
│
├── configs/
│   └── config.yaml             # All hyperparameters in one place
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA and visualisation notebook
│
├── docs/
│   └── architecture.png        # Model architecture diagram
│
├── requirements.txt            # Python dependencies with pinned versions
├── .gitignore                  # Excludes data, checkpoints, __pycache__
├── LICENSE                     # MIT License
└── README.md                   # This file
```

> **Data Note:** FAERS raw data is publicly available at [https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html). Due to size (>50 GB), raw and processed data files are excluded from this repository.

---

## 9. Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU acceleration; CPU fallback is supported)
- Conda or virtualenv recommended

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/adr-multimodal-prediction.git
cd adr-multimodal-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install PyTorch (choose your CUDA version from https://pytorch.org)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 5. Install RDKit
conda install -c conda-forge rdkit           # Recommended via conda
# OR
pip install rdkit

# 6. Install remaining dependencies
pip install -r requirements.txt
```

---

## 10. Usage

### Prepare Data

Place your merged FAERS CSV at `data/processed/faers_final.csv`.  
The CSV must contain columns: `age`, `weight`, `sex`, `smiles`, `indi_pt`, 16 descriptor columns, and `ADR_0` through `ADR_29`.

### Train Models

```bash
# Train Random Forest baseline
python src/train.py --model rf --data data/processed/faers_final.csv

# Train CNN + Tabular model
python src/train.py --model cnn --data data/processed/faers_final.csv \
    --epochs 30 --batch_size 256 --lr 0.001

# Train GNN + CNN + Tabular model (best)
python src/train.py --model gnn --data data/processed/faers_final.csv \
    --epochs 30 --batch_size 128 --lr 0.0005
```

### Run Inference on a New Drug–Patient Pair

```bash
python src/predict.py \
    --model gnn \
    --model_path results/gnn_model.pt \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --age 65 \
    --weight 72 \
    --sex 0 \
    --indication "Pain management"
```

---

## 11. Example Output

```
Predicted ADRs (threshold = 0.5):

ADR                       Probability
----------------------------------------
Nausea                         0.7821  ← PREDICTED
Fatigue                        0.6413  ← PREDICTED
Diarrhoea                      0.3194
Dyspnoea                       0.2087
Headache                       0.5572  ← PREDICTED
Dizziness                      0.4801
...

Total predicted ADRs: 3
```

---

## 12. Future Improvements

- **Drug–Drug Interaction (DDI) modelling** — extend the framework to multi-drug regimens, capturing synergistic and antagonistic ADR effects.
- **Attention mechanisms** — integrate self-attention or cross-modal attention to improve feature fusion and model interpretability.
- **Explainability (XAI)** — apply SHAP values and GNN explainability methods (GNNExplainer) to identify which molecular substructures drive each ADR prediction.
- **Transformer-based molecular encoding** — replace CNN with ChemBERTa or MolBERT for richer SMILES representations.
- **Federated learning** — enable privacy-preserving model training across multiple hospital pharmacovigilance databases.
- **REST API deployment** — wrap the best model in a FastAPI service for integration with clinical decision support systems.
- **Prospective validation** — validate predictions against confirmed ADR cases in prospective clinical datasets.

---

## 13. Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with descriptive messages.
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request.

Please ensure all new code passes basic linting (`flake8 src/`) before submitting.

---

## 14. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 15. Author

**Pawar Aniket Satish**  
M.Tech Biotechnology (2024–2026)  
Roll No.: 24BTM1R02  
National Institute of Technology, Warangal  

Supervisor: **Prof. M. Jerold**  
Assistant Professor, Department of Biotechnology, NIT Warangal

---

*If you find this work useful in your research, please consider citing the associated thesis report.*
