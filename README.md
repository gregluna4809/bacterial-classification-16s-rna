
# Bacterial Genus Classification from 16S rRNA Sequences

## Project Summary

[📄 View Project Summary Poster](docs/images/project-summary-banner.pdf)

---

# Bacterial Genus Classification from 16S rRNA Sequences

> A reproducible machine learning pipeline for classifying bacterial genera using k-mer feature engineering, dimensionality reduction, and GPU-accelerated gradient boosting.
>
> *M.S. Computer Science — Bioinformatics capstone project*

---

## Dataset

| | |
|---|---|
| **Source** | RDP Trainset 18 — 16S rRNA gene sequences |
| **Raw sequences** | ~21,000 across ~3,000 genera (before filtering) |
| **After taxonomy validation** | 18,815 valid genus-labeled sequences |
| **Excluded** | 2,380 invalid or incomplete taxonomy records |

---

## Best Performing Pipeline

```
16S rRNA FASTA
    │
    ▼
01  Taxonomy Validation & Top-100 Genera Selection
    │
    ▼
02  K-mer Feature Engineering  (CountVectorizer → sparse matrix)
    │
    ▼
03  Dimensionality Reduction   (TruncatedSVD, 128 components)
    │
    ▼
04  Class Balancing            (SMOTE k_neighbors=1, training set only)
    │
    ▼
05  Classification             (XGBoost GPU — tree_method=hist, multi:softprob)
```

### Key parameters

```text
TruncatedSVD(n_components=128)
SMOTE(k_neighbors=1)

XGBoost:
  tree_method = "hist"
  device       = "cuda"
  objective    = "multi:softprob"
```

---

## Results — Top-100 Genera

| Metric | Result |
|--------|--------|
| Accuracy | **98.5%+** |
| Macro F1 | **~0.98** |
| Weighted F1 | **~0.99** |

Training performed on an NVIDIA RTX 5070 GPU.

---

## Installation

**1. Create a virtual environment**
```bash
python -m venv .venv
```

**2. Activate**
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Training

Run the final pipeline:
```bash
python scripts/train_final_xgb_svd_smote.py \
    --data-path path/to/trainset18_062020.fa
```

Save trained artifacts:
```bash
python scripts/train_final_xgb_svd_smote.py \
    --data-path path/to/trainset18_062020.fa \
    --save-artifacts
```

Output artifacts:
```text
vectorizer.joblib
svd.joblib
label_encoder.joblib
xgboost_model.json
classification_report.txt
metrics.json
```

---

## Repository Structure

```text
bacterial-classification-16s-rna/
├── notebooks/
├── scripts/
│   ├── analyze_classes.py
│   ├── check_data.py
│   ├── test_features.py
│   ├── train_baseline.py
│   ├── train_xgb_gpu.py
│   └── train_final_xgb_svd_smote.py
├── src/
│   └── bacterial_classifier/
│       ├── data.py
│       ├── features.py
│       ├── config.py
│       └── paths.py
├── outputs/
│   └── models/
├── requirements.txt
└── README.md
```

---

## Technologies

`Python` `Biopython` `Scikit-Learn` `XGBoost` `Imbalanced-Learn` `NumPy` `SciPy` `CUDA GPU Acceleration`

---

## Future Work

- FastAPI inference service
- Web-based sequence classification demo
- Docker deployment on DigitalOcean
- Model explainability and feature importance analysis
- Additional taxonomy levels — family, order, class

---

## Author

**Gregory Luna** — M.S. Computer Science  
*Machine Learning · Bioinformatics · Data Science · GPU-Accelerated Classification*
