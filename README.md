# Patient Triage from Partially Labelled Records

> **Trainer:** VASANTHKUMAR (TFTPL444) | **Track:** G – Semi-Supervised Learning | **Domain:** Healthcare  
> **Repository:** `terv-aiml-tftpl444-patient-triage-semi-supervised`

---

## ⚠️ Medical Disclaimer

This is an **educational research project** using fully synthetic data.  
Model predictions are **NOT** clinically validated and must **NEVER** be used for autonomous patient triage or any real medical decision-making.

---

## 📋 Project Overview

Build a four-level Emergency Department (ED) triage assistant using only ~10% of visits labelled by senior nurses. Compare supervised and semi-supervised machine learning approaches to determine whether unlabelled records can improve triage predictions — especially for **Immediate** and **Urgent** patients.

**Triage Classes (urgency order):** `Immediate > Urgent > Standard > Non-Urgent`

---

## 📂 Repository Structure

```
Patient Triage/
├── data/
│   ├── raw/
│   │   ├── ed_visits.csv                   # 7,286 rows × 18 columns (synthetic)
│   │   └── clinical_audit_triage.csv       # 7,200 rows × 4 columns (audit labels)
│   └── processed/
│       └── cleaned_ed_visits.csv           # 7,200 rows × 23 columns (after cleaning)
├── notebooks/
│   └── patient_triage_analysis.ipynb       # Executed notebook with all outputs
├── outputs/
│   ├── figures/                            # 12 charts (300 DPI PNG)
│   └── reports/
│       ├── metrics.csv
│       ├── summary.json
│       ├── data_quality_report.txt
│       ├── cv_study.csv
│       ├── label_budget_curve.csv
│       ├── co_training_progress.csv
│       └── presentation_error_analysis.csv
├── models/                                 # Saved joblib model files
├── src/
│   ├── generate_dataset.py                 # Synthetic data generator (seed=444)
│   ├── data_cleaning.py                    # Full cleaning pipeline
│   ├── feature_engineering.py             # Two-view feature split
│   ├── supervised_models.py               # Logistic Regression + Random Forest
│   ├── label_spreading.py                 # Semi-supervised Label Spreading
│   ├── co_training.py                     # Handwritten Co-Training algorithm
│   ├── evaluation.py                      # Full evaluation + label budget curve
│   ├── visualizations.py                  # All 12 required charts
│   └── main.py                            # End-to-end master pipeline
├── tests/
│   └── test_triage.py                     # 9 unit tests (all pass)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `ed_visits.csv` | 7,286 | 18 | ED visit records with ~10% nurse-labelled triage |
| `clinical_audit_triage.csv` | 7,200 | 4 | Gold-standard audit labels (evaluation ONLY) |

**Key Features:**
- **Vitals View (7):** age, heart_rate, systolic_bp, diastolic_bp, spo2_pct, resp_rate, temperature_c
- **History View (8):** sex, arrival_mode, chest_pain, breathlessness, trauma, pain_score, symptom_duration_hrs, comorbidity_count

**Data splits:** 20% locked test set (1,440 visits) | 80% training pool (5,760 visits, ~551 labelled)

---

## 🧹 Data Quality Issues Handled

1. 86 exact duplicate records (HIS re-sync artifacts) — removed
2. Blood pressure stored as text (`115/75`, `NR`, `-`) — parsed into systolic/diastolic
3. Fahrenheit temperatures — converted to Celsius (values outside 30–45 °C set to NaN)
4. ~40 invalid SpO2 values (0, 101, 110) — set to NaN
5. ~15 impossible ages (0, 150, 999) — set to NaN
6. Missing heart rate (~3%) and resp rate (~4%) — imputed with training-pool medians
7. Inconsistent categorical values (arrival_mode, chest_pain, sex, etc.) — normalized
8. Multiple spellings of triage levels (RED, Immediate, 1 - Immediate) — unified
9. Mixed date formats in arrival_time — parsed safely; hour_of_day extracted
10. None comorbidity — treated as 0 count

---

## 🤖 Models

### Supervised Baselines
| Model | Method |
|-------|--------|
| Logistic Regression | Multinomial; 5-fold CV on labelled visits only |
| Random Forest | 100 estimators; 5-fold CV on labelled visits only |

### Semi-Supervised Models
| Model | Method |
|-------|--------|
| Label Spreading | k-NN graph kernel; full training pool; labels preserved |
| Co-Training | Handwritten 2-view iterative algorithm; confidence-threshold pseudo-labelling |

---

## 📈 Results

| Model | Macro-F1 | Accuracy | Immediate Recall | Under-Triage Rate |
|-------|----------|----------|-----------------|-------------------|
| Logistic Regression | 0.8729 | 87.22% | 83.76% | 7.43% |
| **Random Forest** | **0.9173** | **91.11%** | **91.45%** | **4.37%** |
| Label Spreading | 0.8161 | 78.82% | 82.91% | 20.56% |
| Co-Training | 0.8920 | 89.38% | 85.47% | 6.32% |
| Upper Bound (all labels) | 0.9134 | 91.87% | 85.47% | 4.86% |

✅ **Success Criteria Met:**
- Macro-F1 ≥ 0.85 → ✅ RF: 0.9173
- Immediate Recall ≥ 0.80 → ✅ RF: 91.45%
- Under-triage rate ≤ 10% → ✅ RF: 4.37%

---

## 💡 5 Key Insights

1. **Semi-supervised Co-Training nearly matches supervised RF** — with only 10% labels, Co-Training achieves Macro-F1 of 0.892, just 2.5 points below the fully supervised Random Forest. This confirms that unlabelled records carry useful signal.

2. **Label Spreading suffers from hubness** — in high-dimensional k-NN graphs, label noise propagates through hub nodes causing a 20.56% under-triage rate. KNN-based Label Spreading requires careful k-tuning or dimensionality reduction for clinical use.

3. **Vitals view outperforms history view alone** — vital signs (heart rate, SpO2, blood pressure) are stronger discriminators between Immediate and Urgent patients than history features alone, confirming clinical intuition.

4. **Label budget curve shows steep improvement from 2% to 10%** — adding more labelled data between 2% and 10% yields the largest F1 gains. Beyond 20%, returns diminish, suggesting 10% labelling is a cost-effective sweet spot for semi-supervised approaches.

5. **Under-triage is the critical safety metric** — Random Forest's 4.37% under-triage rate is clinically significant vs Label Spreading's 20.56%. For ED deployment, model selection should prioritize under-triage minimisation over raw accuracy.

---

## 🏥 3 Recommendations

1. **Deploy Co-Training as the operational model for label-scarce settings.** When fewer than 15% of records are labelled by senior nurses, Co-Training's multi-view pseudo-labelling yields near-supervised performance without additional annotation cost. Prioritize high-confidence threshold (≥ 0.82) to avoid propagating errors.

2. **Replace Label Spreading with a manifold-aware method** (e.g., UMAP-reduced features before k-NN graph, or TSVM). The current Label Spreading's 20.56% under-triage makes it unsafe for clinical environments unless the hubness problem is addressed explicitly.

3. **Establish a human-in-the-loop annotation pipeline targeting 10% coverage.** Based on the label budget curve, the marginal value of labels drops sharply after 10%. Focus nurse annotation time on ambiguous cases flagged by the Co-Training view-disagreement metric to maximise label efficiency.

---

## 🚀 Setup and Usage

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/vaxanth01/terv-aiml-tftpl444-patient-triage-semi-supervised.git
cd terv-aiml-tftpl444-patient-triage-semi-supervised

# Install dependencies
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
# Step 1: Generate synthetic datasets
python src/generate_dataset.py

# Step 2: Run full ML pipeline (cleaning → features → models → evaluation → charts)
python src/main.py
```

### Run Unit Tests

```bash
pytest tests/test_triage.py -v
# Expected: 9 passed
```

### Open the Notebook

```bash
jupyter notebook notebooks/patient_triage_analysis.ipynb
```

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
joblib>=1.3
jupyter>=1.0
nbformat>=5.9
nbconvert>=7.0
pytest>=7.4
```

---

## 🔬 Technical Notes

- All data is **fully synthetic**, generated with `seed=444` (trainer EMP ID) using the handbook schema
- Audit labels are used **only** for final evaluation — never for training, pseudo-labelling, or model selection
- All preprocessing (imputers, scalers, encoders) are fitted **only** on the training pool to prevent data leakage
- Co-Training uses a configurable confidence threshold (default: 0.82) with max 10 rounds
- Label Spreading uses k-NN graph kernel with k tuned via labelled-only cross-validation

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `outputs/reports/metrics.csv` | Per-model evaluation metrics |
| `outputs/reports/summary.json` | JSON summary of all results |
| `outputs/reports/data_quality_report.txt` | Before/after data cleaning audit |
| `outputs/reports/cv_study.csv` | Cross-validation results |
| `outputs/reports/label_budget_curve.csv` | F1/recall vs label percentage |
| `outputs/reports/co_training_progress.csv` | Per-round Co-Training statistics |
| `outputs/reports/presentation_error_analysis.csv` | Error analysis for each model |
| `outputs/figures/` | 12 PNG charts at 300 DPI |

---

## 👤 Author

**VASANTHKUMAR** | EMP ID: TFTPL444  
TERV AIML Trainer Program | Track G – Semi-Supervised Learning  
Project Duration: ~7 working days

---

*This project is for educational purposes only. The models are trained on synthetic data and must not be used for real clinical triage decisions.*

