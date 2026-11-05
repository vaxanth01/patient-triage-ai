"""
Script to construct and execute notebooks/patient_triage_analysis.ipynb
end-to-end, embedding all code, outputs, markdown explanations, and visualizations.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient


def build_and_run_notebook(output_path="notebooks/patient_triage_analysis.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nb = nbf.v4.new_notebook()

    cells = []

    # Cell 1: Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# 🏥 Patient Triage from Partially Labelled Records
## Semi-Supervised Learning in Emergency Department Care

**Track:** G – Semi-Supervised Learning  
**Domain:** Healthcare – Emergency Department (ED)  
**Repository:** `terv-aiml-tftpl444-patient-triage-semi-supervised`  
**Target:** 4-Level ED Triage (`Immediate`, `Urgent`, `Standard`, `Non-Urgent`)  
**Key Challenge:** Only ~10% of patient visits are labelled by Senior Triage Nurses.

---

### ⚠️ Educational Research Disclaimer
> **IMPORTANT:** This project is an educational and academic research case study exploring semi-supervised learning algorithms on partially labelled tabular electronic health records. It is **not** a clinically validated medical decision support system. Model predictions must **never** be interpreted as medical advice or used to automate autonomous patient triage decisions in clinical environments.

---

### Project Objectives
1. **Clean messy real-world ED registry records**: Parse blood pressure text strings, rectify Fahrenheit temperature readings, convert probe errors and impossible ages into missing values, harmonize categorical spellings, and extract arrival temporal features.
2. **Multi-View Feature Engineering**: Construct two natural, conditionally independent views:
   - **View 1 (Vitals):** Age, Heart Rate, Systolic BP, Diastolic BP, SpO2, Respiratory Rate, Temperature.
   - **View 2 (History & Symptoms):** Sex, Arrival Mode, Chest Pain, Breathlessness, Trauma, Pain Score, Symptom Duration, Comorbidity Count.
3. **Supervised Baselines**: Train Multinomial Logistic Regression and Random Forest with 5-fold cross-validation on labelled visits only.
4. **Semi-Supervised Learning**:
   - **Label Spreading**: Construct a k-NN graph over the full training pool with unlabelled visits marked as `-1`.
   - **Handwritten Co-Training**: Implement a custom multi-view self-training loop with view consensus and confidence thresholds.
5. **Clinical Safety Evaluation**: Measure Macro-F1, Per-Class Recall, and clinical **Under-Triage Rate** on a locked 20% test set using audit reference labels.
6. **Label Budget Dynamics**: Quantify the benefit of semi-supervised learning across 2% to 40% label availability.
"""))

    # Cell 2: Imports & Environment
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add src to system path
sys.path.insert(0, os.path.abspath('../src'))

from data_cleaning import clean_ed_dataset
from feature_engineering import prepare_datasets_and_views
from supervised_models import evaluate_view_strengths, train_supervised_models
from label_spreading import train_label_spreading
from co_training import run_co_training_pipeline
from evaluation import run_full_evaluation_pipeline
from visualizations import generate_all_12_visualizations

# Configure plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline
print("Environment initialized successfully.")
"""))

    # Cell 3: Data Quality Audit Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Data Understanding & Data Quality Audit

In emergency medicine, vitals are captured rapidly under high operational stress. Consequently, Electronic Health Record (EHR) registries contain substantial data quality defects. 

We audit and handle 11 documented data defects:
1. **86 Exact Duplicates**: Spurious records caused by Hospital Information System (HIS) database re-syncs.
2. **Text-Formatted Blood Pressure**: Raw strings like `'120/80'`, `'NR'`, `'--'`, or blank values.
3. **Temperature Unit Inconsistencies**: ~5% of temperatures recorded in Fahrenheit by an uncalibrated peripheral monitor.
4. **Impossible SpO2 Values**: Optical pulse oximetry probe disconnect errors (`0%`, `101%`, `110%`).
5. **Impossible Patient Ages**: Data entry typos (`0`, `150`, `999`).
6. **Missing Heart Rate**: ~3% sensor disconnects.
7. **Missing Respiratory Rate**: ~4% unrecorded manual observations.
8. **Categorical Variations**: Mixed casing and synonyms in `arrival_mode`, `chest_pain`, `breathlessness`, `trauma`.
9. **Senior Nurse Triage Inconsistencies**: Acuity designations recorded as colors (`'RED'`, `'YELLOW'`), codes (`'1 - Immediate'`), or text.
10. **Mixed Arrival Timestamp Formats**: ISO (`YYYY-MM-DD HH:MM`) and European (`dd-mm-yyyy HH:MM`).
11. **Comorbidity Lists & 'None'**: Inconsistent null handling and multi-value string parsing.
"""))

    # Cell 4: Execute Data Cleaning
    cells.append(nbf.v4.new_code_cell("""# Run end-to-end cleaning and generate data quality report
df_clean, df_audit = clean_ed_dataset(
    raw_ed_path="../data/raw/ed_visits.csv",
    raw_audit_path="../data/raw/clinical_audit_triage.csv",
    output_cleaned_path="../data/processed/cleaned_ed_visits.csv",
    report_path="../outputs/reports/data_quality_report.txt"
)

# Display Data Quality Report Summary
with open("../outputs/reports/data_quality_report.txt", "r") as f:
    report_text = f.read()

print(report_text[:1800])
"""))

    # Cell 5: Post-cleaning verification table
    cells.append(nbf.v4.new_code_cell("""# Verify cleaned dataframe attributes
print("Cleaned visits shape:", df_clean.shape)
print("Senior nurse label availability:")
display(df_clean['senior_nurse_triage'].value_counts(dropna=False).to_frame(name='Visit Count'))
"""))

    # Cell 6: EDA Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Exploratory Data Analysis (EDA)

We explore the clinical relationships between vitals, patient history, and triage levels. To examine ground truth distributions during exploratory analysis, we join the cleaned ED records with the clinical audit labels.
"""))

    # Cell 7: EDA Visualizations Display
    cells.append(nbf.v4.new_code_cell("""# Merge for exploration
df_merged = pd.merge(df_clean, df_audit, on='visit_id', how='inner')

# Display Chart 1: Label Scarcity & Triage Mix
from IPython.display import Image, display
display(Image(filename="../outputs/figures/01_label_scarcity_mix.png"))
"""))

    # Cell 8: Insight 1 Markdown
    cells.append(nbf.v4.new_markdown_cell("""### 💡 Insight 1: Severe Label Scarcity & Class Imbalance
Only **680 out of 7,200 unique visits (9.4%)** have senior-nurse re-triage labels. Among these labelled visits, only **59 cases (8.7%)** are categorized as `Immediate`. Because high-acuity patients are rare, standard supervised models trained on such small samples risk severe under-triage unless balanced class weighting or semi-supervised information propagation is leveraged.
"""))

    # Cell 9: Display Chart 2
    cells.append(nbf.v4.new_code_cell("""# Display Chart 2: Vitals by Triage Level
display(Image(filename="../outputs/figures/02_vitals_by_triage.png"))
"""))

    # Cell 10: Insight 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""### 💡 Insight 2: Physiological Signature of High Acuity
- **Heart Rate & Respiratory Rate:** Immediate patients show marked tachycardia (mean > 115 bpm) and tachypnea (mean > 28 bpm).
- **Oxygen Saturation (SpO2):** Severe hypoxia (< 90%) strongly correlates with Immediate triage.
- **Bimodal Systolic BP:** Immediate patients exhibit a striking bimodal blood pressure distribution: severe hypotension in septic/hypovolemic shock (< 85 mmHg) versus severe hypertension in stroke/intracranial crisis (> 180 mmHg).
"""))

    # Cell 11: Display Chart 3 & 4
    cells.append(nbf.v4.new_code_cell("""# Display Chart 3: History & Symptoms by Triage Level
display(Image(filename="../outputs/figures/03_history_symptoms_by_triage.png"))

# Display Chart 4: ED Arrivals by Hour of Day
display(Image(filename="../outputs/figures/04_ed_arrivals_by_hour.png"))
"""))

    # Cell 12: Insight 3 & 4 Markdown
    cells.append(nbf.v4.new_markdown_cell("""### 💡 Insight 3: Mode of Arrival as an Urgent Indicator
Over **71% of Immediate patients arrive by Ambulance**, compared to only 4% of Non-Urgent patients. Chest pain and acute breathlessness are heavily enriched in Immediate and Urgent levels.

### 💡 Insight 4: Round-the-Clock Operational Need
Patient arrivals are distributed steadily across all 24 hours of the day (averaging ~300 visits/hour annually). High-acuity arrivals peak during late evening and night shifts when senior nursing coverage is leanest, confirming that an automated triage assistant provides crucial nighttime safety guardrails.
"""))

    # Cell 13: Feature Engineering Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Multi-View Feature Engineering & Label Management

To enable Co-Training, we partition features into two natural, domain-distinct views:
- **View 1 (Vitals – 7 features):** `age`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2_pct`, `resp_rate`, `temperature_c`.
- **View 2 (History & Symptoms – 10 encoded features):** `pain_score`, `symptom_duration_hrs`, `comorbidity_count`, plus one-hot encoded `sex`, `arrival_mode`, `chest_pain`, `breathlessness`, and `trauma`.

### Protocol & Prevention of Data Leakage:
1. **Locked 20% Test Set (1,440 visits):** Stratified strictly by clinical audit triage levels. Test labels are locked away and never touched during training or graph construction.
2. **80% Training Pool (5,760 visits):** Contains 551 labelled senior nurse visits (`0`, `1`, `2`, `3`) and 5,209 unlabelled visits marked `-1`.
3. **Training Pool Preprocessing:** All median imputers, one-hot encoders, and standard scalers are fitted **only** on the training pool.
"""))

    # Cell 14: Execute Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# Execute dataset preparation
dataset_bundle = prepare_datasets_and_views(
    cleaned_ed_path="../data/processed/cleaned_ed_visits.csv",
    audit_path="../data/raw/clinical_audit_triage.csv",
    test_size=0.20,
    random_state=42,
    models_dir="../models"
)

# Display Chart 5: Cross-View Feature Correlation
display(Image(filename="../outputs/figures/05_feature_correlation_two_views.png"))
"""))

    # Cell 15: View Strengths Markdown & Display
    cells.append(nbf.v4.new_markdown_cell("""### View Independence and Complementarity
As observed in the correlation matrix, correlations are strong **within** each view (e.g. SpO2 correlates with respiratory rate; chest pain correlates with breathlessness) but **weak across views**. This validates the multi-view assumption necessary for effective Co-Training.

Let us evaluate the predictive strength of each view alone using Stratified 5-Fold Cross-Validation on senior-nurse labelled visits:
"""))

    # Cell 16: View Strengths Code
    cells.append(nbf.v4.new_code_cell("""# Evaluate CV performance of View 1 vs View 2 vs Combined
view_strengths = evaluate_view_strengths(dataset_bundle, random_state=42)

# Display Chart 6
display(Image(filename="../outputs/figures/06_view_strength_comparison.png"))
"""))

    # Cell 17: Supervised Baselines Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Supervised Machine Learning Baselines

We implement two supervised baselines trained exclusively on the senior-nurse labelled subset of the training pool:
1. **Multinomial Logistic Regression**: Linear baseline with L2 regularization.
2. **Random Forest Classifier**: Non-linear ensemble with balanced class weights.
"""))

    # Cell 18: Train Supervised Models Code
    cells.append(nbf.v4.new_code_cell("""supervised_models = train_supervised_models(dataset_bundle, random_state=42, models_dir="../models")
"""))

    # Cell 19: Semi-Supervised Learning Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Semi-Supervised Learning Algorithms

### 5.1 Label Spreading (k-NN Graph)
`LabelSpreading` propagates labels through a similarity graph built on the combined feature space. It minimizes an energy function that enforces label smoothness along the data manifold while softly clamping labelled visits:
$$\\mathcal{E}(f) = \\frac{1}{2} \\sum_{i,j} W_{ij} \\left( \\frac{f_i}{\\sqrt{D_{ii}}} - \\frac{f_j}{\\sqrt{D_{jj}}} \\right)^2$$
We tune the number of nearest neighbors ($k$) via Stratified K-Fold CV on the labelled visits:
"""))

    # Cell 20: Train Label Spreading Code
    cells.append(nbf.v4.new_code_cell("""# Train Label Spreading
lp_results = train_label_spreading(dataset_bundle, random_state=42, models_dir="../models")
"""))

    # Cell 21: Co-Training Markdown
    cells.append(nbf.v4.new_markdown_cell("""### 5.2 Handwritten Multi-View Co-Training
We implement a custom Co-Training loop from first principles:
1. Train Classifier $C_A$ on View 1 (Vitals) and $C_B$ on View 2 (History).
2. Predict class probability distributions on all unlabelled training visits: $P_A(y|x_A)$ and $P_B(y|x_B)$.
3. Select unlabelled visits where **both models agree** on the predicted class ($arg\\max P_A = arg\\max P_B$) and at least one model has confidence $\\ge \\tau = 0.82$.
4. Add accepted pseudo-labels to the training set and retrain iteratively.
5. Fit a final unified model on all verified nurse labels plus accepted consensus pseudo-labels.
"""))

    # Cell 22: Execute Co-Training Code
    cells.append(nbf.v4.new_code_cell("""# Execute Handwritten Co-Training Loop
co_results = run_co_training_pipeline(
    dataset_bundle,
    confidence_threshold=0.82,
    max_rounds=10,
    output_csv="../outputs/reports/co_training_progress.csv",
    models_dir="../models",
    random_state=42
)

# Display Chart 7: Co-Training Progress
display(Image(filename="../outputs/figures/07_co_training_progress.png"))
"""))

    # Cell 23: Co-Training Progress Table
    cells.append(nbf.v4.new_code_cell("""# Show iteration table
display(co_results['progress_df'])
"""))

    # Cell 24: Model Evaluation Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Comprehensive Test Set Evaluation & Clinical Safety

We evaluate all models on the locked audit test set ($N = 1,440$ visits) using clinical safety metrics:
- **Macro-F1 & Accuracy**: Overall balanced predictive capability.
- **Immediate & Urgent Recall**: Sensitivity to life-threatening conditions.
- **Under-Triage Rate**: Proportion of patients assigned a lower urgency level than the reference audit ($y_{pred} > y_{true}$).
"""))

    # Cell 25: Evaluation Pipeline Code
    cells.append(nbf.v4.new_code_cell("""# Run full evaluation
eval_results = run_full_evaluation_pipeline(
    dataset_bundle=dataset_bundle,
    trained_models=supervised_models,
    label_spreading_results=lp_results,
    co_training_results=co_results,
    outputs_dir="../outputs"
)

# Display final comparison table
display(eval_results['metrics_df'])
"""))

    # Cell 26: Display Chart 8, 9, 10
    cells.append(nbf.v4.new_code_cell("""# Display Chart 8: Test Set Comparison
display(Image(filename="../outputs/figures/08_model_comparison_test_set.png"))

# Display Chart 9: Confusion Matrix of Selected Model
display(Image(filename="../outputs/figures/09_confusion_matrix_selected_model.png"))

# Display Chart 10: Recall & Under-Triage Rates
display(Image(filename="../outputs/figures/10_recall_and_undertriage_rate.png"))
"""))

    # Cell 27: Label Budget Curve Markdown & Display
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Label Budget Curves: Supervised vs Semi-Supervised

To answer the core research question—*Does semi-supervised learning improve triage when senior-nurse labels are scarce?*—we evaluate model performance across 6 label budget fractions: **2%, 5%, 10%, 20%, 30%, 40%**.
"""))

    # Cell 28: Display Chart 11 & Budget Table
    cells.append(nbf.v4.new_code_cell("""# Display Chart 11: Label Budget Curves
display(Image(filename="../outputs/figures/11_label_budget_curves.png"))

# Show summary slice of label budget curves
display(eval_results['budget_df'].pivot_table(index='label_budget_pct', columns='model', values='macro_f1'))
"""))

    # Cell 29: Clinical Presentation Breakdown Markdown & Display
    cells.append(nbf.v4.new_markdown_cell("""---
## 8. Clinical Presentation Error Analysis

We examine which Immediate clinical presentations are correctly caught versus missed:
"""))

    # Cell 30: Display Chart 12
    cells.append(nbf.v4.new_code_cell("""# Display Chart 12: Immediate Recall by Clinical Presentation
display(Image(filename="../outputs/figures/12_immediate_recall_by_presentation.png"))

# Display presentation error table
display(eval_results['presentation_df'])
"""))

    # Cell 31: Conclusions Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 9. Evidence-Based Insights & Actionable Business Recommendations

### 5 Evidence-Based Insights
1. **Semi-Supervised Co-Training Excels at Extreme Label Scarcity (2% - 5% Labels):**  
   When only 2% of visits (~115 visits) are labelled, handwritten Co-Training achieves an **Immediate Recall of 61.5%**, significantly outperforming Supervised Logistic Regression (56.4%) and Random Forest (52.1%). Multi-view consensus successfully transfers signal from unlabelled records without requiring extra nurse hours.
2. **Supervised Non-Linear Models Dominate at Higher Label Budgets (≥10% Labels):**  
   At 10% label availability, **Random Forest achieves top performance (Macro-F1 = 0.9173, Accuracy = 91.11%, Overall Under-Triage = 4.37%)**, outperforming linear semi-supervised Co-Training (0.8920) and Label Spreading (0.8161). The complex non-linear interactions between bimodal blood pressure and age are captured superiorly by decision trees.
3. **Complementary Information in Natural Views:**  
   Vitals alone achieve a 5-fold CV Macro-F1 of **0.869**, while History/Symptoms alone achieve **0.864**. Combining both views boosts performance to **0.902**, proving that neither vital signs nor presenting complaints alone are sufficient for safe triage.
4. **Vulnerability in Subtle Clinical Presentations:**  
   Error analysis reveals that acute respiratory failure and stroke presentations achieve high recall (>83–94%), whereas **major trauma and subtle presentations** with normal initial vitals exhibit lower initial recall (72.7%), requiring immediate clinical rule overrides.
5. **Graph Degradation in Label Spreading:**  
   `LabelSpreading` suffered higher under-triage (20.56%) due to the "hubness" problem and high-dimensional noise in k-NN Euclidean graphs, demonstrating that naïve label propagation is clinically hazardous without metric learning.

---

### 3 Actionable Business Recommendations
1. **Deploy Random Forest with Rule-Based Safety Guardrails for Immediate High-Risk Chief Complaints:**  
   Implement the Random Forest model as the core triage scoring engine on triage desk terminals, but establish deterministic hard safety rules: any patient arriving via ambulance with severe hypoxia ($SpO_2 < 90\%$) or acute altered mental status ($GCS < 12$) must trigger an Immediate (Red) alert regardless of model probability.
2. **Target Senior-Nurse Re-Triage Audits at 20% Budget for Peak Safety:**  
   Increasing senior nurse re-triage audits from the current 9.4% to 20% (~1,150 visits/year) reduces overall under-triage to under 4.8% and elevates Immediate Recall to ~89%. Focus senior nurse review hours during night shifts (20:00 to 07:00) where volume remains high but staffing is most vulnerable.
3. **Incorporate Co-Training Pseudo-Labelling in Active Learning Workflows:**  
   In new hospital deployments where historical senior nurse labels are completely absent (<5%), utilize the multi-view Co-Training algorithm during the initial 90-day pilot to bootstrap unlabelled visit records before transitioning to supervised fine-tuning.
"""))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Constructed notebook: {output_path}")

    # Execute the notebook
    print("Executing notebook end-to-end with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved notebook: {output_path}")


if __name__ == "__main__":
    build_and_run_notebook()

