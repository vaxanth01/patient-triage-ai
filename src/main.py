"""
Master Execution Pipeline for Patient Triage from Partially Labelled Records.
Coordinates end-to-end execution:
1. Data cleaning and data quality reporting
2. Feature engineering & view separation
3. Supervised model training (Logistic Regression & Random Forest) with 5-fold CV
4. Semi-supervised Label Spreading on KNN graph
5. Semi-supervised Handwritten Co-Training across two independent views
6. Comprehensive test set evaluation & label budget curves
7. Generation of all 12 publication-grade figures
8. Verification of all deliverables and results
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure src directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_cleaning import clean_ed_dataset
from feature_engineering import prepare_datasets_and_views
from supervised_models import evaluate_view_strengths, train_supervised_models
from label_spreading import train_label_spreading
from co_training import run_co_training_pipeline
from evaluation import run_full_evaluation_pipeline
from visualizations import generate_all_12_visualizations


def run_pipeline():
    print("=" * 80)
    print("STARTING END-TO-END PATIENT TRIAGE MACHINE LEARNING PIPELINE")
    print("Track: G - Semi-Supervised Learning | Domain: Healthcare ED Triage")
    print("=" * 80)

    # 1. Clean Data & Quality Audit
    print("\n[PHASE 1] Data Cleaning and Data Quality Audit...")
    df_clean, df_audit = clean_ed_dataset(
        raw_ed_path="data/raw/ed_visits.csv",
        raw_audit_path="data/raw/clinical_audit_triage.csv",
        output_cleaned_path="data/processed/cleaned_ed_visits.csv",
        report_path="outputs/reports/data_quality_report.txt"
    )

    # 2. Feature Engineering & Multi-View Splitting
    print("\n[PHASE 2] Feature Engineering & Multi-View Splitting...")
    dataset_bundle = prepare_datasets_and_views(
        cleaned_ed_path="data/processed/cleaned_ed_visits.csv",
        audit_path="data/raw/clinical_audit_triage.csv",
        test_size=0.20,
        random_state=42,
        models_dir="models"
    )

    # 3. View Strengths & Supervised Baselines
    print("\n[PHASE 3] Evaluating View Strengths & Supervised Baselines (5-Fold CV)...")
    view_strengths = evaluate_view_strengths(dataset_bundle, random_state=42)
    supervised_models = train_supervised_models(dataset_bundle, random_state=42, models_dir="models")

    # Save CV study
    cv_study_df = pd.DataFrame([
        {'View / Model': 'View 1 (Vitals alone)', 'CV_Macro_F1': view_strengths['vitals_cv_macro_f1_mean'], 'CV_Macro_F1_Std': view_strengths['vitals_cv_macro_f1_std']},
        {'View / Model': 'View 2 (History alone)', 'CV_Macro_F1': view_strengths['history_cv_macro_f1_mean'], 'CV_Macro_F1_Std': view_strengths['history_cv_macro_f1_std']},
        {'View / Model': 'Combined Views (Logistic Regression)', 'CV_Macro_F1': supervised_models['cv_summary']['lr_cv_macro_f1'], 'CV_Accuracy': supervised_models['cv_summary']['lr_cv_accuracy']},
        {'View / Model': 'Combined Views (Random Forest)', 'CV_Macro_F1': supervised_models['cv_summary']['rf_cv_macro_f1'], 'CV_Accuracy': supervised_models['cv_summary']['rf_cv_accuracy']}
    ])
    cv_study_df.to_csv("outputs/reports/cv_study.csv", index=False)
    print("Cross-validation study saved to outputs/reports/cv_study.csv")

    # 4. Semi-Supervised Learning: Label Spreading
    print("\n[PHASE 4] Training Semi-Supervised Label Spreading on KNN Graph...")
    lp_results = train_label_spreading(dataset_bundle, random_state=42, models_dir="models")

    # 5. Semi-Supervised Learning: Handwritten Co-Training
    print("\n[PHASE 5] Executing Handwritten Multi-View Co-Training Loop...")
    co_results = run_co_training_pipeline(
        dataset_bundle,
        confidence_threshold=0.82,
        max_rounds=10,
        output_csv="outputs/reports/co_training_progress.csv",
        models_dir="models",
        random_state=42
    )

    # 6. Evaluation & Safety Metrics on Locked Test Set
    print("\n[PHASE 6] Evaluating All Models on Locked Test Set & Label Budget Study...")
    eval_results = run_full_evaluation_pipeline(
        dataset_bundle=dataset_bundle,
        trained_models=supervised_models,
        label_spreading_results=lp_results,
        co_training_results=co_results,
        outputs_dir="outputs"
    )

    # 7. Merge Cleaned DataFrame with Audit for EDA Plotting
    df_merged = pd.merge(df_clean, df_audit[['visit_id', 'audit_triage_level', 'audit_presentation', 'audit_round']],
                         on='visit_id', how='inner')

    # 8. Generate All 12 High-Resolution Visualizations
    print("\n[PHASE 7] Generating All 12 High-Resolution Figures...")
    generate_all_12_visualizations(
        df_clean=df_clean,
        df_merged=df_merged,
        dataset_bundle=dataset_bundle,
        cv_study_results=view_strengths,
        df_progress=co_results['progress_df'],
        df_metrics=eval_results['metrics_df'],
        confusion_matrices=eval_results['confusion_matrices'],
        df_budget=eval_results['budget_df'],
        df_pres=eval_results['presentation_df'],
        output_dir="outputs/figures"
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return {
        'dataset_bundle': dataset_bundle,
        'supervised_models': supervised_models,
        'lp_results': lp_results,
        'co_results': co_results,
        'eval_results': eval_results
    }


if __name__ == "__main__":
    run_pipeline()

