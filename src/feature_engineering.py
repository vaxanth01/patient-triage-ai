"""
Feature Engineering Module - Two Independent Views and Data Splitting.
View 1: Vitals (7 features)
View 2: History & Symptoms (8 features)
Locked 20% test set stratified by audit labels; 80% training pool with -1 for unlabelled visits.
No data leakage: all imputations and scalers fit ONLY on training pool.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

LABEL_MAP = {
    'Immediate': 0,
    'Urgent': 1,
    'Standard': 2,
    'Non-Urgent': 3
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

VITALS_FEATURES = [
    'age',
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'spo2_pct',
    'resp_rate',
    'temperature_c'
]

HISTORY_NUM_FEATURES = [
    'pain_score',
    'symptom_duration_hrs',
    'comorbidity_count'
]

HISTORY_CAT_FEATURES = [
    'sex',
    'arrival_mode',
    'chest_pain',
    'breathlessness',
    'trauma'
]


def prepare_datasets_and_views(cleaned_ed_path="data/processed/cleaned_ed_visits.csv",
                               audit_path="data/raw/clinical_audit_triage.csv",
                               test_size=0.20,
                               random_state=42,
                               models_dir="models"):
    """
    Splits data into locked test set and training pool, builds two feature views,
    fits preprocessing strictly on training pool, and returns preprocessed matrices.
    """
    os.makedirs(models_dir, exist_ok=True)

    df_clean = pd.read_csv(cleaned_ed_path)
    df_audit = pd.read_csv(audit_path)

    # 1. Match records by visit_id
    df_merged = pd.merge(df_clean, df_audit[['visit_id', 'audit_triage_level', 'audit_presentation', 'audit_round']],
                         on='visit_id', how='inner')

    # Encode audit labels (0: Immediate, 1: Urgent, 2: Standard, 3: Non-Urgent)
    df_merged['audit_label'] = df_merged['audit_triage_level'].map(LABEL_MAP)

    # Encode senior nurse labels (unlabelled becomes -1)
    df_merged['senior_label'] = df_merged['senior_nurse_triage'].map(LABEL_MAP).fillna(-1).astype(int)

    # 2. Stratified train/test split (20% locked test set based on audit labels)
    train_pool_df, test_df = train_test_split(
        df_merged,
        test_size=test_size,
        stratify=df_merged['audit_label'],
        random_state=random_state
    )

    train_pool_df = train_pool_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # In training pool, audit labels are strictly HIDDEN from models.
    # Training pool uses 'senior_label' (-1 for unlabelled, 0..3 for labelled).
    # Test set uses 'audit_label' (0..3) ONLY for final evaluation.

    # 3. View 1 Preprocessor (Vitals - 7 numerical features)
    vitals_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 4. View 2 Preprocessor (History & Symptoms - 3 numerical + 5 categorical features)
    history_preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), HISTORY_NUM_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), HISTORY_CAT_FEATURES)
    ])

    # 5. Fit preprocessors ONLY on training pool
    vitals_pipeline.fit(train_pool_df[VITALS_FEATURES])
    history_preprocessor.fit(train_pool_df[HISTORY_NUM_FEATURES + HISTORY_CAT_FEATURES])

    # Save fitted pipelines
    joblib.dump(vitals_pipeline, os.path.join(models_dir, "vitals_preprocessor.joblib"))
    joblib.dump(history_preprocessor, os.path.join(models_dir, "history_preprocessor.joblib"))

    # 6. Transform View 1 (Vitals)
    X_train_vitals = vitals_pipeline.transform(train_pool_df[VITALS_FEATURES])
    X_test_vitals = vitals_pipeline.transform(test_df[VITALS_FEATURES])

    # 7. Transform View 2 (History)
    X_train_history = history_preprocessor.transform(train_pool_df[HISTORY_NUM_FEATURES + HISTORY_CAT_FEATURES])
    X_test_history = history_preprocessor.transform(test_df[HISTORY_NUM_FEATURES + HISTORY_CAT_FEATURES])

    # 8. Combined View (Both Views concatenated)
    X_train_combined = np.hstack([X_train_vitals, X_train_history])
    X_test_combined = np.hstack([X_test_vitals, X_test_history])

    # Get feature names
    vitals_names = list(VITALS_FEATURES)
    cat_encoder = history_preprocessor.named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(HISTORY_CAT_FEATURES))
    history_names = list(HISTORY_NUM_FEATURES) + cat_feature_names
    combined_names = vitals_names + history_names

    # Identify labelled mask in training pool
    train_labelled_mask = (train_pool_df['senior_label'] != -1)

    dataset_bundle = {
        # Feature Matrices
        'X_train_vitals': X_train_vitals,
        'X_test_vitals': X_test_vitals,
        'X_train_history': X_train_history,
        'X_test_history': X_test_history,
        'X_train_combined': X_train_combined,
        'X_test_combined': X_test_combined,
        # Feature Names
        'vitals_names': vitals_names,
        'history_names': history_names,
        'combined_names': combined_names,
        # Labels & Masks
        'y_train_pool': train_pool_df['senior_label'].values,  # has -1 for unlabelled
        'y_train_labelled': train_pool_df.loc[train_labelled_mask, 'senior_label'].values,
        'train_labelled_mask': train_labelled_mask.values,
        'y_test_audit': test_df['audit_label'].values,  # ground truth for evaluation only
        'y_train_audit_hidden': train_pool_df['audit_label'].values,  # for evaluating pseudo-label accuracy ONLY
        # Dataframes & Identifiers
        'train_pool_df': train_pool_df,
        'test_df': test_df,
        'train_visit_ids': train_pool_df['visit_id'].values,
        'test_visit_ids': test_df['visit_id'].values
    }

    print(f"Dataset preparation complete:")
    print(f"  Training pool: {X_train_combined.shape[0]} visits (Labelled by Senior Nurse: {train_labelled_mask.sum()})")
    print(f"  Locked Test set: {X_test_combined.shape[0]} visits")
    print(f"  View 1 (Vitals) features: {X_train_vitals.shape[1]}")
    print(f"  View 2 (History) features: {X_train_history.shape[1]}")
    print(f"  Combined features: {X_train_combined.shape[1]}")

    return dataset_bundle


if __name__ == "__main__":
    bundle = prepare_datasets_and_views()

