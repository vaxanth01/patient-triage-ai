"""
Supervised Machine Learning Baselines Module.
Implements:
1. Multinomial Logistic Regression
2. Random Forest Classifier
Includes Stratified 5-Fold Cross Validation on senior-nurse labelled training visits only.
Evaluates single view strength (Vitals alone vs History alone) and combined views.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, recall_score


def evaluate_view_strengths(dataset_bundle, random_state=42):
    """
    Evaluates how strong each view is alone vs together using Stratified 5-Fold CV
    on labelled training visits only.
    """
    X_v = dataset_bundle['X_train_vitals'][dataset_bundle['train_labelled_mask']]
    X_h = dataset_bundle['X_train_history'][dataset_bundle['train_labelled_mask']]
    X_c = dataset_bundle['X_train_combined'][dataset_bundle['train_labelled_mask']]
    y = dataset_bundle['y_train_labelled']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)

    scoring = ['accuracy', 'f1_macro']
    cv_v = cross_validate(lr, X_v, y, cv=cv, scoring=scoring)
    cv_h = cross_validate(lr, X_h, y, cv=cv, scoring=scoring)
    cv_c = cross_validate(lr, X_c, y, cv=cv, scoring=scoring)

    results = {
        'vitals_cv_macro_f1_mean': float(np.mean(cv_v['test_f1_macro'])),
        'vitals_cv_macro_f1_std': float(np.std(cv_v['test_f1_macro'])),
        'history_cv_macro_f1_mean': float(np.mean(cv_h['test_f1_macro'])),
        'history_cv_macro_f1_std': float(np.std(cv_h['test_f1_macro'])),
        'combined_cv_macro_f1_mean': float(np.mean(cv_c['test_f1_macro'])),
        'combined_cv_macro_f1_std': float(np.std(cv_c['test_f1_macro']))
    }

    print("\n--- VIEW STRENGTH COMPARISON (5-Fold CV on Labelled Visits) ---")
    print(f"View 1 (Vitals alone): Macro-F1 = {results['vitals_cv_macro_f1_mean']:.4f} ± {results['vitals_cv_macro_f1_std']:.4f}")
    print(f"View 2 (History alone): Macro-F1 = {results['history_cv_macro_f1_mean']:.4f} ± {results['history_cv_macro_f1_std']:.4f}")
    print(f"Combined (Both Views): Macro-F1 = {results['combined_cv_macro_f1_mean']:.4f} ± {results['combined_cv_macro_f1_std']:.4f}")

    return results


def train_supervised_models(dataset_bundle, random_state=42, models_dir="models"):
    """
    Trains Logistic Regression and Random Forest on labelled training pool data.
    Saves models and returns trained model instances and CV study details.
    """
    os.makedirs(models_dir, exist_ok=True)

    X_train_c = dataset_bundle['X_train_combined'][dataset_bundle['train_labelled_mask']]
    y_train = dataset_bundle['y_train_labelled']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 1. Multinomial Logistic Regression
    lr = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=random_state
    )

    cv_lr = cross_validate(lr, X_train_c, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
    lr.fit(X_train_c, y_train)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state
    )

    cv_rf = cross_validate(rf, X_train_c, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
    rf.fit(X_train_c, y_train)

    # Save models
    joblib.dump(lr, os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))

    # Also fit upper bound model (if all visits had audit labels, for research comparison)
    lr_upper = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_upper.fit(dataset_bundle['X_train_combined'], dataset_bundle['y_train_audit_hidden'])
    joblib.dump(lr_upper, os.path.join(models_dir, "upper_bound_all_labelled.joblib"))

    cv_summary = {
        'lr_cv_accuracy': float(np.mean(cv_lr['test_accuracy'])),
        'lr_cv_macro_f1': float(np.mean(cv_lr['test_f1_macro'])),
        'rf_cv_accuracy': float(np.mean(cv_rf['test_accuracy'])),
        'rf_cv_macro_f1': float(np.mean(cv_rf['test_f1_macro'])),
    }

    print("\n--- SUPERVISED BASELINE CV RESULTS ---")
    print(f"Logistic Regression: CV Macro-F1 = {cv_summary['lr_cv_macro_f1']:.4f}, Accuracy = {cv_summary['lr_cv_accuracy']:.4f}")
    print(f"Random Forest:       CV Macro-F1 = {cv_summary['rf_cv_macro_f1']:.4f}, Accuracy = {cv_summary['rf_cv_accuracy']:.4f}")

    return {
        'logistic_regression': lr,
        'random_forest': rf,
        'upper_bound_model': lr_upper,
        'cv_summary': cv_summary
    }


if __name__ == "__main__":
    from feature_engineering import prepare_datasets_and_views
    bundle = prepare_datasets_and_views()
    strengths = evaluate_view_strengths(bundle)
    models = train_supervised_models(bundle)
