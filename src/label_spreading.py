"""
Semi-Supervised Learning Module: Label Spreading.
Uses Scikit-learn LabelSpreading with a k-NN graph on the combined feature space.
Hyperparameters tuned using Stratified K-Fold CV on labelled training visits only.
Transductive training performed on the full training pool with unlabelled visits set to -1.
Predicts locked test set visits.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def tune_label_spreading_k(dataset_bundle, k_candidates=[5, 7, 10, 15, 20], random_state=42):
    """
    Tunes n_neighbors (k) for KNN graph using Stratified K-Fold CV
    on the labelled visits within the training pool.
    """
    X_c = dataset_bundle['X_train_combined']
    y_pool = dataset_bundle['y_train_pool'].copy()
    labelled_mask = dataset_bundle['train_labelled_mask']
    labelled_indices = np.where(labelled_mask)[0]
    y_labelled = y_pool[labelled_indices]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_k = 7
    best_score = -1.0
    k_scores = {}

    print("\n--- TUNING LABEL SPREADING (KNN Graph Kernel) ---")
    for k in k_candidates:
        fold_scores = []
        for train_fold_idx, val_fold_idx in cv.split(labelled_indices, y_labelled):
            # In each fold, hide the labels of the validation fold by setting to -1
            y_fold = y_pool.copy()
            val_actual_indices = labelled_indices[val_fold_idx]
            y_fold[val_actual_indices] = -1

            lp = LabelSpreading(kernel='knn', n_neighbors=k, alpha=0.8, max_iter=30)
            lp.fit(X_c, y_fold)

            # Predict on the validation fold
            val_preds = lp.predict(X_c[val_actual_indices])
            val_true = y_labelled[val_fold_idx]
            score = f1_score(val_true, val_preds, average='macro', zero_division=0)
            fold_scores.append(score)

        mean_f1 = float(np.mean(fold_scores))
        k_scores[k] = mean_f1
        print(f"  k = {k:2d}: CV Macro-F1 = {mean_f1:.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_k = k

    print(f"Selected Optimal k: {best_k} (CV Macro-F1: {best_score:.4f})")
    return best_k, k_scores


def train_label_spreading(dataset_bundle, best_k=None, random_state=42, models_dir="models"):
    """
    Fits LabelSpreading on the complete training pool (5,760 visits) with unlabelled set to -1.
    Saves the model bundle and predicts labels on the locked test set.
    """
    os.makedirs(models_dir, exist_ok=True)

    if best_k is None:
        best_k, _ = tune_label_spreading_k(dataset_bundle, random_state=random_state)

    X_train_c = dataset_bundle['X_train_combined']
    y_train_pool = dataset_bundle['y_train_pool']  # unlabelled are -1

    # Graph construction assumptions:
    # 1. k-Nearest Neighbor graph captures manifold geometry across vitals and presentation space.
    # 2. Clamping factor alpha=0.8 allows slight label smoothing while preserving nurse annotations.
    lp_model = LabelSpreading(
        kernel='knn',
        n_neighbors=best_k,
        alpha=0.8,
        max_iter=50
    )

    print(f"\nFitting LabelSpreading on training pool ({X_train_c.shape[0]} visits: {dataset_bundle['train_labelled_mask'].sum()} labelled, {(y_train_pool == -1).sum()} unlabelled)...")
    lp_model.fit(X_train_c, y_train_pool)

    # Save fitted model
    joblib.dump(lp_model, os.path.join(models_dir, "label_spreading.joblib"))

    # Predict locked test set
    y_test_pred = lp_model.predict(dataset_bundle['X_test_combined'])

    print(f"Label Spreading training complete. Model saved to {models_dir}/label_spreading.joblib")
    return {
        'model': lp_model,
        'best_k': best_k,
        'y_test_pred': y_test_pred
    }


if __name__ == "__main__":
    from feature_engineering import prepare_datasets_and_views
    bundle = prepare_datasets_and_views()
    res = train_label_spreading(bundle)

