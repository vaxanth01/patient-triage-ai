"""
Handwritten Co-Training Algorithm Module.
Implements a multi-view semi-supervised learning loop from scratch using:
- View A: Vitals Features (7 physiological variables)
- View B: History & Symptoms (8 clinical variables)
Iteratively labels confident unlabelled training visits with inter-view agreement checks.
Saves round-by-round progress to outputs/reports/co_training_progress.csv.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class HandwrittenCoTrainingClassifier:
    """
    Handwritten Co-Training implementation for multi-class ED triage.
    Trains two independent classifiers over View A (Vitals) and View B (History).
    Propagates pseudo-labels on unlabelled training visits that meet confidence
    thresholds and view consensus criteria.
    """
    def __init__(self,
                 confidence_threshold=0.85,
                 max_rounds=10,
                 batch_size_per_class=60,
                 agreement_required=True,
                 random_state=42):
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.batch_size_per_class = batch_size_per_class
        self.agreement_required = agreement_required
        self.random_state = random_state

        # Base estimators for each view
        self.clf_a = None  # View A (Vitals)
        self.clf_b = None  # View B (History)
        self.clf_combined = None  # Final joint model

        self.history_ = []

    def fit(self, X_a_pool, X_b_pool, y_pool, y_true_audit_hidden=None):
        """
        Runs the iterative Co-Training loop strictly on the training pool.
        y_pool: array with 0, 1, 2, 3 for senior nurse labels and -1 for unlabelled.
        y_true_audit_hidden: hidden ground-truth labels for auditing pseudo-label accuracy (NEVER used to guide selection).
        """
        n_samples = len(y_pool)
        current_labels = y_pool.copy()

        # Track indices
        labelled_mask = (current_labels != -1)
        unlabelled_mask = ~labelled_mask

        self.history_ = []

        print("\n" + "=" * 65)
        print("STARTING HANDWRITTEN CO-TRAINING ALGORITHM")
        print("=" * 65)
        print(f"Initial labelled visits: {labelled_mask.sum()} | Unlabelled pool: {unlabelled_mask.sum()}")
        print(f"Confidence threshold: {self.confidence_threshold} | Agreement required: {self.agreement_required}")
        print("-" * 65)

        for round_idx in range(1, self.max_rounds + 1):
            if unlabelled_mask.sum() == 0:
                print(f"Round {round_idx}: Unlabelled pool exhausted. Stopping.")
                break

            # 1. Train model on View A (Vitals)
            clf_a = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=self.random_state)
            clf_a.fit(X_a_pool[labelled_mask], current_labels[labelled_mask])

            # 2. Train model on View B (History)
            clf_b = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=self.random_state)
            clf_b.fit(X_b_pool[labelled_mask], current_labels[labelled_mask])

            # 3. Predict probabilities on currently unlabelled visits
            unlabelled_indices = np.where(unlabelled_mask)[0]
            probs_a = clf_a.predict_proba(X_a_pool[unlabelled_indices])
            probs_b = clf_b.predict_proba(X_b_pool[unlabelled_indices])

            preds_a = np.argmax(probs_a, axis=1)
            preds_b = np.argmax(probs_b, axis=1)

            max_prob_a = np.max(probs_a, axis=1)
            max_prob_b = np.max(probs_b, axis=1)

            # 4. Measure agreement between the two views on unlabelled data
            agreement_mask = (preds_a == preds_b)
            agreement_rate = float(np.mean(agreement_mask))

            # 5. Identify high-confidence pseudo-label candidates
            # High-confidence consensus: both models agree and confidence >= threshold
            if self.agreement_required:
                candidate_mask = agreement_mask & ((max_prob_a >= self.confidence_threshold) | (max_prob_b >= self.confidence_threshold))
                selected_preds = preds_a[candidate_mask]
                selected_indices = unlabelled_indices[candidate_mask]
            else:
                # View consensus via probability averaging
                avg_probs = (probs_a + probs_b) / 2.0
                max_avg_prob = np.max(avg_probs, axis=1)
                candidate_mask = (max_avg_prob >= self.confidence_threshold)
                selected_preds = np.argmax(avg_probs[candidate_mask], axis=1)
                selected_indices = unlabelled_indices[candidate_mask]

            n_selected = len(selected_indices)

            # Cap additions per class to avoid runaway majority-class drift
            if n_selected > 0:
                final_selected_idx = []
                final_selected_preds = []
                for cls in range(4):
                    cls_mask = (selected_preds == cls)
                    cls_indices = selected_indices[cls_mask]
                    cls_preds = selected_preds[cls_mask]
                    if len(cls_indices) > self.batch_size_per_class:
                        # Take top confidence
                        cls_probs = np.maximum(max_prob_a[candidate_mask][cls_mask], max_prob_b[candidate_mask][cls_mask])
                        top_rank = np.argsort(cls_probs)[::-1][:self.batch_size_per_class]
                        final_selected_idx.extend(cls_indices[top_rank])
                        final_selected_preds.extend(cls_preds[top_rank])
                    else:
                        final_selected_idx.extend(cls_indices)
                        final_selected_preds.extend(cls_preds)

                final_selected_idx = np.array(final_selected_idx)
                final_selected_preds = np.array(final_selected_preds)
                n_added = len(final_selected_idx)
            else:
                n_added = 0

            # Compute pseudo-label accuracy against hidden audit ground truth
            if n_added > 0 and y_true_audit_hidden is not None:
                true_labels_added = y_true_audit_hidden[final_selected_idx]
                pseudo_acc = float(accuracy_score(true_labels_added, final_selected_preds))
            else:
                pseudo_acc = np.nan

            # Log round statistics
            round_log = {
                'round': round_idx,
                'labelled_count': int(labelled_mask.sum()),
                'unlabelled_count': int(unlabelled_mask.sum()),
                'newly_pseudo_labelled': int(n_added),
                'view_agreement_rate': float(round(agreement_rate, 4)),
                'pseudo_label_accuracy': float(round(pseudo_acc, 4)) if not np.isnan(pseudo_acc) else np.nan
            }
            self.history_.append(round_log)

            print(f"Round {round_idx:2d} | Labelled: {labelled_mask.sum():4d} | Unlabelled: {unlabelled_mask.sum():4d} | "
                  f"Accepted: {n_added:4d} | Agreement: {agreement_rate*100:5.1f}% | Pseudo-Acc: {pseudo_acc*100:5.1f}%")

            # Check stopping conditions
            if n_added == 0:
                print(f"Stopping: No candidate met confidence threshold {self.confidence_threshold}.")
                break

            # Update labelled sets
            current_labels[final_selected_idx] = final_selected_preds
            labelled_mask[final_selected_idx] = True
            unlabelled_mask[final_selected_idx] = False

        # Store final models
        self.clf_a = clf_a
        self.clf_b = clf_b

        # 6. Fit Final Combined Model on all senior nurse + high-confidence pseudo-labelled visits
        X_combined = np.hstack([X_a_pool, X_b_pool])
        self.clf_combined = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1.0,
            random_state=self.random_state
        )
        self.clf_combined.fit(X_combined[labelled_mask], current_labels[labelled_mask])

        total_accepted = sum(r['newly_pseudo_labelled'] for r in self.history_)
        print("-" * 65)
        print(f"Co-Training Complete: {total_accepted} pseudo-labels accepted over {len(self.history_)} rounds.")
        print(f"Final training set size: {labelled_mask.sum()} visits.")
        return self

    def predict(self, X_combined_test):
        """Predicts triage classes for the locked test set."""
        return self.clf_combined.predict(X_combined_test)

    def predict_proba(self, X_combined_test):
        """Predicts triage class probabilities for the locked test set."""
        return self.clf_combined.predict_proba(X_combined_test)


def run_co_training_pipeline(dataset_bundle,
                            confidence_threshold=0.82,
                            max_rounds=10,
                            output_csv="outputs/reports/co_training_progress.csv",
                            models_dir="models",
                            random_state=42):
    """
    Executes the handwritten Co-Training loop, records progress, and saves artifacts.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_a_train = dataset_bundle['X_train_vitals']
    X_b_train = dataset_bundle['X_train_history']
    y_train_pool = dataset_bundle['y_train_pool']
    y_train_hidden = dataset_bundle['y_train_audit_hidden']

    co_trainer = HandwrittenCoTrainingClassifier(
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
        batch_size_per_class=80,
        agreement_required=True,
        random_state=random_state
    )

    co_trainer.fit(
        X_a_pool=X_a_train,
        X_b_pool=X_b_train,
        y_pool=y_train_pool,
        y_true_audit_hidden=y_train_hidden
    )

    # Save progress DataFrame
    df_progress = pd.DataFrame(co_trainer.history_)
    df_progress.to_csv(output_csv, index=False)
    print(f"Co-Training progress saved to {output_csv}")

    # Save co-training model bundle
    joblib.dump(co_trainer, os.path.join(models_dir, "co_training_model.joblib"))

    # Predict test set
    X_test_c = dataset_bundle['X_test_combined']
    y_test_pred = co_trainer.predict(X_test_c)

    return {
        'model': co_trainer,
        'progress_df': df_progress,
        'y_test_pred': y_test_pred
    }


if __name__ == "__main__":
    from feature_engineering import prepare_datasets_and_views
    bundle = prepare_datasets_and_views()
    res = run_co_training_pipeline(bundle)
