"""
Evaluation and Safety Analysis Module.
Calculates:
- Standard ML metrics: Accuracy, Macro-F1, Per-Class Precision & Recall
- Healthcare Safety Metrics: Immediate Recall, Urgent Recall, Under-Triage Rates (Overall, Immediate, Urgent)
- Label Budget Study (2% to 40% senior-nurse label availability)
- Error Analysis by Clinical Presentation (e.g. Sepsis vs Stroke)
- Generates outputs/reports/metrics.csv, outputs/metrics.csv, and outputs/reports/summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading

CLASS_NAMES = ['Immediate', 'Urgent', 'Standard', 'Non-Urgent']


def compute_undertriage_metrics(y_true, y_pred):
    """
    Computes clinical under-triage rates based on urgency hierarchy:
    Immediate (0) > Urgent (1) > Standard (2) > Non-Urgent (3)
    A patient is under-triaged when y_pred > y_true (assigned a less urgent category).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_n = len(y_true)

    # Overall under-triage
    undertriaged_mask = (y_pred > y_true)
    overall_undertriage_rate = float(np.mean(undertriaged_mask))

    # Over-triage (for completeness)
    overtriaged_mask = (y_pred < y_true)
    overall_overtriage_rate = float(np.mean(overtriaged_mask))

    # Immediate under-triage: true is Immediate (0), predicted is 1, 2, or 3
    imm_mask = (y_true == 0)
    if imm_mask.sum() > 0:
        imm_undertriage_rate = float(np.mean(y_pred[imm_mask] > 0))
        imm_recall = float(np.mean(y_pred[imm_mask] == 0))
    else:
        imm_undertriage_rate = 0.0
        imm_recall = 0.0

    # Urgent under-triage: true is Urgent (1), predicted is Standard (2) or Non-Urgent (3)
    urg_mask = (y_true == 1)
    if urg_mask.sum() > 0:
        urg_undertriage_rate = float(np.mean(y_pred[urg_mask] > 1))
        urg_recall = float(np.mean(y_pred[urg_mask] == 1))
    else:
        urg_undertriage_rate = 0.0
        urg_recall = 0.0

    # Standard under-triage: true is Standard (2), predicted is Non-Urgent (3)
    std_mask = (y_true == 2)
    std_undertriage_rate = float(np.mean(y_pred[std_mask] > 2)) if std_mask.sum() > 0 else 0.0

    return {
        'overall_undertriage_rate': overall_undertriage_rate,
        'overall_overtriage_rate': overall_overtriage_rate,
        'immediate_undertriage_rate': imm_undertriage_rate,
        'urgent_undertriage_rate': urg_undertriage_rate,
        'standard_undertriage_rate': std_undertriage_rate,
        'immediate_recall': imm_recall,
        'urgent_recall': urg_recall
    }


def evaluate_model_predictions(model_name, y_true, y_pred):
    """
    Computes comprehensive performance metrics for a model's test predictions.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    ut_metrics = compute_undertriage_metrics(y_true, y_pred)

    record = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Macro_F1': round(macro_f1, 4),
        'Weighted_F1': round(weighted_f1, 4),
        'Immediate_Recall': round(rec_per_class[0], 4),
        'Urgent_Recall': round(rec_per_class[1], 4),
        'Standard_Recall': round(rec_per_class[2], 4),
        'NonUrgent_Recall': round(rec_per_class[3], 4),
        'Immediate_Precision': round(prec_per_class[0], 4),
        'Urgent_Precision': round(prec_per_class[1], 4),
        'Standard_Precision': round(prec_per_class[2], 4),
        'NonUrgent_Precision': round(prec_per_class[3], 4),
        'Overall_Under_Triage_Rate': round(ut_metrics['overall_undertriage_rate'], 4),
        'Immediate_Under_Triage_Rate': round(ut_metrics['immediate_undertriage_rate'], 4),
        'Urgent_Under_Triage_Rate': round(ut_metrics['urgent_undertriage_rate'], 4),
        'confusion_matrix': cm.tolist()
    }
    return record


def run_label_budget_study(dataset_bundle,
                           budgets=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40],
                           output_csv="outputs/reports/label_budget_curve.csv",
                           random_state=42):
    """
    Studies performance across different label budget fractions of the training pool:
    2%, 5%, 10%, 20%, 30%, 40% senior-nurse label availability.
    Evaluates Supervised LR, Supervised RF, Label Spreading, and Co-Training.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    X_train_c = dataset_bundle['X_train_combined']
    X_train_v = dataset_bundle['X_train_vitals']
    X_train_h = dataset_bundle['X_train_history']

    X_test_c = dataset_bundle['X_test_combined']
    y_test = dataset_bundle['y_test_audit']

    y_train_audit = dataset_bundle['y_train_audit_hidden']
    n_train = len(y_train_audit)

    rows = []

    print("\n" + "=" * 65)
    print("RUNNING LABEL BUDGET STUDY (2% - 40% Senior-Nurse Labels)")
    print("=" * 65)

    from co_training import HandwrittenCoTrainingClassifier

    for budget in budgets:
        n_labelled = int(round(n_train * budget))
        np.random.seed(random_state + int(budget * 100))

        # Stratified sampling of budget from hidden training audit labels to simulate senior nurse sampling
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=budget, random_state=random_state)
        sampled_train_idx, _ = next(sss.split(X_train_c, y_train_audit))

        y_budget_pool = np.full(n_train, -1)
        y_budget_pool[sampled_train_idx] = y_train_audit[sampled_train_idx]

        X_budget_labelled = X_train_c[sampled_train_idx]
        y_budget_labelled = y_train_audit[sampled_train_idx]

        # 1. Supervised Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr.fit(X_budget_labelled, y_budget_labelled)
        pred_lr = lr.predict(X_test_c)
        ut_lr = compute_undertriage_metrics(y_test, pred_lr)
        rows.append({
            'label_budget_pct': int(budget * 100),
            'n_labelled_visits': n_labelled,
            'model': 'Logistic Regression (Supervised)',
            'macro_f1': round(f1_score(y_test, pred_lr, average='macro', zero_division=0), 4),
            'immediate_recall': round(recall_score(y_test, pred_lr, average=None, zero_division=0)[0], 4),
            'urgent_recall': round(recall_score(y_test, pred_lr, average=None, zero_division=0)[1], 4),
            'under_triage_rate': round(ut_lr['overall_undertriage_rate'], 4)
        })

        # 2. Supervised Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=random_state)
        rf.fit(X_budget_labelled, y_budget_labelled)
        pred_rf = rf.predict(X_test_c)
        ut_rf = compute_undertriage_metrics(y_test, pred_rf)
        rows.append({
            'label_budget_pct': int(budget * 100),
            'n_labelled_visits': n_labelled,
            'model': 'Random Forest (Supervised)',
            'macro_f1': round(f1_score(y_test, pred_rf, average='macro', zero_division=0), 4),
            'immediate_recall': round(recall_score(y_test, pred_rf, average=None, zero_division=0)[0], 4),
            'urgent_recall': round(recall_score(y_test, pred_rf, average=None, zero_division=0)[1], 4),
            'under_triage_rate': round(ut_rf['overall_undertriage_rate'], 4)
        })

        # 3. Label Spreading
        lp = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.8, max_iter=30)
        lp.fit(X_train_c, y_budget_pool)
        pred_lp = lp.predict(X_test_c)
        ut_lp = compute_undertriage_metrics(y_test, pred_lp)
        rows.append({
            'label_budget_pct': int(budget * 100),
            'n_labelled_visits': n_labelled,
            'model': 'Label Spreading (Semi-Supervised)',
            'macro_f1': round(f1_score(y_test, pred_lp, average='macro', zero_division=0), 4),
            'immediate_recall': round(recall_score(y_test, pred_lp, average=None, zero_division=0)[0], 4),
            'urgent_recall': round(recall_score(y_test, pred_lp, average=None, zero_division=0)[1], 4),
            'under_triage_rate': round(ut_lp['overall_undertriage_rate'], 4)
        })

        # 4. Handwritten Co-Training
        cotrain = HandwrittenCoTrainingClassifier(confidence_threshold=0.82, max_rounds=5, random_state=random_state)
        cotrain.fit(X_train_v, X_train_h, y_budget_pool, y_true_audit_hidden=y_train_audit)
        pred_co = cotrain.predict(X_test_c)
        ut_co = compute_undertriage_metrics(y_test, pred_co)
        rows.append({
            'label_budget_pct': int(budget * 100),
            'n_labelled_visits': n_labelled,
            'model': 'Co-Training (Semi-Supervised)',
            'macro_f1': round(f1_score(y_test, pred_co, average='macro', zero_division=0), 4),
            'immediate_recall': round(recall_score(y_test, pred_co, average=None, zero_division=0)[0], 4),
            'urgent_recall': round(recall_score(y_test, pred_co, average=None, zero_division=0)[1], 4),
            'under_triage_rate': round(ut_co['overall_undertriage_rate'], 4)
        })

        print(f"Budget {int(budget*100):2d}% ({n_labelled:4d} labels) evaluated across all 4 models.")

    df_budget = pd.DataFrame(rows)
    df_budget.to_csv(output_csv, index=False)
    print(f"Label budget study saved to {output_csv}")
    return df_budget


def evaluate_clinical_presentations(test_df, y_pred, output_csv="outputs/reports/presentation_error_analysis.csv"):
    """
    Evaluates model recall specifically across clinical presentations for Immediate patients
    (e.g. stroke vs sepsis vs cardiac arrest vs trauma).
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = test_df.copy()
    df['pred_label'] = y_pred

    # Focus on true Immediate cases
    df_imm = df[df['audit_label'] == 0].copy()
    df_imm['caught_immediate'] = (df_imm['pred_label'] == 0)

    pres_summary = df_imm.groupby('audit_presentation').agg(
        total_cases=('visit_id', 'count'),
        caught_cases=('caught_immediate', 'sum')
    ).reset_index()

    pres_summary['immediate_recall'] = (pres_summary['caught_cases'] / pres_summary['total_cases']).round(4)
    pres_summary = pres_summary.sort_values(by='immediate_recall', ascending=True).reset_index(drop=True)
    pres_summary.to_csv(output_csv, index=False)
    print(f"Presentation error analysis saved to {output_csv}")
    return pres_summary


def run_full_evaluation_pipeline(dataset_bundle,
                                trained_models,
                                label_spreading_results,
                                co_training_results,
                                outputs_dir="outputs"):
    """
    Evaluates all 4 models on the locked test set, runs label budget study,
    presentation analysis, and generates metrics.csv and summary.json.
    """
    reports_dir = os.path.join(outputs_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    y_test = dataset_bundle['y_test_audit']
    X_test_c = dataset_bundle['X_test_combined']

    # Predictions
    preds = {
        'Multinomial Logistic Regression': trained_models['logistic_regression'].predict(X_test_c),
        'Random Forest Classifier': trained_models['random_forest'].predict(X_test_c),
        'Label Spreading (kNN Graph)': label_spreading_results['y_test_pred'],
        'Handwritten Co-Training': co_training_results['y_test_pred'],
        'All-Labelled Upper Bound': trained_models['upper_bound_model'].predict(X_test_c)
    }

    metrics_list = []
    confusion_matrices = {}

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS (Locked Test Set: 1,440 visits)")
    print("=" * 70)

    for m_name, y_pred in preds.items():
        res = evaluate_model_predictions(m_name, y_test, y_pred)
        confusion_matrices[m_name] = res.pop('confusion_matrix')
        metrics_list.append(res)
        print(f"\nModel: {m_name}")
        print(f"  Accuracy: {res['Accuracy']:.4f} | Macro-F1: {res['Macro_F1']:.4f}")
        print(f"  Immediate Recall: {res['Immediate_Recall']:.4f} | Urgent Recall: {res['Urgent_Recall']:.4f}")
        print(f"  Under-Triage Rate (Overall): {res['Overall_Under_Triage_Rate']*100:.2f}% | Immediate UT: {res['Immediate_Under_Triage_Rate']*100:.2f}%")

    df_metrics = pd.DataFrame(metrics_list)

    # Save metrics.csv in outputs/reports/ and in outputs/
    csv_path_reports = os.path.join(reports_dir, "metrics.csv")
    csv_path_root = os.path.join(outputs_dir, "metrics.csv")
    df_metrics.to_csv(csv_path_reports, index=False)
    df_metrics.to_csv(csv_path_root, index=False)
    print(f"\nMetrics saved to {csv_path_reports} and {csv_path_root}")

    # Label Budget Study
    budget_csv = os.path.join(reports_dir, "label_budget_curve.csv")
    df_budget = run_label_budget_study(dataset_bundle, output_csv=budget_csv)

    # Clinical presentation breakdown for selected model (Random Forest)
    pres_csv = os.path.join(reports_dir, "presentation_error_analysis.csv")
    df_pres = evaluate_clinical_presentations(dataset_bundle['test_df'], preds['Random Forest Classifier'], output_csv=pres_csv)

    # Build summary.json
    selected_model_name = "Random Forest Classifier"
    selected_metrics = df_metrics[df_metrics['Model'] == selected_model_name].iloc[0].to_dict()

    summary_data = {
        'project_name': 'Patient Triage from Partially Labelled Records',
        'track': 'G - Semi-Supervised Learning',
        'domain': 'Healthcare - Emergency Department',
        'seed': 444,
        'dataset': {
            'total_raw_rows': 7286,
            'unique_visits': 7200,
            'senior_nurse_labelled_visits': int(dataset_bundle['train_labelled_mask'].sum()),
            'locked_test_visits': len(y_test)
        },
        'selected_model': selected_model_name,
        'selected_model_metrics': selected_metrics,
        'model_comparison': df_metrics.to_dict(orient='records'),
        'confusion_matrices': confusion_matrices
    }

    summary_json_path = os.path.join(reports_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary JSON saved to {summary_json_path}")

    return {
        'metrics_df': df_metrics,
        'budget_df': df_budget,
        'presentation_df': df_pres,
        'confusion_matrices': confusion_matrices,
        'predictions': preds
    }


if __name__ == "__main__":
    from feature_engineering import prepare_datasets_and_views
    from supervised_models import train_supervised_models
    from label_spreading import train_label_spreading
    from co_training import run_co_training_pipeline

    bundle = prepare_datasets_and_views()
    sup_models = train_supervised_models(bundle)
    lp_res = train_label_spreading(bundle)
    co_res = run_co_training_pipeline(bundle)
    eval_res = run_full_evaluation_pipeline(bundle, sup_models, lp_res, co_res)
