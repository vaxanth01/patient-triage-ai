"""
Visualizations Module - 12 Publication-Grade High-Resolution Charts.
Generates all 12 required figures saved to outputs/figures/ with dpi=300:
1.  01_label_scarcity_mix.png
2.  02_vitals_by_triage.png
3.  03_history_symptoms_by_triage.png
4.  04_ed_arrivals_by_hour.png
5.  05_feature_correlation_two_views.png
6.  06_view_strength_comparison.png
7.  07_co_training_progress.png
8.  08_model_comparison_test_set.png
9.  09_confusion_matrix_selected_model.png
10. 10_recall_and_undertriage_rate.png
11. 11_label_budget_curves.png
12. 12_immediate_recall_by_presentation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

CLASS_COLORS = {
    'Immediate': '#d9534f',   # Red
    'Urgent': '#f0ad4e',      # Orange
    'Standard': '#5bc0de',    # Blue/Yellow
    'Non-Urgent': '#5cb85c'   # Green
}

ORDERED_CLASSES = ['Immediate', 'Urgent', 'Standard', 'Non-Urgent']


def save_fig(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved chart: {filepath}")


def plot_01_label_scarcity(df_clean, output_path="outputs/figures/01_label_scarcity_mix.png"):
    """Chart 1: Label scarcity and senior nurse triage mix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    total = len(df_clean)
    labelled_count = df_clean['senior_nurse_triage'].dropna().count()
    unlabelled_count = total - labelled_count

    # Pie/Donut of label scarcity
    axes[0].pie(
        [labelled_count, unlabelled_count],
        labels=[f'Senior Nurse Labelled\n({labelled_count:,} visits, {labelled_count/total*100:.1f}%)',
                f'Unlabelled Training Pool\n({unlabelled_count:,} visits, {unlabelled_count/total*100:.1f}%)'],
        colors=['#2b5c8f', '#e0e0e0'],
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    axes[0].set_title("A. Label Scarcity in 1-Year ED Registry\n(Total Visits = 7,200)", fontsize=13, weight='bold', pad=15)

    # Bar chart of Senior Nurse triage mix
    sr_counts = df_clean['senior_nurse_triage'].value_counts().reindex(ORDERED_CLASSES).fillna(0)
    bars = axes[1].bar(
        ORDERED_CLASSES,
        sr_counts.values,
        color=[CLASS_COLORS[c] for c in ORDERED_CLASSES],
        edgecolor='#333333',
        linewidth=1.2,
        alpha=0.88
    )
    axes[1].set_title(f"B. Senior Nurse Triage Category Mix\n(Labelled Subset = {labelled_count} visits, 59 Immediate)", fontsize=13, weight='bold', pad=15)
    axes[1].set_ylabel("Number of Visits", fontsize=11, weight='bold')
    axes[1].set_xlabel("Triage Level", fontsize=11, weight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        pct = h / labelled_count * 100
        axes[1].annotate(f"{int(h)}\n({pct:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_02_vitals_by_triage(df_merged, output_path="outputs/figures/02_vitals_by_triage.png"):
    """Chart 2: Vitals by triage level (Heart Rate, SpO2, Respiratory Rate, Systolic BP)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vitals_map = [
        ('heart_rate', 'Heart Rate (bpm)', axes[0, 0], 'Heart Rate by Triage Level (Immediate: Tachycardia)'),
        ('spo2_pct', 'Oxygen Saturation SpO2 (%)', axes[0, 1], 'SpO2 (%) by Triage Level (Immediate: Severe Hypoxia)'),
        ('resp_rate', 'Respiratory Rate (breaths/min)', axes[1, 0], 'Respiratory Rate by Triage Level (Immediate: Tachypnea)'),
        ('systolic_bp', 'Systolic Blood Pressure (mmHg)', axes[1, 1], 'Systolic BP by Triage Level (Immediate: Bimodal Shock vs Stroke)')
    ]

    for col, label, ax, title in vitals_map:
        sns.boxplot(
            data=df_merged,
            x='audit_triage_level',
            y=col,
            hue='audit_triage_level',
            order=ORDERED_CLASSES,
            palette=[CLASS_COLORS[c] for c in ORDERED_CLASSES],
            ax=ax,
            fliersize=2.5,
            linewidth=1.2,
            boxprops=dict(alpha=0.85)
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_title(title, fontsize=11.5, weight='bold')
        ax.set_xlabel("Clinical Audit Triage Level", fontsize=10, weight='bold')
        ax.set_ylabel(label, fontsize=10, weight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_03_history_by_triage(df_merged, output_path="outputs/figures/03_history_symptoms_by_triage.png"):
    """Chart 3: History and symptoms by triage level."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # A: % Ambulance arrivals
    amb_pct = df_merged.groupby('audit_triage_level')['arrival_mode'].apply(lambda s: (s == 'Ambulance').mean() * 100).reindex(ORDERED_CLASSES)
    bars_amb = axes[0, 0].bar(ORDERED_CLASSES, amb_pct.values, color=[CLASS_COLORS[c] for c in ORDERED_CLASSES], edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("A. Ambulance Arrival Share by Triage Level\n(Immediate: 71% arrive via Ambulance)", fontsize=11.5, weight='bold')
    axes[0, 0].set_ylabel("% Arriving by Ambulance", fontsize=10, weight='bold')
    axes[0, 0].set_ylim(0, 85)
    for b in bars_amb:
        axes[0, 0].annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', weight='bold')

    # B: % Chest Pain
    cp_pct = df_merged.groupby('audit_triage_level')['chest_pain'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(ORDERED_CLASSES)
    bars_cp = axes[0, 1].bar(ORDERED_CLASSES, cp_pct.values, color=[CLASS_COLORS[c] for c in ORDERED_CLASSES], edgecolor='black', alpha=0.85)
    axes[0, 1].set_title("B. Chest Pain Presentation Rate by Triage Level", fontsize=11.5, weight='bold')
    axes[0, 1].set_ylabel("% Reporting Chest Pain", fontsize=10, weight='bold')
    axes[0, 1].set_ylim(0, 80)
    for b in bars_cp:
        axes[0, 1].annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', weight='bold')

    # C: % Breathlessness
    br_pct = df_merged.groupby('audit_triage_level')['breathlessness'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(ORDERED_CLASSES)
    bars_br = axes[1, 0].bar(ORDERED_CLASSES, br_pct.values, color=[CLASS_COLORS[c] for c in ORDERED_CLASSES], edgecolor='black', alpha=0.85)
    axes[1, 0].set_title("C. Breathlessness Presentation Rate by Triage Level", fontsize=11.5, weight='bold')
    axes[1, 0].set_ylabel("% Reporting Breathlessness", fontsize=10, weight='bold')
    axes[1, 0].set_ylim(0, 85)
    for b in bars_br:
        axes[1, 0].annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', weight='bold')

    # D: Average Pain Score & Comorbidity Count
    sns.barplot(
        data=df_merged,
        x='audit_triage_level',
        y='pain_score',
        hue='audit_triage_level',
        order=ORDERED_CLASSES,
        palette=[CLASS_COLORS[c] for c in ORDERED_CLASSES],
        ax=axes[1, 1],
        edgecolor='black',
        alpha=0.85
    )
    if axes[1, 1].get_legend():
        axes[1, 1].get_legend().remove()
    axes[1, 1].set_title("D. Mean Pain Score (0-10) by Triage Level", fontsize=11.5, weight='bold')
    axes[1, 1].set_ylabel("Mean Pain Score", fontsize=10, weight='bold')
    axes[1, 1].set_ylim(0, 10)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_04_arrivals_by_hour(df_clean, output_path="outputs/figures/04_ed_arrivals_by_hour.png"):
    """Chart 4: ED arrivals by hour of day (24-hour round-the-clock volume)."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    hourly_counts = df_clean['hour_of_day'].value_counts().sort_index()
    hours = np.arange(24)
    counts = [hourly_counts.get(h, 0) for h in hours]

    bars = ax.bar(hours, counts, color='#2c7bb6', edgecolor='#1a476f', alpha=0.85, width=0.75)

    # Highlight night shifts (20:00 to 07:00)
    for h in hours:
        if h >= 20 or h <= 7:
            bars[h].set_color('#d7191c')
            bars[h].set_alpha(0.75)

    ax.set_title("Emergency Department Patient Arrivals by Hour of Day\n(Red: Night Shift 20:00 - 07:00 | Blue: Day Shift 08:00 - 19:00)", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Hour of Arrival (24-Hour Clock)", fontsize=11, weight='bold')
    ax.set_ylabel("Total Patient Visits in Year", fontsize=11, weight='bold')
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Average line
    mean_count = np.mean(counts)
    ax.axhline(mean_count, color='#333333', linestyle=':', linewidth=1.5, label=f'Mean Arrivals/Hour ({mean_count:.1f})')
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_05_feature_correlation(dataset_bundle, output_path="outputs/figures/05_feature_correlation_two_views.png"):
    """Chart 5: Feature correlation across View 1 (Vitals) and View 2 (History)."""
    fig, ax = plt.subplots(figsize=(13, 11))

    X_train_c = dataset_bundle['X_train_combined']
    col_names = dataset_bundle['combined_names']
    df_feat = pd.DataFrame(X_train_c, columns=col_names)

    corr = df_feat.corr()

    # Draw heatmap
    sns.heatmap(
        corr,
        cmap='vlag',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=True,
        fmt=".2f",
        annot_kws={'size': 7.5},
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )

    ax.set_title("Cross-View Feature Correlation Matrix\n(View 1: First 7 Vitals vs View 2: History & Symptoms)", fontsize=13, weight='bold', pad=15)

    # Highlight boundary between views
    ax.axvline(7, color='black', linewidth=2.5, linestyle='--')
    ax.axhline(7, color='black', linewidth=2.5, linestyle='--')

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_06_view_strengths(cv_study_results, output_path="outputs/figures/06_view_strength_comparison.png"):
    """Chart 6: How strong each view is alone vs combined."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    views = ['View 1: Vitals Alone\n(7 Features)', 'View 2: History Alone\n(10 Features)', 'Combined Views\n(17 Features)']
    f1_means = [
        cv_study_results['vitals_cv_macro_f1_mean'],
        cv_study_results['history_cv_macro_f1_mean'],
        cv_study_results['combined_cv_macro_f1_mean']
    ]
    f1_stds = [
        cv_study_results['vitals_cv_macro_f1_std'],
        cv_study_results['history_cv_macro_f1_std'],
        cv_study_results['combined_cv_macro_f1_std']
    ]

    colors = ['#4575b4', '#74add1', '#313695']
    bars = ax.bar(views, f1_means, yerr=f1_stds, capsize=6, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9, width=0.55)

    ax.set_title("Cross-Validation Macro-F1: View 1 (Vitals) vs View 2 (History) vs Combined\n(Stratified 5-Fold CV on Senior-Nurse Labelled Visits)", fontsize=12, weight='bold', pad=15)
    ax.set_ylabel("Macro-F1 Score", fontsize=11, weight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for b, mean, std in zip(bars, f1_means, f1_stds):
        ax.annotate(f"Macro-F1 = {mean:.3f}\n(±{std:.3f})",
                    xy=(b.get_x() + b.get_width() / 2, mean),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_07_co_training_progress(df_progress, output_path="outputs/figures/07_co_training_progress.png"):
    """Chart 7: Handwritten Co-Training progress across iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    rounds = df_progress['round']

    # A: Accepted pseudo-labels per round
    axes[0].bar(rounds, df_progress['newly_pseudo_labelled'], color='#2b5c8f', edgecolor='black', alpha=0.85, width=0.6)
    axes[0].plot(rounds, df_progress['newly_pseudo_labelled'], color='#d9534f', marker='o', linewidth=2)
    axes[0].set_title("A. High-Confidence Pseudo-Labels Accepted per Round", fontsize=12, weight='bold')
    axes[0].set_xlabel("Co-Training Iteration Round", fontsize=10, weight='bold')
    axes[0].set_ylabel("Number of Added Visits", fontsize=10, weight='bold')
    axes[0].set_xticks(rounds)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    for _, row in df_progress.iterrows():
        axes[0].annotate(f"{int(row['newly_pseudo_labelled'])}",
                         xy=(row['round'], row['newly_pseudo_labelled']),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, weight='bold')

    # B: Agreement rate & Pseudo-label accuracy
    axes[1].plot(rounds, df_progress['view_agreement_rate'] * 100, marker='s', color='#2ca02c', linewidth=2.2, label='View Agreement Rate (%)')
    if 'pseudo_label_accuracy' in df_progress.columns and not df_progress['pseudo_label_accuracy'].isna().all():
        axes[1].plot(rounds, df_progress['pseudo_label_accuracy'] * 100, marker='^', color='#ff7f0e', linewidth=2.2, label='Pseudo-Label Accuracy vs Audit (%)')

    axes[1].set_title("B. Inter-View Agreement Rate & Pseudo-Label Accuracy", fontsize=12, weight='bold')
    axes[1].set_xlabel("Co-Training Iteration Round", fontsize=10, weight='bold')
    axes[1].set_ylabel("Percentage (%)", fontsize=10, weight='bold')
    axes[1].set_ylim(60, 102)
    axes[1].set_xticks(rounds)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower left', frameon=True)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_08_model_comparison(df_metrics, output_path="outputs/figures/08_model_comparison_test_set.png"):
    """Chart 8: Comprehensive test set performance comparison across models."""
    fig, ax = plt.subplots(figsize=(13, 6))

    models = df_metrics['Model'].tolist()
    metrics_to_plot = ['Accuracy', 'Macro_F1', 'Immediate_Recall', 'Urgent_Recall']
    x = np.arange(len(models))
    width = 0.18

    colors = ['#1f77b4', '#aec7e8', '#d62728', '#ff7f0e']

    for i, m in enumerate(metrics_to_plot):
        vals = df_metrics[m].values
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=m.replace('_', ' '), color=colors[i], edgecolor='black', linewidth=0.8, alpha=0.9)

    ax.set_title("Model Performance Comparison on Locked Test Set (1,440 Visits)\n(10% Senior Nurse Labels vs Semi-Supervised Methods vs Upper Bound)", fontsize=13, weight='bold', pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, weight='bold', rotation=15, ha='right')
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_09_confusion_matrix(cm, model_name="Random Forest Classifier", output_path="outputs/figures/09_confusion_matrix_selected_model.png"):
    """Chart 9: Confusion matrix of the selected model."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    cm_arr = np.array(cm)
    cm_pct = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis] * 100

    annot_matrix = np.empty_like(cm_arr, dtype=object)
    for i in range(4):
        for j in range(4):
            annot_matrix[i, j] = f"{cm_arr[i, j]}\n({cm_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm_pct,
        annot=annot_matrix,
        fmt="",
        cmap='Blues',
        xticklabels=ORDERED_CLASSES,
        yticklabels=ORDERED_CLASSES,
        cbar_kws={'label': 'Recall (% of Actual Cases)'},
        ax=ax,
        linewidths=1.2,
        linecolor='white'
    )

    ax.set_title(f"Confusion Matrix: {model_name}\n(Actual Audit Level vs Model Predicted Level)", fontsize=12, weight='bold', pad=15)
    ax.set_xlabel("Predicted Triage Category", fontsize=11, weight='bold')
    ax.set_ylabel("Actual Audit Triage Category (Reference)", fontsize=11, weight='bold')

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_10_recall_and_undertriage(df_metrics, output_path="outputs/figures/10_recall_and_undertriage_rate.png"):
    """Chart 10: Recall by triage level and under-triage rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    models = df_metrics['Model'].tolist()
    x = np.arange(len(models))

    # A: Recall for Immediate vs Urgent
    axes[0].plot(x, df_metrics['Immediate_Recall'] * 100, marker='o', linewidth=2.5, color='#d9534f', label='Immediate Recall (%)')
    axes[0].plot(x, df_metrics['Urgent_Recall'] * 100, marker='s', linewidth=2.5, color='#f0ad4e', label='Urgent Recall (%)')
    axes[0].set_title("A. High-Acuity Patient Recall by Model", fontsize=12, weight='bold')
    axes[0].set_ylabel("Recall Rate (%)", fontsize=10, weight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha='right', fontsize=9, weight='bold')
    axes[0].set_ylim(60, 100)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower left', frameon=True)

    # B: Under-Triage Rates (Overall & Immediate)
    axes[1].bar(x - 0.15, df_metrics['Overall_Under_Triage_Rate'] * 100, width=0.3, label='Overall Under-Triage Rate (%)', color='#337ab7', edgecolor='black', alpha=0.85)
    axes[1].bar(x + 0.15, df_metrics['Immediate_Under_Triage_Rate'] * 100, width=0.3, label='Immediate Under-Triage Rate (%)', color='#d9534f', edgecolor='black', alpha=0.85)
    axes[1].set_title("B. Patient Under-Triage Safety Failure Rate\n(Lower is Safer)", fontsize=12, weight='bold')
    axes[1].set_ylabel("Under-Triage Rate (%)", fontsize=10, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha='right', fontsize=9, weight='bold')
    axes[1].set_ylim(0, 30)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_11_label_budget_curves(df_budget, output_path="outputs/figures/11_label_budget_curves.png"):
    """Chart 11: Performance vs label budget curves (2% to 40% labelled training data)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    models = df_budget['model'].unique()
    palette = {
        'Logistic Regression (Supervised)': '#1f77b4',
        'Random Forest (Supervised)': '#2ca02c',
        'Label Spreading (Semi-Supervised)': '#d62728',
        'Co-Training (Semi-Supervised)': '#9467bd'
    }

    # Subplot 1: Macro-F1 vs Budget
    for m in models:
        sub = df_budget[df_budget['model'] == m]
        axes[0].plot(sub['label_budget_pct'], sub['macro_f1'], marker='o', linewidth=2.2, label=m, color=palette.get(m, '#333'))
    axes[0].set_title("A. Test Macro-F1 vs Label Budget", fontsize=12, weight='bold')
    axes[0].set_xlabel("Labelled Training Pool Proportion (%)", fontsize=10, weight='bold')
    axes[0].set_ylabel("Macro-F1 Score", fontsize=10, weight='bold')
    axes[0].set_ylim(0.70, 0.94)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', fontsize=8.5, frameon=True)

    # Subplot 2: Immediate Recall vs Budget
    for m in models:
        sub = df_budget[df_budget['model'] == m]
        axes[1].plot(sub['label_budget_pct'], sub['immediate_recall'], marker='s', linewidth=2.2, label=m, color=palette.get(m, '#333'))
    axes[1].set_title("B. Immediate Recall vs Label Budget", fontsize=12, weight='bold')
    axes[1].set_xlabel("Labelled Training Pool Proportion (%)", fontsize=10, weight='bold')
    axes[1].set_ylabel("Immediate Patient Recall", fontsize=10, weight='bold')
    axes[1].set_ylim(0.65, 0.96)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: Under-Triage Rate vs Budget
    for m in models:
        sub = df_budget[df_budget['model'] == m]
        axes[2].plot(sub['label_budget_pct'], sub['under_triage_rate'] * 100, marker='^', linewidth=2.2, label=m, color=palette.get(m, '#333'))
    axes[2].set_title("C. Under-Triage Rate vs Label Budget\n(Safety Failure Rate - Lower is Better)", fontsize=12, weight='bold')
    axes[2].set_xlabel("Labelled Training Pool Proportion (%)", fontsize=10, weight='bold')
    axes[2].set_ylabel("Under-Triage Rate (%)", fontsize=10, weight='bold')
    axes[2].set_ylim(5, 25)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_12_presentation_recall(df_pres, output_path="outputs/figures/12_immediate_recall_by_presentation.png"):
    """Chart 12: Immediate recall by clinical presentation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df_pres.sort_values(by='immediate_recall', ascending=True)
    y_pos = np.arange(len(df_sorted))

    # Color code bars: green if recall >= 0.85, orange if 0.70-0.85, red if < 0.70
    colors = []
    for r in df_sorted['immediate_recall']:
        if r >= 0.85:
            colors.append('#5cb85c')
        elif r >= 0.70:
            colors.append('#f0ad4e')
        else:
            colors.append('#d9534f')

    bars = ax.barh(y_pos, df_sorted['immediate_recall'] * 100, color=colors, edgecolor='black', alpha=0.88, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['audit_presentation'], fontsize=10, weight='bold')
    ax.set_xlabel("Immediate Triage Recall Rate (%)", fontsize=11, weight='bold')
    ax.set_xlim(0, 115)
    ax.set_title("Immediate Patient Recall by Clinical Presentation (Selected Random Forest Model)\n(Strokes Always Caught 100%, Sepsis Frequently Under-Triaged to Urgent)", fontsize=12, weight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        w = bar.get_width()
        total = int(row['total_cases'])
        caught = int(row['caught_cases'])
        ax.annotate(f"{w:.1f}% ({caught}/{total})",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, weight='bold')

    plt.tight_layout()
    save_fig(fig, output_path)


def generate_all_12_visualizations(df_clean,
                                   df_merged,
                                   dataset_bundle,
                                   cv_study_results,
                                   df_progress,
                                   df_metrics,
                                   confusion_matrices,
                                   df_budget,
                                   df_pres,
                                   output_dir="outputs/figures"):
    """
    Executes all plotting routines and saves all 12 required figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print("GENERATING ALL 12 REQUIRED CHARTS (300 DPI PNG)")
    print("=" * 65)

    plot_01_label_scarcity(df_clean, os.path.join(output_dir, "01_label_scarcity_mix.png"))
    plot_02_vitals_by_triage(df_merged, os.path.join(output_dir, "02_vitals_by_triage.png"))
    plot_03_history_by_triage(df_merged, os.path.join(output_dir, "03_history_symptoms_by_triage.png"))
    plot_04_arrivals_by_hour(df_clean, os.path.join(output_dir, "04_ed_arrivals_by_hour.png"))
    plot_05_feature_correlation(dataset_bundle, os.path.join(output_dir, "05_feature_correlation_two_views.png"))
    plot_06_view_strengths(cv_study_results, os.path.join(output_dir, "06_view_strength_comparison.png"))
    plot_07_co_training_progress(df_progress, os.path.join(output_dir, "07_co_training_progress.png"))
    plot_08_model_comparison(df_metrics, os.path.join(output_dir, "08_model_comparison_test_set.png"))

    selected_cm = confusion_matrices.get('Random Forest Classifier')
    plot_09_confusion_matrix(selected_cm, 'Random Forest Classifier', os.path.join(output_dir, "09_confusion_matrix_selected_model.png"))

    plot_10_recall_and_undertriage(df_metrics, os.path.join(output_dir, "10_recall_and_undertriage_rate.png"))
    plot_11_label_budget_curves(df_budget, os.path.join(output_dir, "11_label_budget_curves.png"))
    plot_12_presentation_recall(df_pres, os.path.join(output_dir, "12_immediate_recall_by_presentation.png"))

    print(f"\nAll 12 high-resolution figures successfully saved to {output_dir}")
