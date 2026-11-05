"""
Unit Test Suite for Emergency Department Patient Triage ML System.
Tests data cleaning functions, view generation, under-triage calculation,
and model training & prediction.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_cleaning import (
    parse_blood_pressure,
    convert_fahrenheit_to_celsius,
    sanitize_spo2,
    sanitize_age,
    normalize_categorical,
    parse_arrival_time_and_extract_hour,
    process_comorbidities,
    ARRIVAL_MODE_MAP,
    BINARY_YES_NO_MAP,
    SENIOR_TRIAGE_MAP
)
from evaluation import compute_undertriage_metrics
from co_training import HandwrittenCoTrainingClassifier


def test_parse_blood_pressure():
    bp_series = pd.Series(['120/80', '140/90', 'NR', '--', '', None, '240/130', '500/20'])
    sys, dia = parse_blood_pressure(bp_series)
    assert sys.iloc[0] == 120.0 and dia.iloc[0] == 80.0
    assert sys.iloc[1] == 140.0 and dia.iloc[1] == 90.0
    assert pd.isna(sys.iloc[2]) and pd.isna(dia.iloc[2])
    assert pd.isna(sys.iloc[3]) and pd.isna(dia.iloc[3])
    assert pd.isna(sys.iloc[4]) and pd.isna(dia.iloc[4])
    assert pd.isna(sys.iloc[5]) and pd.isna(dia.iloc[5])
    assert sys.iloc[6] == 240.0 and dia.iloc[6] == 130.0
    assert pd.isna(sys.iloc[7])  # 500 mmHg is physiologically impossible


def test_convert_fahrenheit_to_celsius():
    temp_series = pd.Series([36.6, 38.5, 98.6, 102.2, np.nan, 20.0, 150.0])
    celsius = convert_fahrenheit_to_celsius(temp_series)
    assert celsius.iloc[0] == 36.6
    assert celsius.iloc[1] == 38.5
    assert celsius.iloc[2] == 37.0  # (98.6 - 32) * 5/9 = 37.0
    assert celsius.iloc[3] == 39.0  # (102.2 - 32) * 5/9 = 39.0
    assert pd.isna(celsius.iloc[4])
    assert pd.isna(celsius.iloc[5])  # 20 C is impossible
    assert pd.isna(celsius.iloc[6])  # 150 F is 65 C, impossible


def test_sanitize_spo2():
    spo2_series = pd.Series([98.0, 85.0, 0.0, 101.0, 110.0, -5.0, np.nan])
    clean_spo2 = sanitize_spo2(spo2_series)
    assert clean_spo2.iloc[0] == 98.0
    assert clean_spo2.iloc[1] == 85.0
    assert pd.isna(clean_spo2.iloc[2])
    assert pd.isna(clean_spo2.iloc[3])
    assert pd.isna(clean_spo2.iloc[4])
    assert pd.isna(clean_spo2.iloc[5])
    assert pd.isna(clean_spo2.iloc[6])


def test_sanitize_age():
    age_series = pd.Series([45.0, 82.0, 0.0, 150.0, 999.0, -10.0, np.nan])
    clean_age = sanitize_age(age_series)
    assert clean_age.iloc[0] == 45.0
    assert clean_age.iloc[1] == 82.0
    assert pd.isna(clean_age.iloc[2])
    assert pd.isna(clean_age.iloc[3])
    assert pd.isna(clean_age.iloc[4])
    assert pd.isna(clean_age.iloc[5])
    assert pd.isna(clean_age.iloc[6])


def test_normalize_categorical():
    # Arrival mode
    arr = pd.Series(['108 Ambulance', 'AMBULANCE', 'walk in', 'WALK-IN', 'Referral', 'mlc/police', 'Unknown'])
    norm_arr = normalize_categorical(arr, ARRIVAL_MODE_MAP, default='Walk-in')
    assert norm_arr.iloc[0] == 'Ambulance'
    assert norm_arr.iloc[1] == 'Ambulance'
    assert norm_arr.iloc[2] == 'Walk-in'
    assert norm_arr.iloc[3] == 'Walk-in'
    assert norm_arr.iloc[4] == 'Referred'
    assert norm_arr.iloc[5] == 'Police'
    assert norm_arr.iloc[6] == 'Walk-in'

    # Binary Yes/No
    yn = pd.Series(['YES', 'y', 'Yes', 'no', 'N', 'NO'])
    norm_yn = normalize_categorical(yn, BINARY_YES_NO_MAP, default='No')
    assert (norm_yn == ['Yes', 'Yes', 'Yes', 'No', 'No', 'No']).all()

    # Senior triage
    tri = pd.Series(['RED', '1 - Immediate', 'orange', '3 - Standard', 'green', '4 - Non-Urgent'])
    norm_tri = normalize_categorical(tri, SENIOR_TRIAGE_MAP, default=np.nan)
    assert (norm_tri == ['Immediate', 'Immediate', 'Urgent', 'Standard', 'Non-Urgent', 'Non-Urgent']).all()


def test_parse_arrival_time():
    times = pd.Series(['2025-03-15 14:30', '15-03-2025 08:15', '2025-11-04 23:59', 'invalid_date'])
    dt, hour = parse_arrival_time_and_extract_hour(times)
    assert hour.iloc[0] == 14
    assert hour.iloc[1] == 8
    assert hour.iloc[2] == 23
    assert pd.isna(hour.iloc[3])


def test_process_comorbidities():
    comorb = pd.Series(['Diabetes;Hypertension', 'COPD', 'None', '', None, 'CKD;Asthma;Heart disease'])
    clean_str, counts = process_comorbidities(comorb)
    assert counts.iloc[0] == 2
    assert counts.iloc[1] == 1
    assert counts.iloc[2] == 0 and clean_str.iloc[2] == 'None'
    assert counts.iloc[3] == 0 and clean_str.iloc[3] == 'None'
    assert counts.iloc[4] == 0 and clean_str.iloc[4] == 'None'
    assert counts.iloc[5] == 3


def test_compute_undertriage_metrics():
    # True: 0 (Immediate), Pred: 1 (Urgent) -> Under-triage!
    # True: 1 (Urgent), Pred: 0 (Immediate) -> Over-triage!
    # True: 2 (Standard), Pred: 2 (Standard) -> Correct!
    # True: 0 (Immediate), Pred: 0 (Immediate) -> Correct!
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([1, 0, 2, 0])

    m = compute_undertriage_metrics(y_true, y_pred)
    assert m['overall_undertriage_rate'] == 0.25  # 1 out of 4 is undertriaged (patient 0: 0 -> 1)
    assert m['overall_overtriage_rate'] == 0.25   # 1 out of 4 is overtriaged (patient 1: 1 -> 0)
    assert m['immediate_recall'] == 0.50          # 1 of 2 immediate caught
    assert m['immediate_undertriage_rate'] == 0.50


def test_co_training_fit_and_predict():
    np.random.seed(42)
    n = 200
    # Synthetic 2-view data
    X_a = np.random.randn(n, 5)
    X_b = np.random.randn(n, 6)
    y_true = np.random.choice([0, 1, 2, 3], size=n)

    # 20% labelled, 80% unlabelled (-1)
    y_pool = np.full(n, -1)
    labelled_idx = np.random.choice(n, size=40, replace=False)
    y_pool[labelled_idx] = y_true[labelled_idx]

    co = HandwrittenCoTrainingClassifier(confidence_threshold=0.6, max_rounds=3, random_state=42)
    co.fit(X_a, X_b, y_pool, y_true_audit_hidden=y_true)

    X_test_c = np.hstack([X_a[:20], X_b[:20]])
    preds = co.predict(X_test_c)
    assert len(preds) == 20
    assert set(preds).issubset({0, 1, 2, 3})

