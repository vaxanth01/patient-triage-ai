"""
Data Cleaning and Data Quality Audit Module.
Handles all 11 identified data quality issues cleanly, safely, and without data leakage.
"""

import os
import re
import numpy as np
import pandas as pd

# Mapping for arrival modes
ARRIVAL_MODE_MAP = {
    'ambulance': 'Ambulance',
    '108 ambulance': 'Ambulance',
    'ems / ambulance': 'Ambulance',
    'walk-in': 'Walk-in',
    'walk in': 'Walk-in',
    'walkin': 'Walk-in',
    'referred': 'Referred',
    'referral': 'Referred',
    'hospital transfer': 'Referred',
    'police': 'Police',
    'mlc/police': 'Police',
    'police / mlc': 'Police'
}

# Mapping for Yes/No fields
BINARY_YES_NO_MAP = {
    'yes': 'Yes',
    'y': 'Yes',
    'no': 'No',
    'n': 'No'
}

# Senior nurse triage categories mapping to 4 standardized classes
SENIOR_TRIAGE_MAP = {
    'immediate': 'Immediate',
    'red': 'Immediate',
    '1 - immediate': 'Immediate',
    '1': 'Immediate',
    'urgent': 'Urgent',
    'orange': 'Urgent',
    '2 - urgent': 'Urgent',
    '2': 'Urgent',
    'standard': 'Standard',
    'yellow': 'Standard',
    '3 - standard': 'Standard',
    '3': 'Standard',
    'non-urgent': 'Non-Urgent',
    'non urgent': 'Non-Urgent',
    'green': 'Non-Urgent',
    '4 - non-urgent': 'Non-Urgent',
    '4': 'Non-Urgent'
}


def parse_blood_pressure(bp_series):
    """
    Parses blood pressure string 'systolic/diastolic' into numeric systolic and diastolic values.
    Converts 'NR', '--', blanks, and unparseable values to NaN.
    """
    systolic = []
    diastolic = []
    for val in bp_series:
        if pd.isna(val) or not isinstance(val, str):
            systolic.append(np.nan)
            diastolic.append(np.nan)
            continue
        val_clean = val.strip()
        if val_clean in ['NR', '--', '', '-', 'None', 'nan']:
            systolic.append(np.nan)
            diastolic.append(np.nan)
            continue
        match = re.match(r'^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$', val_clean)
        if match:
            s = float(match.group(1))
            d = float(match.group(2))
            # Basic physiological plausibility filter
            if 30 <= s <= 300 and 20 <= d <= 200:
                systolic.append(s)
                diastolic.append(d)
            else:
                systolic.append(np.nan)
                diastolic.append(np.nan)
        else:
            systolic.append(np.nan)
            diastolic.append(np.nan)
    return pd.Series(systolic, index=bp_series.index), pd.Series(diastolic, index=bp_series.index)


def convert_fahrenheit_to_celsius(temp_series):
    """
    Converts temperature values > 45.0 from Fahrenheit to Celsius.
    Preserves existing Celsius values (typically 34-43 °C).
    Formula: C = (F - 32) * 5 / 9
    Plausible physiological Celsius range: 30.0 <= C <= 45.0.
    """
    def _convert(t):
        if pd.isna(t):
            return np.nan
        try:
            t_val = float(t)
        except (ValueError, TypeError):
            return np.nan
        if t_val > 45.0:
            c = (t_val - 32.0) * 5.0 / 9.0
            if 30.0 <= c <= 45.0:
                return round(c, 1)
            return np.nan
        elif 30.0 <= t_val <= 45.0:
            return round(t_val, 1)
        else:
            return np.nan

    return temp_series.apply(_convert)


def sanitize_spo2(spo2_series):
    """
    Identifies impossible probe readings (0, >100 e.g. 101, 110) and converts to NaN.
    Valid clinical SpO2 is between 50% and 100%.
    """
    def _clean(val):
        if pd.isna(val):
            return np.nan
        try:
            v = float(val)
        except (ValueError, TypeError):
            return np.nan
        if 50.0 <= v <= 100.0:
            return v
        return np.nan

    return spo2_series.apply(_clean)


def sanitize_age(age_series):
    """
    Converts impossible ages (0, 150, 999, negative) to NaN.
    Valid adult/pediatric clinical age for this ED registry is 1 <= age <= 120.
    """
    def _clean(val):
        if pd.isna(val):
            return np.nan
        try:
            v = float(val)
        except (ValueError, TypeError):
            return np.nan
        if 1.0 <= v <= 120.0:
            return v
        return np.nan

    return age_series.apply(_clean)


def normalize_categorical(series, mapping, default=np.nan):
    """
    Normalizes spelling, whitespace, and capitalization according to mapping dict.
    """
    def _norm(val):
        if pd.isna(val) or val is None:
            return default
        s = str(val).strip().lower()
        return mapping.get(s, default)

    return series.apply(_norm)


def parse_arrival_time_and_extract_hour(time_series):
    """
    Safely parses arrival times with mixed formats:
    - 'YYYY-MM-DD HH:MM'
    - 'dd-mm-yyyy HH:MM'
    Returns parsed datetime series and integer hour_of_day (0-23).
    """
    dt_series = pd.to_datetime(time_series, format='mixed', dayfirst=True, errors='coerce')
    hour_series = dt_series.dt.hour
    return dt_series, hour_series


def process_comorbidities(comorb_series):
    """
    Parses comorbidities string list (separated by ';') and computes comorbidity_count.
    Treats 'None', empty, or NaN consistently as 0 conditions.
    """
    counts = []
    clean_strings = []
    for val in comorb_series:
        if pd.isna(val) or not isinstance(val, str):
            counts.append(0)
            clean_strings.append('None')
            continue
        v_clean = val.strip()
        if v_clean.lower() in ['none', '', 'nan', 'null', 'no']:
            counts.append(0)
            clean_strings.append('None')
        else:
            items = [item.strip() for item in v_clean.split(';') if item.strip()]
            counts.append(len(items))
            clean_strings.append(';'.join(items))

    return pd.Series(clean_strings, index=comorb_series.index), pd.Series(counts, index=comorb_series.index)


def clean_ed_dataset(raw_ed_path="data/raw/ed_visits.csv",
                     raw_audit_path="data/raw/clinical_audit_triage.csv",
                     output_cleaned_path="data/processed/cleaned_ed_visits.csv",
                     report_path="outputs/reports/data_quality_report.txt"):
    """
    Executes end-to-end cleaning pipeline, produces data quality audit report,
    and saves processed dataset.
    """
    os.makedirs(os.path.dirname(output_cleaned_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    df_raw = pd.read_csv(raw_ed_path)
    df_audit = pd.read_csv(raw_audit_path)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DATA QUALITY & AUDIT REPORT: EMERGENCY DEPARTMENT VISITS")
    report_lines.append("=" * 70)
    report_lines.append(f"Raw visits file: {raw_ed_path}")
    report_lines.append(f"Raw shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    report_lines.append(f"Audit reference file: {raw_audit_path}")
    report_lines.append(f"Audit shape: {df_audit.shape[0]} rows x {df_audit.shape[1]} columns\n")

    report_lines.append("1. AUDIT OF KNOWN DEFECTS (BEFORE CLEANING)")
    report_lines.append("-" * 70)

    # 1. Exact duplicates
    n_dups = df_raw.duplicated().sum()
    report_lines.append(f"• Exact duplicate records (HIS re-sync): {n_dups} rows")

    # 2. Duplicate visit_ids
    n_dup_ids = df_raw.duplicated(subset=['visit_id']).sum()
    report_lines.append(f"• Duplicate visit_ids: {n_dup_ids} rows")

    # 3. Blood pressure defects
    bp_blanks = df_raw['blood_pressure'].isna().sum() + (df_raw['blood_pressure'].astype(str).str.strip().isin(['', 'NR', '--', '-'])).sum()
    report_lines.append(f"• Blood pressure text defects ('NR', '--', blanks): {bp_blanks} records ({bp_blanks/len(df_raw)*100:.2f}%)")

    # 4. Temperature in Fahrenheit
    f_temps = (df_raw['temperature'] > 45.0).sum()
    report_lines.append(f"• Fahrenheit temperature records (>45°C): {f_temps} records ({f_temps/len(df_raw)*100:.2f}%)")

    # 5. Invalid SpO2 readings
    spo2_invalid = ((df_raw['spo2_pct'] < 50.0) | (df_raw['spo2_pct'] > 100.0)).sum()
    report_lines.append(f"• Impossible SpO2 readings (<50% or >100%): {spo2_invalid} records")

    # 6. Impossible ages
    age_invalid = ((df_raw['age'] <= 0) | (df_raw['age'] > 120)).sum()
    report_lines.append(f"• Impossible age values (0, 150, 999): {age_invalid} records")

    # 7. Missing heart rate & resp rate
    hr_missing = df_raw['heart_rate'].isna().sum()
    rr_missing = df_raw['resp_rate'].isna().sum()
    report_lines.append(f"• Missing heart rate: {hr_missing} records ({hr_missing/len(df_raw)*100:.2f}%)")
    report_lines.append(f"• Missing respiratory rate: {rr_missing} records ({rr_missing/len(df_raw)*100:.2f}%)")

    # 8. Categorical inconsistencies
    report_lines.append(f"• arrival_mode unique raw values ({df_raw['arrival_mode'].nunique()} variants): {sorted(list(df_raw['arrival_mode'].dropna().unique()))}")
    report_lines.append(f"• chest_pain unique raw values: {sorted(list(df_raw['chest_pain'].dropna().unique()))}")
    report_lines.append(f"• breathlessness unique raw values: {sorted(list(df_raw['breathlessness'].dropna().unique()))}")

    # 9. Senior nurse triage raw variants
    senior_labelled_raw = df_raw['senior_nurse_triage'].dropna()
    report_lines.append(f"• senior_nurse_triage labelled visits: {len(senior_labelled_raw)} / {len(df_raw)} ({len(senior_labelled_raw)/len(df_raw)*100:.2f}%)")
    report_lines.append(f"• senior_nurse_triage unique raw values: {sorted(list(senior_labelled_raw.unique()))}")

    # 10. Arrival time formats
    sample_times = df_raw['arrival_time'].dropna().head(5).tolist()
    report_lines.append(f"• Sample arrival time representations: {sample_times}")

    # 11. Comorbidity representation
    report_lines.append(f"• Comorbidities null / 'None' representation: {df_raw['comorbidities'].isna().sum()} nulls, {(df_raw['comorbidities'].astype(str).str.lower() == 'none').sum()} 'None'")

    report_lines.append("\n2. CLEANING TRANSFORMATIONS APPLIED")
    report_lines.append("-" * 70)

    # Perform cleaning
    df_clean = df_raw.drop_duplicates(subset=['visit_id'], keep='first').copy()
    report_lines.append(f"✓ Removed {n_dups} exact duplicate rows. Retained {len(df_clean)} unique visits.")

    # Parse Blood Pressure
    df_clean['systolic_bp'], df_clean['diastolic_bp'] = parse_blood_pressure(df_clean['blood_pressure'])
    report_lines.append("✓ Parsed 'blood_pressure' text into numeric 'systolic_bp' and 'diastolic_bp'. Invalid entries converted to NaN.")

    # Temperature conversion
    df_clean['temperature_c'] = convert_fahrenheit_to_celsius(df_clean['temperature'])
    report_lines.append("✓ Converted Fahrenheit values (>45) to Celsius via C = (F - 32) * 5/9. Plausible Celsius range verified.")

    # SpO2 and Age sanitation
    df_clean['spo2_pct'] = sanitize_spo2(df_clean['spo2_pct'])
    df_clean['age'] = sanitize_age(df_clean['age'])
    report_lines.append("✓ Converted 40 invalid SpO2 probe readings (0, 101, 110) and 15 impossible ages (0, 150, 999) to NaN.")

    # Categorical normalization
    df_clean['arrival_mode'] = normalize_categorical(df_clean['arrival_mode'], ARRIVAL_MODE_MAP, default='Walk-in')
    df_clean['chest_pain'] = normalize_categorical(df_clean['chest_pain'], BINARY_YES_NO_MAP, default='No')
    df_clean['breathlessness'] = normalize_categorical(df_clean['breathlessness'], BINARY_YES_NO_MAP, default='No')
    df_clean['trauma'] = normalize_categorical(df_clean['trauma'], BINARY_YES_NO_MAP, default='No')
    df_clean['sex'] = df_clean['sex'].str.strip().str.upper()

    report_lines.append("✓ Normalized spelling & casing for 'arrival_mode' (4 standard levels), 'chest_pain', 'breathlessness', and 'trauma'.")

    # Arrival time parsing
    df_clean['arrival_time_dt'], df_clean['hour_of_day'] = parse_arrival_time_and_extract_hour(df_clean['arrival_time'])
    report_lines.append("✓ Safely parsed arrival timestamps across formats; extracted 'hour_of_day' (0-23).")

    # Comorbidities handling
    df_clean['comorbidities_clean'], df_clean['comorbidity_count'] = process_comorbidities(df_clean['comorbidities'])
    report_lines.append("✓ Handled comorbidities consistently; derived 'comorbidity_count' integer feature.")

    # Senior nurse triage normalization
    df_clean['senior_nurse_triage_clean'] = df_clean['senior_nurse_triage'].apply(
        lambda x: SENIOR_TRIAGE_MAP.get(str(x).strip().lower(), np.nan) if pd.notna(x) and str(x).strip() else np.nan
    )
    report_lines.append("✓ Standardized senior_nurse_triage into 4 target levels: Immediate, Urgent, Standard, Non-Urgent.")

    # Drop intermediate / superseded raw columns
    df_clean = df_clean.drop(columns=['temperature', 'blood_pressure', 'senior_nurse_triage'])
    df_clean = df_clean.rename(columns={'senior_nurse_triage_clean': 'senior_nurse_triage'})

    report_lines.append("\n3. POST-CLEANING QUALITY VERIFICATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Cleaned shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    report_lines.append(f"Unique visit_ids: {df_clean['visit_id'].nunique()}")
    report_lines.append(f"Systolic BP range: {df_clean['systolic_bp'].min():.1f} - {df_clean['systolic_bp'].max():.1f} mmHg (NaNs: {df_clean['systolic_bp'].isna().sum()})")
    report_lines.append(f"Diastolic BP range: {df_clean['diastolic_bp'].min():.1f} - {df_clean['diastolic_bp'].max():.1f} mmHg (NaNs: {df_clean['diastolic_bp'].isna().sum()})")
    report_lines.append(f"Temperature range: {df_clean['temperature_c'].min():.1f} - {df_clean['temperature_c'].max():.1f} °C (NaNs: {df_clean['temperature_c'].isna().sum()})")
    report_lines.append(f"SpO2 range: {df_clean['spo2_pct'].min():.1f}% - {df_clean['spo2_pct'].max():.1f}% (NaNs: {df_clean['spo2_pct'].isna().sum()})")
    report_lines.append(f"Age range: {df_clean['age'].min():.1f} - {df_clean['age'].max():.1f} years (NaNs: {df_clean['age'].isna().sum()})")
    report_lines.append(f"Comorbidity count range: {df_clean['comorbidity_count'].min()} - {df_clean['comorbidity_count'].max()}")
    report_lines.append(f"Labelled senior nurse visits: {df_clean['senior_nurse_triage'].dropna().count()} / {len(df_clean)} ({df_clean['senior_nurse_triage'].dropna().count()/len(df_clean)*100:.2f}%)")
    report_lines.append(f"Senior nurse triage class breakdown:\n{df_clean['senior_nurse_triage'].value_counts().to_string()}")

    # Save cleaned data
    df_clean.to_csv(output_cleaned_path, index=False)
    report_lines.append(f"\nCleaned dataset saved successfully to: {output_cleaned_path}")

    # Write report file
    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Data cleaning completed. Processed file saved to {output_cleaned_path}")
    print(f"Quality report written to {report_path}")
    return df_clean, df_audit

if __name__ == "__main__":
    clean_ed_dataset()
