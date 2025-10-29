"""
data_drift.py
-------------
Detects and visualizes feature drift between training, validation, and test sets
using KS-test and summary tables.
"""

import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift(train_df, test_df):
    """
    Perform KS-test for each feature and detect drift.
    """
    drift_table = []

    for col in train_df.columns:
        stat, pval = ks_2samp(train_df[col], test_df[col])
        drift_table.append({
            "feature": col,
            "p_value": pval,
            "drift_detected": pval < 0.05
        })

    drift_df = pd.DataFrame(drift_table)
    print("\nTop 5 Drifted Features:")
    print(drift_df.sort_values("p_value").head(5))
    return drift_df
