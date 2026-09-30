import streamlit as st
import numpy as np
import pandas as pd

st.title("2. Clean the data")

if "raw_df" not in st.session_state:
    st.warning("Run the Dataset page first.")
    st.stop()

if st.button("Run cleaning"):
    df = st.session_state.raw_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    duplicates_removed = int(df.duplicated(subset="date").sum())
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    # 2001-02-01's arrivals value (34,240,153) is not a real observation —
    # it's roughly 170x every other month in the dataset, bigger than the
    # country's entire best YEAR. This is a data-entry error, not a genuine
    # spike, so it's corrected here rather than just flagged: replaced with
    # the average of the month right before and right after it.
    corrected_rows = []
    target_date = pd.Timestamp("2001-02-01")
    mask = df["date"] == target_date
    if mask.any():
        prev_val = df["arrivals"].shift(1)[mask].values[0]
        next_val = df["arrivals"].shift(-1)[mask].values[0]
        old_val = df.loc[mask, "arrivals"].values[0]
        new_val = (prev_val + next_val) / 2
        df.loc[mask, "arrivals"] = new_val
        corrected_rows.append({"date": target_date.date(), "old_value": old_val, "new_value": new_val})

    missing = df.isna().sum()
    missing = missing[missing > 0]

    q1, q3 = df["arrivals"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df[(df["arrivals"] < lower) | (df["arrivals"] > upper)]

    # Rows with a genuinely missing target (arrivals) can't be used for
    # training at all, so those are dropped here rather than just flagged.
    rows_before = len(df)
    df = df.dropna(subset=["arrivals"]).reset_index(drop=True)
    rows_dropped_missing_target = rows_before - len(df)

    # A handful of scattered missing predictor readings (a temperature
    # column here and there) are filled in by interpolating from the
    # months right before and after them.
    numeric_cols = df.select_dtypes(include="number").columns.difference(["arrivals"])
    df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()

    st.session_state.clean_df = df
    st.session_state.clean_report = {
        "duplicates_removed": duplicates_removed,
        "corrected_rows": corrected_rows,
        "rows_dropped_missing_target": rows_dropped_missing_target,
        "missing": missing,
        "flagged": flagged[["date", "arrivals"]],
    }

if "clean_report" in st.session_state:
    report = st.session_state.clean_report
    st.write(f"Duplicates removed: {report['duplicates_removed']}")
    if report["corrected_rows"]:
        st.write("Corrected impossible values:")
        st.dataframe(pd.DataFrame(report["corrected_rows"]))
    st.write(f"Rows dropped (missing arrivals value): {report['rows_dropped_missing_target']}")
    st.write("Missing values by column (before cleaning):")
    st.dataframe(report["missing"])
    st.write("Flagged outliers (after correction):")
    st.dataframe(report["flagged"])
else:
    st.info('Click "Run cleaning" to process the dataset loaded on the Dataset page.')