# app.py
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Rate Compression & Lock-in Dashboard",
    layout="wide"
)

# CHANGE THIS to your actual DB file name if needed
DB_PATH = "9877b706-0a23-4a88-b791-d4fdc36ddf3a.db"


# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner=True)
def load_tenor_rates(db_path: str) -> pd.DataFrame:
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM tenor_rates", conn)
    finally:
        conn.close()

    # Basic cleaning
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["report_date", "Tenor"])
    return df


@st.cache_data(show_spinner=True)
def build_yield_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot tenor_rates into a date × tenor matrix of Mid Rates.
    """
    curve = df.pivot_table(
        index="report_date",
        columns="Tenor",
        values="Mid Rate"
    ).sort_index()

    return curve


def compute_spread_and_zscore(
    curve: pd.DataFrame,
    short_tenor: str,
    long_tenor: str,
    window: int
) -> pd.DataFrame:
    """
    Compute spread (long - short) and rolling Z-score.
    """
    if short_tenor not in curve.columns or long_tenor not in curve.columns:
        raise KeyError(f"Selected tenors not in data: {short_tenor}, {long_tenor}")

    spread = curve[long_tenor] - curve[short_tenor]
    roll = spread.rolling(window)

    mean = roll.mean()
    std = roll.std()

    zscore = (spread - mean) / std.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "spread": spread,
            "rolling_mean": mean,
            "rolling_std": std,
            "zscore": zscore,
        }
    )
    return out


# =========================
# MAIN APP
# =========================
def main():
    st.title("Rate Compression & Lock-in Analysis")

    st.markdown(
        """
        This dashboard analyses **PKRV tenor rates** to detect **yield curve compression** and 
        highlight potential **early lock-in windows**.
        """
    )

    # -------- Sidebar: parameters --------
    st.sidebar.header("Configuration")

    db_path = st.sidebar.text_input(
        "SQLite DB path",
        value=DB_PATH,
        help="Path to the SQLite file containing the tenor_rates table.",
    )

    with st.sidebar:
        st.markdown("---")
        rolling_window = st.number_input(
            "Rolling window (days)",
            min_value=5,
            max_value=90,
            value=30,
            step=1,
            help="Used for rolling mean/std and Z-score of the spread.",
        )

        z_threshold = st.number_input(
            "Lock-in Z-score threshold (compression)",
            min_value=-5.0,
            max_value=0.0,
            value=-1.5,
            step=0.1,
            help="Alerts when Z-score is below this value (strong compression).",
        )

    # -------- Load data --------
    try:
        df = load_tenor_rates(db_path)
    except Exception as e:
        st.error(f"Error loading tenor_rates from DB: {e}")
        st.stop()

    curve = build_yield_curve(df)

    # Sidebar tenor controls (after curve is known)
    all_tenors = sorted(curve.columns.tolist(), key=lambda x: (len(x), x))
    st.sidebar.subheader("Tenor selection")

    default_plot_tenors = [t for t in ["1M", "3M", "6M", "1Y", "3Y"] if t in all_tenors]
    selected_tenors = st.sidebar.multiselect(
        "Tenors to plot",
        options=all_tenors,
        default=default_plot_tenors or all_tenors[:5],
    )

    default_short = "1M" if "1M" in all_tenors else all_tenors[0]
    default_long = "1Y" if "1Y" in all_tenors else (all_tenors[-1] if all_tenors else None)

    col_s, col_l = st.sidebar.columns(2)
    with col_s:
        short_tenor = st.selectbox(
            "Short tenor (spread)",
            options=all_tenors,
            index=all_tenors.index(default_short) if default_short in all_tenors else 0,
        )
    with col_l:
        long_tenor = st.selectbox(
            "Long tenor (spread)",
            options=all_tenors,
            index=all_tenors.index(default_long) if default_long in all_tenors else len(all_tenors) - 1,
        )

    # Optional date filter
    st.sidebar.subheader("Date filter")
    min_date = curve.index.min().date()
    max_date = curve.index.max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        curve = curve.loc[(curve.index.date >= start_date) & (curve.index.date <= end_date)]

    # Guard against empty curve
    if curve.empty:
        st.warning("No data in selected date range.")
        st.stop()

    # -------- Compute spread & z-score --------
    try:
        spread_df = compute_spread_and_zscore(curve, short_tenor, long_tenor, rolling_window)
    except Exception as e:
        st.error(f"Error computing spread/z-score: {e}")
        st.stop()

    # Merge spread_df into curve index alignment
    # (They already share the same index, but just to be explicit)
    spread_df = spread_df.sort_index()

    # =========================
    # TOP METRICS
    # =========================
    latest_date = spread_df.dropna().index.max()
    latest_row = spread_df.loc[latest_date]

    latest_short = curve.loc[latest_date, short_tenor]
    latest_long = curve.loc[latest_date, long_tenor]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest date", latest_date.strftime("%Y-%m-%d"))
    col2.metric(f"{short_tenor} rate", f"{latest_short:.2f}%")
    col3.metric(f"{long_tenor} rate", f"{latest_long:.2f}%")
    col4.metric(
        f"Spread ({long_tenor} - {short_tenor})",
        f"{latest_row['spread']:.2f}%",
    )

    # =========================
    # YIELD CURVE CHART
    # =========================
    st.markdown("### Yield Curve History")
    st.caption("Selected tenors | daily PKRV mid rates")

    if selected_tenors:
        yield_plot_df = curve[selected_tenors].dropna(how="all")
        st.line_chart(yield_plot_df)
    else:
        st.info("Select at least one tenor in the sidebar to plot the yield curve.")

    # =========================
    # SPREAD & Z-SCORE CHARTS
    # =========================
    st.markdown(f"### Spread & Compression: {long_tenor} – {short_tenor}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Spread over time")
        st.caption("Positive = curve steeper; Negative = inverted / compressed relative to short tenor.")
        spread_plot_df = spread_df[["spread"]].dropna()
        st.line_chart(spread_plot_df)

    with col_right:
        st.subheader("Z-score of spread (rolling)")
        st.caption("Z-score < threshold → compression signal (potential early lock-in window).")
        z_plot_df = spread_df[["zscore"]].dropna()
        st.line_chart(z_plot_df)

    # =========================
    # LOCK-IN SIGNALS TABLE
    # =========================
    st.markdown("### Lock-in Signal Dates")

    signals = spread_df.copy()
    signals = signals[signals["zscore"] <= z_threshold].dropna(subset=["zscore"])

    if signals.empty:
        st.info(
            f"No dates found where Z-score ≤ {z_threshold:.2f} "
            f"for the selected tenors and window."
        )
    else:
        st.write(
            f"Showing dates where **Z-score ≤ {z_threshold:.2f}** "
            f"(strong compression):"
        )

        display_df = signals[["spread", "zscore"]].copy()
        display_df = display_df.rename(
            columns={
                "spread": f"Spread ({long_tenor}-{short_tenor})",
                "zscore": "Z-score",
            }
        )
        display_df = display_df.sort_index(ascending=False)

        st.dataframe(
            display_df.style.format(
                {
                    f"Spread ({long_tenor}-{short_tenor})": "{:.2f}",
                    "Z-score": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "These are potential **early lock-in windows**: the curve is significantly "
            "compressed relative to its recent history."
        )

    # =========================
    # RAW DATA (OPTIONAL)
    # =========================
    with st.expander("Show raw data (for debugging / export)"):
        st.write("Tenor rates (head):")
        st.dataframe(df.head())

        st.write("Yield curve matrix (head):")
        st.dataframe(curve.head())

        st.write("Spread & Z-score (head):")
        st.dataframe(spread_df.head())


if __name__ == "__main__":
    main()
