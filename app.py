# app.py
import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# Add project root (parent of "streamlit") to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add project root (parent of "streamlit") to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CSV with the dashboard output
# put latest_earnings_df.csv in the repo root (same level as /streamlit)
CSV_PATH = ROOT / "latest_earnings_df.csv"  # change if you keep it elsewhere

from pipeline.pipeline import run_pipeline

# Streamlit page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)


@st.cache_data(show_spinner="Running Pipeline…")
def get_dashboard_df(use_cached_eps: bool = True) -> pd.DataFrame:
    """
    Calls your engine and returns the final dashboard dataframe.

    Assumes run_pipeline(...) returns a DataFrame with at least:
      Date, Stock, Risk Score, Recommendation,
      Excessive Move, No Reaction, Reaction Divergence,
      Muted Response, Extreme Volatility, Divergence Alert
    """
    df = run_pipeline()
    #df = run_pipeline(use_cached_eps=use_cached_eps)

    # Sanity check for expected columns
    expected_cols = [
        "Date",
        "Stock",
        "Risk Score",
        "Recommendation",
        "Excessive Move",
        "No Reaction",
        "Reaction Divergence",
        "Muted Response",
        "Extreme Volatility",
        "Divergence Alert",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    # Convenience: pre-compute any_alert flag
    df["any_alert"] = False
    if "Reaction Divergence" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Reaction Divergence"].fillna(False)
    if "Muted Response" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Muted Response"].fillna(False)
    if "Extreme Volatility" in df.columns:
        df["any_alert"] = df["any_alert"] | (df["Extreme Volatility"].fillna(0) != 0)
    if "No Reaction" in df.columns:
        df["any_alert"] = df["any_alert"] | df["No Reaction"].notna()
    if "Excessive Move" in df.columns:
        # flag anything that is not the plain "No - Within normal range." text
        df["any_alert"] = df["any_alert"] | ~df["Excessive Move"].fillna("").str.contains(
            "No - Within normal range.", na=False
        )
    if "Divergence Alert" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Divergence Alert"].notna()

    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Stock filter
    if "Stock" in df.columns:
        stocks = sorted(df["Stock"].dropna().unique())
        stock_choice = st.sidebar.selectbox(
            "Stock (optional)",
            options=["(All)"] + stocks,
        )
        if stock_choice != "(All)":
            df = df[df["Stock"] == stock_choice]

    # Risk score range
    if "Risk Score" in df.columns and not df["Risk Score"].isna().all():
        min_rs = int(df["Risk Score"].min())
        max_rs = int(df["Risk Score"].max())
        lo, hi = st.sidebar.slider(
            "Risk Score range",
            min_value=min_rs,
            max_value=max_rs,
            value=(min_rs, max_rs),
        )
        df = df[(df["Risk Score"] >= lo) & (df["Risk Score"] <= hi)]

    # Recommendation filter
    if "Recommendation" in df.columns:
        recs = sorted(df["Recommendation"].dropna().unique())
        selected_recs = st.sidebar.multiselect(
            "Recommendations",
            options=recs,
            default=recs,
        )
        if selected_recs:
            df = df[df["Recommendation"].isin(selected_recs)]

    # Only rows with any alert?
    if "any_alert" in df.columns:
        only_alerts = st.sidebar.checkbox(
            "Only rows with any risk/alert flag",
            value=False,
        )
        if only_alerts:
            df = df[df["any_alert"]]

    return df


def main():
    st.title("Breakwater")
    st.title("Earnings Risk & Alerts Dashboard")

    with st.sidebar:
        st.markdown("### Data options")
        if st.button("Reload CSV from disk"):
            # clear cache and reload on next get_dashboard_df() call
            get_dashboard_df.clear()

    raw_df = get_dashboard_df()
    df = raw_df.copy()

    # Apply sidebar filters
    df = sidebar_filters(df)

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Earnings events", len(df))
    with col2:
        if "Stock" in df.columns:
            st.metric("Unique stocks", df["Stock"].nunique())
    with col3:
        if "Risk Score" in df.columns:
            # naive threshold: 4+ considered high risk
            high_risk = (df["Risk Score"] >= 4).sum()
            st.metric("High-risk events (Score ≥ 4)", int(high_risk))
    with col4:
        if "any_alert" in df.columns:
            st.metric("Rows with alerts", int(df["any_alert"].sum()))

    # Tabs: Overview / Alerts / Stock detail
    tab_overview, tab_alerts, tab_stock = st.tabs(
        ["Overview", "Risk Alerts", "Stock drill-down"]
    )

    # -------- Overview tab --------
    with tab_overview:
        st.subheader("Filtered earnings events")

        cols_to_show = [
            c
            for c in [
                "Date",
                "Stock",
                "Risk Score",
                "Recommendation",
                "Excessive Move",
                "No Reaction",
                "Reaction Divergence",
                "Muted Response",
                "Extreme Volatility",
                "Divergence Alert",
            ]
            if c in df.columns
        ]

        if "Date" in df.columns:
            df_display = df.sort_values("Date", ascending=False)
        else:
            df_display = df

        st.dataframe(
            df_display[cols_to_show],
            column_config={
                "Date": st.column_config.DateColumn(format="DD/MM/YYYY")
            }
        )
        # st.dataframe(df_display[cols_to_show])

        # Simple aggregate: count of events by risk score
        if "Risk Score" in df.columns:
            st.markdown("#### Count of events by Risk Score")
            agg = (
                df.groupby("Risk Score")["Stock"]
                .count()
                .rename("count")
                .reset_index()
                .sort_values("Risk Score")
            )
            chart_df = agg.set_index("Risk Score")["count"]
            st.bar_chart(chart_df)

    # -------- Risk Alerts tab --------
    with tab_alerts:
        st.subheader("Flagged risk cases")

        if "any_alert" not in df.columns or not df["any_alert"].any():
            st.info("No rows with alert flags in the filtered data.")
        else:
            alerts = df[df["any_alert"]].copy()
            if "Date" in alerts.columns:
                alerts = alerts.sort_values("Date", ascending=False)

            alert_cols = [
                c
                for c in [
                    "Date",
                    "Stock",
                    "Risk Score",
                    "Recommendation",
                    "Excessive Move",
                    "No Reaction",
                    "Reaction Divergence",
                    "Muted Response",
                    "Extreme Volatility",
                    "Divergence Alert",
                ]
                if c in alerts.columns
            ]

            st.dataframe(alerts[alert_cols])

    # -------- Stock drill-down tab --------
    with tab_stock:
        st.subheader("Single-stock history")

        if "Stock" not in df.columns:
            st.info("Stock column not found.")
        else:
            stocks = sorted(df["Stock"].dropna().unique())
            selected_stock = st.selectbox("Choose stock", options=stocks)

            stock_df = df[df["Stock"] == selected_stock].copy()
            if "Date" in stock_df.columns:
                stock_df = stock_df.sort_values("Date")

            cols = [
                c
                for c in [
                    "Date",
                    "Risk Score",
                    "Recommendation",
                    "Excessive Move",
                    "No Reaction",
                    "Reaction Divergence",
                    "Muted Response",
                    "Extreme Volatility",
                    "Divergence Alert",
                ]
                if c in stock_df.columns
            ]
            st.dataframe(stock_df[cols])

            # Quick line chart: Risk Score over time
            if {"Date", "Risk Score"}.issubset(stock_df.columns):
                chart_df = stock_df.set_index("Date")["Risk Score"]
                st.line_chart(chart_df)


if __name__ == "__main__":
    main()









# st.set_page_config(
#     page_title="Dashboard",
#     layout="wide"
# )

# @st.cache_data(show_spinner="Running pipeline…")
# def get_dashboard_df(use_cached_eps: bool = True) -> pd.DataFrame:
#     #return run_pipeline(use_cached_eps=use_cached_eps)
#     return run_pipeline()


# def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
#     st.sidebar.header("Filters")

#     # EPS surprise threshold (your surprisePercentage is in decimals 0–1)
#     min_surprise_pct = st.sidebar.slider(
#         "Min |EPS surprise| (%)",
#         min_value=0.0,
#         max_value=30.0,
#         value=5.0,
#         step=0.5,
#     )
#     if "surprisePercentage" in df.columns:
#         df = df[df["surprisePercentage"].abs() >= (min_surprise_pct / 100.0)]

#     # Sector filter
#     if "sector" in df.columns:
#         sectors = sorted(df["sector"].dropna().unique())
#         selected_sectors = st.sidebar.multiselect(
#             "Sectors",
#             options=sectors,
#             default=sectors,
#         )
#         if selected_sectors:
#             df = df[df["sector"].isin(selected_sectors)]

#     # Stock filter
#     if "stock" in df.columns:
#         stocks = sorted(df["stock"].dropna().unique())
#         stock_choice = st.sidebar.selectbox(
#             "Single stock (optional)",
#             options=["(All)"] + stocks,
#         )
#         if stock_choice != "(All)":
#             df = df[df["stock"] == stock_choice]

#     # Only big moves?
#     if {"flag_diff_3d", "flag_diff_10d"}.issubset(df.columns):
#         only_big_moves = st.sidebar.checkbox(
#             "Only large post-earnings moves vs sector",
#             value=False,
#         )
#         if only_big_moves:
#             df = df[(df["flag_diff_3d"] == 1) | (df["flag_diff_10d"] == 1)]

#     # Sector & peer risk only?
#     if "sector_peer_risk_flag" in df.columns:
#         only_peer_risk = st.sidebar.checkbox(
#             "Only Sector & Peer Performance Risk cases",
#             value=False,
#         )
#         if only_peer_risk:
#             df = df[df["sector_peer_risk_flag"] == 1]

#     # Direction filter
#     if "reaction_3d" in df.columns:
#         st.sidebar.markdown("**3-day reaction direction**")
#         up = st.sidebar.checkbox("Up", value=True)
#         down = st.sidebar.checkbox("Down", value=True)
#         flat = st.sidebar.checkbox("No change", value=True)

#         allowed = []
#         if up:
#             allowed.append(1)
#         if down:
#             allowed.append(-1)
#         if flat:
#             allowed.append(0)
#         if allowed:
#             df = df[df["reaction_3d"].isin(allowed)]

#     return df


# def main():
#     st.title("Earnings Reaction & Risk Dashboard")

#     with st.sidebar:
#         st.markdown("### Data options")
#         use_cached_eps = st.checkbox(
#             "Use existing EPS CSV (no fresh API calls)",
#             value=True,
#             help="Turn off if you later add a 'force refresh' path in fetch_EPS.",
#         )
#         run_btn = st.button("Run / Refresh pipeline")

#     if run_btn:
#         st.session_state["earnings_df"] = get_dashboard_df(use_cached_eps=use_cached_eps)
    
#     # First run in a session
#     if "earnings_df" not in st.session_state:
#         st.session_state["earnings_df"] = get_dashboard_df(use_cached_eps=True)

#     raw_df = st.session_state["earnings_df"]
#     df = raw_df.copy()
#     df.to_csv("output/streamlit_latest_earnings_df.csv", index=False)
#     # Apply sidebar filters
#     df = sidebar_filters(df)

#     # High-level KPIs
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Earnings events", len(df))
#     with col2:
#         st.metric("Unique stocks", df["stock"].nunique() if "stock" in df.columns else 0)
#     with col3:
#         if "sector_peer_risk_flag" in df.columns:
#             st.metric("Peer-risk cases", int(df["sector_peer_risk_flag"].sum()))
#     with col4:
#         if {"flag_diff_3d", "flag_diff_10d"}.issubset(df.columns):
#             n_big = int(((df["flag_diff_3d"] == 1) | (df["flag_diff_10d"] == 1)).sum())
#             st.metric("Big move anomalies", n_big)

#     # Tabs: Overview / Alerts / Stock detail
#     tab_overview, tab_alerts, tab_stock = st.tabs(
#         ["Overview", "Risk Alerts", "Stock drill-down"]
#     )

#     with tab_overview:
#         st.subheader("Filtered earnings events")
#         # Show a reasonable subset of columns
#         cols_to_show = [
#             c
#             for c in [
#                 "Stock",
#                 "sector",
#                 "Date",
#                 "surprisePercentage",
#                 "ret_3d_from_earnings",
#                 "ret_10d_from_earnings",
#                 "reaction_3d",
#                 "reaction_10d",
#                 "surprise_bucket",
#             ]
#             # "stock": "Stock",
#             # "Date": "Date",
#             # "risk_score": "Risk Score",
#             # "risk_recommendation": "Recommendation",
#             # "excessive_move_label": "Excessive Move",
#             # "surprise_no_reaction_alert": "No Reaction",
#             # "reaction_divergence": "Reaction Divergence",
#             # "muted_response_alert_flag": "Muted Response",
#             # "extreme_volatility_alert_flag": "Extreme Volatility"
#             if c in df.columns
#         ]
#         st.dataframe(df[cols_to_show].sort_values("Date", ascending=False))

#         # Simple aggregate: average 3-day reaction by surprise bucket
#         if {"surprise_bucket", "ret_3d_from_earnings"}.issubset(df.columns):
#             st.markdown("#### Avg 3-day reaction by surprise bucket")
#             agg = (
#                 df.groupby("surprise_bucket")["ret_3d_from_earnings"]
#                 .mean()
#                 .rename("avg_ret_3d")
#                 .reset_index()
#             )
#             st.bar_chart(
#                 agg.set_index("surprise_bucket")["avg_ret_3d"]
#             )

#     with tab_alerts:
#         st.subheader("Flagged risk cases")

#         alerts = df.copy()
#         # Define a combined flag
#         if "sector_peer_risk_flag" in alerts.columns:
#             alerts["any_peer_risk"] = alerts["sector_peer_risk_flag"]
#         if {"flag_diff_3d", "flag_diff_10d"}.issubset(alerts.columns):
#             alerts["any_big_move"] = (
#                 (alerts["flag_diff_3d"] == 1) | (alerts["flag_diff_10d"] == 1)
#             ).astype(int)

#         alert_cols = [
#             c
#             for c in [
#                 "stock",
#                 "sector",
#                 "Date",
#                 "surprisePercentage",
#                 "ret_3d_from_earnings",
#                 "ret_10d_from_earnings",
#                 "relative_3d",
#                 "relative_10d",
#                 "sector_mean_3d_same_day",
#                 "sector_peer_risk_flag",
#                 "flag_diff_3d",
#                 "flag_diff_10d",
#             ]
#             if c in alerts.columns
#         ]

#         # Show only rows with at least one interesting flag
#         mask = pd.Series(True, index=alerts.index)
#         if "sector_peer_risk_flag" in alerts.columns:
#             mask = mask & (alerts["sector_peer_risk_flag"] == 1)
#         if {"flag_diff_3d", "flag_diff_10d"}.issubset(alerts.columns):
#             mask = mask | (alerts["flag_diff_3d"] == 1) | (alerts["flag_diff_10d"] == 1)

#         alerts = alerts[mask]
#         st.dataframe(alerts[alert_cols].sort_values("Date", ascending=False))

#     with tab_stock:
#         st.subheader("Single-stock earnings history")

#         if "stock" not in df.columns:
#             st.info("Stock column not found.")
#         else:
#             stocks = sorted(df["stock"].unique())
#             selected_stock = st.selectbox("Choose stock", options=stocks)
#             stock_df = df[df["stock"] == selected_stock].copy()

#             if "Date" in stock_df.columns:
#                 stock_df = stock_df.sort_values("Date")

#             cols = [
#                 c
#                 for c in [
#                     "Date",
#                     "surprisePercentage",
#                     "ret_3d_from_earnings",
#                     "ret_10d_from_earnings",
#                     "reaction_3d",
#                     "reaction_10d",
#                     "surprise_bucket",
#                     "past_up_freq",
#                     "past_down_freq",
#                     "past_nochange_freq",
#                     "sector_peer_risk_flag",
#                 ]
#                 if c in stock_df.columns
#             ]
#             st.dataframe(stock_df[cols])

#             # Quick line chart: 3-day reaction over time
#             if {"Date", "ret_3d_from_earnings"}.issubset(stock_df.columns):
#                 chart_df = stock_df.set_index("Date")["ret_3d_from_earnings"]
#                 st.line_chart(chart_df)


# if __name__ == "__main__":
#     main()
