# app.py
import streamlit as st
import pandas as pd

from pipeline import run_pipeline # TODO: change to main.py or edit main.py to have a run_pipeline function


st.set_page_config(
    page_title="Earnings Reaction & Risk Dashboard",
    layout="wide"
)


@st.cache_data(show_spinner="Running pipeline…")
def get_earnings_df(use_cached_eps: bool = True) -> pd.DataFrame:
    return run_pipeline(use_cached_eps=use_cached_eps)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # EPS surprise threshold (your surprisePercentage is in decimals 0–1)
    min_surprise_pct = st.sidebar.slider(
        "Min |EPS surprise| (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
    )
    if "surprisePercentage" in df.columns:
        df = df[df["surprisePercentage"].abs() >= (min_surprise_pct / 100.0)]

    # Sector filter
    if "sector" in df.columns:
        sectors = sorted(df["sector"].dropna().unique())
        selected_sectors = st.sidebar.multiselect(
            "Sectors",
            options=sectors,
            default=sectors,
        )
        if selected_sectors:
            df = df[df["sector"].isin(selected_sectors)]

    # Stock filter
    if "stock" in df.columns:
        stocks = sorted(df["stock"].dropna().unique())
        stock_choice = st.sidebar.selectbox(
            "Single stock (optional)",
            options=["(All)"] + stocks,
        )
        if stock_choice != "(All)":
            df = df[df["stock"] == stock_choice]

    # Only big moves?
    if {"flag_diff_3d", "flag_diff_10d"}.issubset(df.columns):
        only_big_moves = st.sidebar.checkbox(
            "Only large post-earnings moves vs sector",
            value=False,
        )
        if only_big_moves:
            df = df[(df["flag_diff_3d"] == 1) | (df["flag_diff_10d"] == 1)]

    # Sector & peer risk only?
    if "sector_peer_risk_flag" in df.columns:
        only_peer_risk = st.sidebar.checkbox(
            "Only Sector & Peer Performance Risk cases",
            value=False,
        )
        if only_peer_risk:
            df = df[df["sector_peer_risk_flag"] == 1]

    # Direction filter
    if "reaction_3d" in df.columns:
        st.sidebar.markdown("**3-day reaction direction**")
        up = st.sidebar.checkbox("Up", value=True)
        down = st.sidebar.checkbox("Down", value=True)
        flat = st.sidebar.checkbox("No change", value=True)

        allowed = []
        if up:
            allowed.append(1)
        if down:
            allowed.append(-1)
        if flat:
            allowed.append(0)
        if allowed:
            df = df[df["reaction_3d"].isin(allowed)]

    return df


def main():
    st.title("Earnings Reaction & Risk Dashboard")

    with st.sidebar:
        st.markdown("### Data options")
        use_cached_eps = st.checkbox(
            "Use existing EPS CSV (no fresh API calls)",
            value=True,
            help="Turn off if you later add a 'force refresh' path in fetch_EPS.",
        )
        run_btn = st.button("Run / Refresh pipeline")

    if run_btn:
        st.session_state["earnings_df"] = get_earnings_df(use_cached_eps=use_cached_eps)

    # First run in a session
    if "earnings_df" not in st.session_state:
        st.session_state["earnings_df"] = get_earnings_df(use_cached_eps=True)

    raw_df = st.session_state["earnings_df"]
    df = raw_df.copy()

    # Apply sidebar filters
    df = sidebar_filters(df)

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Earnings events", len(df))
    with col2:
        st.metric("Unique stocks", df["stock"].nunique() if "stock" in df.columns else 0)
    with col3:
        if "sector_peer_risk_flag" in df.columns:
            st.metric("Peer-risk cases", int(df["sector_peer_risk_flag"].sum()))
    with col4:
        if {"flag_diff_3d", "flag_diff_10d"}.issubset(df.columns):
            n_big = int(((df["flag_diff_3d"] == 1) | (df["flag_diff_10d"] == 1)).sum())
            st.metric("Big move anomalies", n_big)

    # Tabs: Overview / Alerts / Stock detail
    tab_overview, tab_alerts, tab_stock = st.tabs(
        ["Overview", "Risk Alerts", "Stock drill-down"]
    )

    with tab_overview:
        st.subheader("Filtered earnings events")
        # Show a reasonable subset of columns
        cols_to_show = [
            c
            for c in [
                "stock",
                "sector",
                "earnings_date",
                "surprisePercentage",
                "ret_3d_from_earnings",
                "ret_10d_from_earnings",
                "reaction_3d",
                "reaction_10d",
                "surprise_bucket",
            ]
            if c in df.columns
        ]
        st.dataframe(df[cols_to_show].sort_values("earnings_date", ascending=False))

        # Simple aggregate: average 3-day reaction by surprise bucket
        if {"surprise_bucket", "ret_3d_from_earnings"}.issubset(df.columns):
            st.markdown("#### Avg 3-day reaction by surprise bucket")
            agg = (
                df.groupby("surprise_bucket")["ret_3d_from_earnings"]
                .mean()
                .rename("avg_ret_3d")
                .reset_index()
            )
            st.bar_chart(
                agg.set_index("surprise_bucket")["avg_ret_3d"]
            )

    with tab_alerts:
        st.subheader("Flagged risk cases")

        alerts = df.copy()
        # Define a combined flag
        if "sector_peer_risk_flag" in alerts.columns:
            alerts["any_peer_risk"] = alerts["sector_peer_risk_flag"]
        if {"flag_diff_3d", "flag_diff_10d"}.issubset(alerts.columns):
            alerts["any_big_move"] = (
                (alerts["flag_diff_3d"] == 1) | (alerts["flag_diff_10d"] == 1)
            ).astype(int)

        alert_cols = [
            c
            for c in [
                "stock",
                "sector",
                "earnings_date",
                "surprisePercentage",
                "ret_3d_from_earnings",
                "ret_10d_from_earnings",
                "relative_3d",
                "relative_10d",
                "sector_mean_3d_same_day",
                "sector_peer_risk_flag",
                "flag_diff_3d",
                "flag_diff_10d",
            ]
            if c in alerts.columns
        ]

        # Show only rows with at least one interesting flag
        mask = pd.Series(True, index=alerts.index)
        if "sector_peer_risk_flag" in alerts.columns:
            mask = mask & (alerts["sector_peer_risk_flag"] == 1)
        if {"flag_diff_3d", "flag_diff_10d"}.issubset(alerts.columns):
            mask = mask | (alerts["flag_diff_3d"] == 1) | (alerts["flag_diff_10d"] == 1)

        alerts = alerts[mask]
        st.dataframe(alerts[alert_cols].sort_values("earnings_date", ascending=False))

    with tab_stock:
        st.subheader("Single-stock earnings history")

        if "stock" not in df.columns:
            st.info("Stock column not found.")
        else:
            stocks = sorted(df["stock"].unique())
            selected_stock = st.selectbox("Choose stock", options=stocks)
            stock_df = df[df["stock"] == selected_stock].copy()

            if "earnings_date" in stock_df.columns:
                stock_df = stock_df.sort_values("earnings_date")

            cols = [
                c
                for c in [
                    "earnings_date",
                    "surprisePercentage",
                    "ret_3d_from_earnings",
                    "ret_10d_from_earnings",
                    "reaction_3d",
                    "reaction_10d",
                    "surprise_bucket",
                    "past_up_freq",
                    "past_down_freq",
                    "past_nochange_freq",
                    "sector_peer_risk_flag",
                ]
                if c in stock_df.columns
            ]
            st.dataframe(stock_df[cols])

            # Quick line chart: 3-day reaction over time
            if {"earnings_date", "ret_3d_from_earnings"}.issubset(stock_df.columns):
                chart_df = stock_df.set_index("earnings_date")["ret_3d_from_earnings"]
                st.line_chart(chart_df)


if __name__ == "__main__":
    main()
