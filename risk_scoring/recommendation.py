import numpy as np
from config import (SIGNIFICANT_EARNINGS_SURPRISE_THRESHOLD,
                    LOW_REACTION_THRESHOLD,
                    POSITIVE_SURPRISE_THRESHOLD,
                    NEGATIVE_SURPRISE_THRESHOLD)

def recommendation_from_risk(score):
    """
        Sell / Sell 50% / Hold

        TODO: Very simple mapping: it assumes risk = “position size to cut”.

        Doesn't use:
        trend
        valuation
        investor time horizon
        sector conditions

        But as a first-pass sizing suggestion it's reasonable for a tool called Risk/Volatility Tracker.
    """
    # Filter for Upcoming Earnings Only
    # today = datetime.today()
    # future_window = timedelta(days=100)  # or whatever window you want
    # upcoming = earnings_df[
    #     (earnings_df['earnings_date'] >= today) &
    #     (earnings_df['earnings_date'] <= today + future_window)
    # ]
    if score >= 4.0:
        return "SELL"
    elif score >= 3.0:
        return "SELL 50%"
    else:
        return "HOLD"

def add_risk_recommendation(earnings_df):
    # Select relevant columns from earnings_df
    output_cols = [
        'stock',
        'earnings_date',
        'risk_score',
        'sector',
        'sub_sector',
        'beta_5y_monthly',
        'vol_10d',
        'sector_vol_10d',
        'sector_peer_risk_flag'
    ]

    output_df = earnings_df[output_cols].copy()

    output_df['risk_recommendation'] = output_df['risk_score'].apply(recommendation_from_risk)
    output_df = output_df.sort_values(['earnings_date', 'stock'])
    return output_df

def add_recommendation_explanations(earnings_df, output_df):
    """ 
        Add competitor-sensitivity explanations onto an existing output_df.

        Keeps all existing columns in output_df and just adds:
        - competitor_sensitivity
        - competitor_trend
        - competitor_influence
    """

    # --- Define grouping key ---
    group_key = 'sub_sector'

    # --- Derive competitor metrics (without editing earnings_df) ---
    tmp = earnings_df.copy()

    # Peer 3-day avg after earnings
    tmp['peer_avg_3d'] = (
        tmp.groupby([group_key, 'earnings_date'])['ret_3d_from_earnings']
        .transform('mean')
    )

    # Rolling competitor trend (8-event window)
    tmp['competitor_trend'] = (
        tmp.groupby(group_key)['peer_avg_3d']
        .transform(lambda x: x.rolling(8, min_periods=2).mean())
    )

    # Relative performance vs competitors
    tmp['relative_to_competitors'] = tmp['ret_3d_from_earnings'] - tmp['competitor_trend']

    # Sensitivity: correlation between own reaction and peer trend
    def competitor_sensitivity(group):
        if len(group) < 4:
            return np.nan
        return group['ret_3d_from_earnings'].corr(group['competitor_trend'])

    competitor_sensitivity_df = (
        tmp.groupby('stock', group_keys=False)
        .apply(competitor_sensitivity)
        .rename('competitor_sensitivity')
        .reset_index()
    )

    # Most recent competitor trend per stock
    latest_comp_trend = (
        tmp.sort_values('earnings_date')
        .groupby('stock', as_index=False)
        .last()[['stock', 'competitor_trend']]
    )

    # Merge the two
    competitor_output = competitor_sensitivity_df.merge(
        latest_comp_trend, on='stock', how='left'
    )

    # --- Flag competitor influence ---
    competitor_output['competitor_influence'] = np.select(
        [
            (competitor_output['competitor_sensitivity'] > 0.4) &
            (competitor_output['competitor_trend'] < 0),

            (competitor_output['competitor_sensitivity'] > 0.4) &
            (competitor_output['competitor_trend'] > 0)
        ],
        [
            "High - Peers recently weak, stock tends to follow their downside.",
            "Moderate - Peers recently strong, stock tends to benefit."
        ],
        default="Low - Weak or inverse linkage to competitors."
    )

    # # --- Join with recommendation and risk score for final output ---
    # output_df = earnings_df[['stock', 'earnings_date', 'risk_score']].drop_duplicates()
    # output_df['recommendation'] = output_df['risk_score'].apply(recommendation_from_risk)

    output_df = output_df.merge(competitor_output, on='stock', how='left')

    # Reorder for readability
    output_df = output_df.sort_values(['earnings_date', 'stock'])
    # output_df = output_df[
    #     ['stock', 'earnings_date', 'risk_score', 'recommendation',
    #     'competitor_sensitivity', 'competitor_trend', 'competitor_influence']
    # ].sort_values(['earnings_date', 'stock'], ascending=[False, True] )

    return output_df

def pre_risk_label(x):
        if x > 0.5:
            return "High - Often drops despite positive earnings."
        elif x > 0.2:
            return "Moderate - Mixed post-earnings reactions."
        else:
            return "Low - Typically rewarded for good earnings."

def add_pre_earnings_risk_flag(earnings_df, output_df):
    """ 
        Adds the pre_earnings_risk_flag feature:
        Flags based on a stock-level risk score (pre_earnings_risk_score) built from the frequency of drops after positive earnings (mean).

        Answers “How often does this stock sell off after good earnings?”
        High score = market doesn't trust the company even when numbers are good
        high → beware optimism traps
        low → market trusts the company 
    """
    # --- 1. Temporary working copy ---
    tmp = earnings_df.copy()

    # Positive surprise & negative post-earnings return
    tmp['positive_surprise'] = tmp['surprise'] > 0
    tmp['drop_after_positive'] = tmp['positive_surprise'] & (tmp['ret_3d_from_earnings'] < 0)

    # --- 2. Compute per-stock frequency ---
    pre_risk = (
        tmp.groupby('stock')['drop_after_positive']
        .mean()
        .rename('pre_earnings_risk_score')
        .reset_index()
    )

    # --- 3. Descriptive label for interpretability ---

    pre_risk['pre_earnings_risk_flag'] = pre_risk['pre_earnings_risk_score'].apply(pre_risk_label)

    # --- 4. Merge with existing output (recommendation + competitor influence) ---
    output_df = (
        output_df
        .merge(pre_risk, on='stock', how='left')
        .sort_values(['earnings_date', 'stock'], ascending=[False, True])
    )

    return output_df

def add_sector_level_risk_flags(earnings_df, output_df):
    # --- 1. Temporary working copy ---
    tmp = earnings_df.copy()

    # --- 2. Compute sector-level rolling drift & volatility averages (10-event window) ---
    sector_trend = (
        tmp.groupby('sector', group_keys=False)
        .agg({
            'sector_drift_10d': 'mean',
            'sector_vol_10d': 'mean',
            'sector_beta': 'mean'
        })
        .rename(columns={
            'sector_drift_10d': 'sector_avg_drift',
            'sector_vol_10d': 'sector_avg_vol',
            'sector_beta': 'sector_avg_beta'
        })
        .reset_index()
    )

    # --- 3. Label risk based on drift/volatility pattern ---
    def sector_risk_label(row):
        # Compute 30th, 70th percentiles for drift and vol to set cutoffs
        drift_low, drift_high = sector_trend['sector_avg_drift'].quantile([0.3, 0.7])
        vol_high = sector_trend['sector_avg_vol'].quantile(0.7)

        if row['sector_avg_drift'] <= drift_low and row['sector_avg_vol'] >= vol_high:
            return "High - Sector under pressure with elevated volatility."
        elif row['sector_avg_drift'] <= drift_low:
            return "Moderate - Sector showing mild weakness."
        elif row['sector_avg_drift'] >= drift_high and row['sector_avg_vol'] < vol_high:
            return "Low - Sector performing strongly with stable volatility."
        else:
            return "Neutral - No strong sector trend detected."

    sector_trend['sector_risk_flag'] = sector_trend.apply(sector_risk_label, axis=1)

    # --- 4. Bring sector info to output_df temporarily ---
    # (earnings_df may have multiple rows per stock, so take latest known sector)
    latest_sector = (
        tmp.sort_values('earnings_date')
        .groupby('stock', as_index=False)
        .last()[['stock', 'sector']]
    )

    output_df = output_df.merge(latest_sector, on='stock', how='left')
    # --- 5. Merge sector-level risk flags ---
    # output_df = output_df.merge(sector_trend[['sector', 'sector_risk_flag']], on='sector', how='left')
    # output_df = output_df.drop(columns=['sector']).sort_values(['earnings_date', 'stock'])

    return output_df

def add_excessive_price_move_alert(earnings_df, output_df):
    """
        Excessive Price Move Alert
        Adds avg_reaction_8q, excessive_move_flag, excessive_move_label
    """
    tmp = earnings_df.sort_values(['stock', 'earnings_date']).copy()

    # --- 2. Compute rolling average of absolute 3-day post-earnings moves (last 8 quarters) ---
    tmp['avg_reaction_8q'] = (
        tmp.groupby('stock')['ret_3d_from_earnings']
        .transform(lambda x: x.abs().shift().rolling(8, min_periods=2).mean())
    )

    # --- 3. Compare current reaction to historical average ---
    tmp['excessive_move_flag'] = np.where(
        tmp['ret_3d_from_earnings'].abs() > tmp['avg_reaction_8q'],
        1, 0
    )

    # --- 4. Create human-readable label for easier interpretation ---
    tmp['excessive_move_label'] = np.select(
        [
            tmp['excessive_move_flag'] == 1,
            tmp['excessive_move_flag'] == 0
        ],
        [
            "Yes - Move exceeds 8-quarter average.",
            "No - Within normal range."
        ],
        default="No data"
    )

    # --- 5. Keep only relevant fields ---
    move_alert = tmp[['stock', 'earnings_date', 'ret_3d_from_earnings',
                    'avg_reaction_8q', 'excessive_move_label']].copy()

    # --- 6. Merge into your main output_df ---
    output_df = output_df.merge(move_alert, on=['stock', 'earnings_date'], how='left')

    # --- 7. Save updated output ---
    output_df = output_df.sort_values(['earnings_date', 'stock'])

    return output_df

def add_surprise_no_reaction_alert(earnings_df, output_df):
    """ Earnings Surprise with No Reaction """
    tmp = earnings_df.copy()

    # --- 3. Create component flags ---
    tmp['strong_positive_surprise'] = tmp['surprisePercentage'] >= SIGNIFICANT_EARNINGS_SURPRISE_THRESHOLD
    tmp['no_reaction'] = tmp['ret_3d_from_earnings'].abs() < LOW_REACTION_THRESHOLD

    # --- 4. Combine both to form the alert condition ---
    tmp['earnings_surprise_no_reaction'] = (
        tmp['strong_positive_surprise'] & tmp['no_reaction']
    )

    # --- 5. Human-readable alert label ---
    tmp['surprise_no_reaction_alert'] = np.where(
        tmp['earnings_surprise_no_reaction'],
        "Strong earnings but muted price response.",
        "None"
    )

    # --- 6. Keep only relevant columns for merge ---
    no_reaction_alert = tmp[['stock', 'earnings_date', 'surprise_no_reaction_alert']].copy()

    # --- 7. Merge into the main output_df ---
    output_df = output_df.merge(no_reaction_alert, on=['stock', 'earnings_date'], how='left')

    # --- 8. Save updated output ---
    output_df = output_df.sort_values(['earnings_date', 'stock'])

    return output_df

def add_eps_reaction_divergence_alert(earnings_df, output_df):
    """ Implied vs. Actual Reaction Divergence """

    tmp = earnings_df.copy()

    # --- 3. Expected direction based on earnings surprise ---
    tmp['expected_positive'] = tmp['surprisePercentage'] > POSITIVE_SURPRISE_THRESHOLD
    tmp['expected_negative'] = tmp['surprisePercentage'] < NEGATIVE_SURPRISE_THRESHOLD

    # --- 4. Actual direction based on price reaction ---
    tmp['actual_up']   = tmp['ret_3d_from_earnings'] > 0
    tmp['actual_down'] = tmp['ret_3d_from_earnings'] < 0

    # --- 5. Flag divergence (earnings beat but price falls, or miss but price rises) ---
    tmp['reaction_divergence'] = (
        (tmp['expected_positive'] & tmp['actual_down']) |
        (tmp['expected_negative'] & tmp['actual_up'])
    )

    # --- 6. Descriptive alert text ---
    tmp['divergence_alert'] = np.select(
        [
            tmp['expected_positive'] & tmp['actual_down'],
            tmp['expected_negative'] & tmp['actual_up']
        ],
        [
            "EPS BEAT but price FELL.",
            "EPS MISS but price ROSE."
        ],
        default="None"
    )

    # --- 7. Keep only needed columns ---
    divergence_alert = tmp[['stock', 'earnings_date', 'reaction_divergence', 'divergence_alert']].copy()

    # --- 8. Merge with existing output ---
    output_df = output_df.merge(divergence_alert, on=['stock', 'earnings_date'], how='left')

    # --- 9. Save updated file ---
    output_df = output_df.sort_values(['earnings_date', 'stock'])

    return output_df

def add_negative_sentiment_alert(earnings_df, output_df):
    """ Negative Sentiment Alert 

        Finish 3B Earnings Call Sentiment Analysis First!
        Flag cases where a company’s latest earnings call transcript shows a spike in negative sentiment relative to its past 6-8 quarters.
        This will help identify potential management pessimism, guidance downgrades, or hidden bad news not yet reflected in the numbers.

        Finish 3B Earnings Call Sentiment Analysis First!
    """
    pass

def add_muted_response_alert(earnings_df, output_df):
    """
        Muted Stock Response to Earnings Beat/Miss
        Adds 'muted_response_alert_flag', 'muted_response_alert' features.
     """
    tmp = earnings_df.copy()

    

    # --- 3. Flag strong beat or miss ---
    tmp['big_beat'] = tmp['surprisePercentage'] >= SIGNIFICANT_EARNINGS_SURPRISE_THRESHOLD
    tmp['big_miss'] = tmp['surprisePercentage'] <= -SIGNIFICANT_EARNINGS_SURPRISE_THRESHOLD

    # --- 4. Flag muted stock movement ---
    tmp['muted_move'] = tmp['ret_3d_from_earnings'].abs() < LOW_REACTION_THRESHOLD

    # --- 5. Combine conditions into a single alert flag ---
    tmp['muted_response_alert_flag'] = (
        (tmp['big_beat'] | tmp['big_miss']) & tmp['muted_move']
    )

    # --- 6. Human-readable alert label ---
    tmp['muted_response_alert'] = np.select(
        [
            tmp['big_beat'] & tmp['muted_move'],
            tmp['big_miss'] & tmp['muted_move']
        ],
        [
            "Strong EPS beat (≥ +10%) but muted price reaction (±1%).",
            "Big EPS miss (≤ −10%) but muted price reaction (±1%)."
        ],
        default="None"
    )

    # --- 7. Keep only relevant columns for merging ---
    muted_alert = tmp[['stock', 'earnings_date', 'muted_response_alert_flag',
                    'muted_response_alert']].copy()

    # --- 8. Merge into main output_df ---
    output_df = output_df.merge(muted_alert, on=['stock', 'earnings_date'], how='left')

    # --- 9. Save updated output ---
    output_df = output_df.sort_values(['earnings_date', 'stock'])

    return output_df

def add_extreme_volatility_alert(earnings_df, output_df):
    tmp = earnings_df.sort_values(['stock', 'earnings_date']).copy()
    # --- 2) Parameters (tunable) ---
    vol_floor = 1e-6          # avoid division by zero
    ratio_threshold = 3.0      # "exceeds 3x implied range"
    use_return_cols = ['ret_3d_from_earnings', 'ret_10d_from_earnings']

    # --- 3) Realized absolute moves ---
    tmp['realized_move_3d']  = tmp['ret_3d_from_earnings'].abs()
    tmp['realized_move_10d'] = tmp['ret_10d_from_earnings'].abs()

    # --- 4) Ratios vs (implied) volatility proxy (vol_10d) ---
    # If vol_10d is NaN or zero, replace with floor so ratios stay defined
    denom = (tmp['vol_10d'].fillna(0) + vol_floor)
    tmp['move_vs_vol_3d']  = tmp['realized_move_3d']  / denom
    tmp['move_vs_vol_10d'] = tmp['realized_move_10d'] / denom

    # --- 5) Flag & human label ---
    tmp['extreme_volatility_alert_flag'] = (
        (tmp['move_vs_vol_3d']  > ratio_threshold) |
        (tmp['move_vs_vol_10d'] > ratio_threshold)
    ).astype(int)

    tmp['extreme_volatility_alert'] = np.where(
        tmp['extreme_volatility_alert_flag'] == 1,
        "Extreme post-earnings volatility: move > 2× normal 10-day range.",
        "None"
    )

    # Optional: concise numeric context for dashboards/CSV
    # (rounded to bps/% for readability)
    tmp['move_ctx'] = (
        "3d:" + (tmp['realized_move_3d']*100).round(2).astype(str) + "% "
        + "(x_sigma:" + tmp['move_vs_vol_3d'].round(2).astype(str) + "); "
        + "10d:" + (tmp['realized_move_10d']*100).round(2).astype(str) + "% "
        + "(x_sigma:" + tmp['move_vs_vol_10d'].round(2).astype(str) + ")"
    )

    # --- 6) Keep only needed columns for merge ---
    extreme_alert = tmp[[
        'stock', 'earnings_date',
        'realized_move_3d', 'realized_move_10d',
        'move_vs_vol_3d', 'move_vs_vol_10d',
        'extreme_volatility_alert_flag', 'extreme_volatility_alert',
        'move_ctx'
    ]].copy()

    # --- 7) Merge into your existing output_df  ---
    output_df = output_df.merge(extreme_alert, on=['stock', 'earnings_date'], how='left')
    output_df = output_df.sort_values(['earnings_date', 'stock'])
    
    return output_df

def prepare_df_for_dashboard(output_df):
    """ Clean and prepare output for the Dashboard """
    
    # Optional: keep only key columns for now
    cols_to_keep = [
        "earnings_date", "stock", "risk_score", "risk_recommendation",
        "excessive_move_label", "surprise_no_reaction_alert", "reaction_divergence",
        "muted_response_alert_flag", "extreme_volatility_alert_flag", "divergence_alert"
    ]

    dashboard_df = output_df[cols_to_keep]
    dashboard_df = dashboard_df.dropna(subset=["risk_score"]) # instead of dashboard_df = dashboard_df.dropna()
    

    # Rename for clean display
    dashboard_df = dashboard_df.rename(columns={
        "stock": "Stock",
        "earnings_date": "Date",
        "risk_score": "Risk Score",
        "risk_recommendation": "Recommendation",
        "excessive_move_label": "Excessive Move",
        "surprise_no_reaction_alert": "No Reaction",
        "reaction_divergence": "Reaction Divergence",
        "muted_response_alert_flag": "Muted Response",
        "extreme_volatility_alert_flag": "Extreme Volatility"
    })

    return dashboard_df