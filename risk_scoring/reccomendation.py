import numpy as np
def recommendation_from_risk(score):
    """
        Sell / Sell 50% / Hold
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

def add_reccomendation_explanations(earnings_df):
    """ 
        Adds the following to the output_df:

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

    # --- Join with recommendation and risk score for final output ---
    output_df = earnings_df[['stock', 'earnings_date', 'risk_score']].drop_duplicates()
    output_df['recommendation'] = output_df['risk_score'].apply(recommendation_from_risk)

    output_df = output_df.merge(competitor_output, on='stock', how='left')

    # Reorder for readability
    output_df = output_df[
        ['stock', 'earnings_date', 'risk_score', 'recommendation',
        'competitor_sensitivity', 'competitor_trend', 'competitor_influence']
    ].sort_values(['earnings_date', 'stock'], ascending=[False, True] )

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
    output_df = output_df.merge(sector_trend[['sector', 'sector_risk_flag']], on='sector', how='left')
    output_df = output_df.drop(columns=['sector']).sort_values(['earnings_date', 'stock'])

    return output_df

def add_excessive_price_move_alert(earnings_df, output_df):
    """
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
    tmp = earnings_df.copy()

    # --- 2. Thresholds ---
    significant_earnings_surprise_threshold = 5.0   # +5% or more EPS surprise
    low_reaction_threshold = 0.5                    # |price move| < 0.5%

    # --- 3. Create component flags ---
    tmp['strong_positive_surprise'] = tmp['surprisePercentage'] >= significant_earnings_surprise_threshold
    tmp['no_reaction'] = tmp['ret_3d_from_earnings'].abs() < low_reaction_threshold / 100

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