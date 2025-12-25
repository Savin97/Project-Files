
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

def add_reccomendation_explanations():
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