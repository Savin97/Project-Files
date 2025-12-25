
def risk_score(row):
    """
        Algorithm for scoring stocks based on absolute 10 day return, 10 day volatility, beta, surprise.

        Risk	                                  Behavior	                                                        Quantitative rule

        1 (Very Low)             stable, low beta & volatility                    abs_ret < 0.02 and volatility < 0.02 and beta < 0.8
        2 (Low)	                   stable, low beta & volatility                  	abs(ret_10d) < 0.04, beta < 0.8, volatility < 0.02
        3 (Moderate)    	    occasional inconsistency                        abs(ret_10d) between 2-5%, or moderate volatility (0.02-0.04)
        4 (High)	              frequent strong reactions	                        abs(ret_10d) > 0.05 or ret_10d opposite sign of surprise
        5 (Extreme)	                wild movements                                   abs(ret_10d) > 0.1 or very high volatility, volatility > 0.06
    """

    abs_ret = abs(row['ret_10d_from_earnings'])
    beta = row['beta_5y_monthly']
    volatility = row['vol_10d']
    surprise = row['surprisePercentage']
    ret10 = row['ret_10d_from_earnings']

    if abs_ret > 0.10 or volatility > 0.06:
        return 5  # Extreme
    if (surprise > 0 and ret10 < 0) or (surprise < 0 and ret10 > 0):
        return 4  # High – opposite reaction to surprise
    if (0.02 <= abs_ret <= 0.05) or (0.02 <= volatility <= 0.04):
        return 3  # Moderate
    if abs_ret < 0.02 and volatility < 0.02 and beta < 0.8:
        return 1  # Very Low
    if 0.02 <= abs_ret < 0.04 and beta < 0.8 and volatility < 0.02:
        return 2  # Low
    return 3

def per_stock_risk_score(df, n_quarters=8):
    """
        Returns a separate DF.
        Computes one current risk score per stock, based on n_quarters( 8 ) recent quarters of data.
        Allows this type of analysis, for example:
        AAPL usually moves about 4% after earnings: risk = 3 (moderate)
        TSLA swings wildly: risk = 5 (extreme)
    """
    results = []

    for stock, group in df.groupby("stock"):
        # Sort by earnings date (oldest → newest)
        group = group.sort_values("earnings_date").tail(n_quarters)

        if len(group) < 3:  # Only keep stocks with enough history
            continue

        # Magnitude and variability of post-earnings moves
        mean_abs_move = group["ret_3d_from_earnings"].abs().mean()
        std_abs_move = group["ret_3d_from_earnings"].abs().std()

        # Consistency between surprise direction and price move

        direction_corr = np.corrcoef(
            group["surprisePercentage"].fillna(0),
            group["ret_3d_from_earnings"].fillna(0)
        )[0, 1] if len(group) > 2 else 0

        # Typical volatility and beta
        avg_vol = group["vol_10d"].mean()
        latest_beta = group["beta_5y_monthly"].iloc[-1]

        # Weighted composite raw score
        raw_score = (
            3.0 * mean_abs_move +
            1.5 * std_abs_move +
            2.0 * avg_vol +
            1.0 * latest_beta -
            1.5 * max(direction_corr, -1)  # negative correlation → higher risk
        )

        results.append({"stock": stock, "raw_score": raw_score})

    result_df = pd.DataFrame(results).dropna(subset=["raw_score"])
    result_df = result_df[result_df["raw_score"].notna() & np.isfinite(result_df["raw_score"])]

    # rank-based quantiles instead of qcut directly to avoid bin-edge issues
    try:
        result_df["risk_score"] = pd.qcut(
            result_df["raw_score"], q=5, labels=[1, 2, 3, 4, 5]
        )
    except ValueError:
        result_df["risk_score"] = 3

    result_df["risk_score"] = result_df["risk_score"].astype(float).fillna(3).astype(int)

    return result_df