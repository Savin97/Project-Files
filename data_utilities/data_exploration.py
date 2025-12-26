import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to explore data for insights
def explore_data(earnings_df):
    """ Beta by Sector """
    # Beta bucket inside sectors
    beta_sector = earnings_df.groupby(["sector","beta_bucket"])[
        ["ret_10d_from_earnings"]
    ].mean()

    print(beta_sector)

    """ Sector / Sub-Sector Trends """

    """ Compare stock 3-day return to Sector 10-day Drift
    Tells you whether the stock moved with its industry or independently """
    #earnings_df['relative_to_sector'] = earnings_df['ret_3d_from_earnings'] - earnings_df['sector_drift_10d']

    # Market cap bucket inside sectors
    cap_sector = earnings_df.groupby(["sector","cap_bucket"])[
        ["ret_3d_from_earnings","ret_10d_from_earnings"]
    ].mean()

    print(cap_sector)

    """ Peer/Competitor Movements """
    # Relative to sector
    earnings_df["relative_to_sector"] = earnings_df["ret_3d_from_earnings"] - earnings_df["sector_mean_3d"]

    # Correlation of stock’s earnings reactions with sector peers
    # peer_corr = earnings_df.groupby("sector")[["ret_3d_from_earnings","sector_mean_3d"]].corr().iloc[0::2,-1]
    # More robust implementation
    peer_corr = (
        earnings_df.groupby("sector")
        .apply(lambda g: g["ret_3d_from_earnings"].corr(g["sector_mean_3d"]))
    )
    print(peer_corr)


    sector_alignment = earnings_df.groupby("sector")["relative_to_sector"].mean()
    print(sector_alignment.sort_values())

    """ Graph of Surprise vs 10-Day Return"""
    # Copy to avoid modifying original
    tmp = earnings_df[['surprisePercentage', 'ret_10d_from_earnings']].dropna().copy()

    # Remove outliers by percentile trimming
    lower_p = 1
    upper_p = 99

    low_surp, high_surp = np.percentile(tmp['surprisePercentage'], [lower_p, upper_p])
    low_ret, high_ret = np.percentile(tmp['ret_10d_from_earnings'], [lower_p, upper_p])

    clean = tmp[
        (tmp['surprisePercentage'].between(low_surp, high_surp)) &
        (tmp['ret_10d_from_earnings'].between(low_ret, high_ret))
    ]
    x = earnings_df['surprisePercentage']
    y = earnings_df['ret_10d_from_earnings']



    plt.figure(figsize=(12, 6))
    plt.scatter(clean['surprisePercentage'], clean['ret_10d_from_earnings'])
    plt.xlabel('Earnings Surprise (%)')
    plt.ylabel('10-Day Return After Earnings')
    plt.title('Surprise vs 10-Day Return (Outliers Removed)')
    plt.grid(True)
    plt.show()

def earnings_surprise_impact_analysis(earnings_df):
    """
        3A
        Outputs show a monotonic relationship - the higher the earnings surprise, the stronger the price reaction which supports that
        positive surprises lead to consistent positive price movements
    """
    print("Actual EPS vs. analyst consensus:")
    print(earnings_df[['reportedEPS', 'estimatedEPS']].describe())
    print("\n--------------------------------------------------\n")
    impact = earnings_df.groupby('surprise_bucket').agg({
        'ret_3d_from_earnings': ['mean','std'],
        'ret_10d_from_earnings': ['mean','std']
    })
    print("Stock pricing movement post-earnings:")
    print(impact)
    print("\n--------------------------------------------------\n")

    # % of cases where stock went up after earnings
    reaction = earnings_df.groupby('surprise_bucket')['ret_3d_from_earnings'].apply(
        lambda x: (x > 0).mean()
    )
    print("Upward reaction rate (3-day):")
    print(reaction.head())

    # plt.figure(figsize=(10,6), dpi=150)
    # sns.boxplot(x='surprise_bucket', y='ret_3d_from_earnings', data=earnings_df)
    # plt.title("3-Day Returns by Surprise Bucket")
    # plt.show()

    # plt.figure(figsize=(10,6), dpi=150)
    # sns.boxplot(x='surprise_bucket', y='ret_10d_from_earnings', data=earnings_df)
    # plt.title("10-Day Returns by Surprise Bucket")
    # plt.show()

def show_stock_volatility_trend(earnings_df):
    # Volatility trend (last vs. previous)
    def vol_trend(series):
        if len(series) < 2:
            return 0
        return series.iloc[-1] - series.iloc[0]

    vol_summary = (
        earnings_df.groupby("stock")
        .apply(lambda group: pd.Series({
            "vol_trend_10d": vol_trend(group.tail(8)["stdev_ret_10d"]),
            "avg_vol_10d": group.tail(8)["stdev_ret_10d"].mean()
        }))
    )

    trend_summary = earnings_df.groupby("stock").apply(
        lambda group: pd.Series({
            "Neg_to_PosSurpriseRate": group.tail(8)["neg_reaction_to_positive_surprise_10d"].mean(),
            "ConsistencyScore": group.tail(8)["consistent_10d"].mean(),
            "VolatilityTrend": vol_trend(group.tail(8)["stdev_ret_10d"]),
            "AvgVolatility": group.tail(8)["stdev_ret_10d"].mean()
        })
    ).reset_index()

    return vol_summary, trend_summary