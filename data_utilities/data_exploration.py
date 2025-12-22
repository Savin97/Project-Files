import numpy as np
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