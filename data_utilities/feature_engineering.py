# Feature Engineering
import numpy as np
import pandas as pd
from data_utilities.data_processing import classify_reaction, expected_direction

def ret_3d_and_ret_10d_cols(df):
    """ 1.6 3-day, 10-day Return Columns"""

    # Column of True/False for Earning Report Dates
    df['is_earnings'] = (df['date'] == df['earnings_date'])

    # Shift future prices by business days *within each stock*
    df = df.sort_values(['stock', 'date'])
    df['value_t+3']  = df.groupby('stock')['value'].shift(-3)  # 3 rows forward (not strictly 3 bizdays if missing)
    df['value_t+10'] = df.groupby('stock')['value'].shift(-10)

    # Compute percent of returns only for rows that are earnings dates; 3-day
    df['ret_3d_from_earnings'] = np.where(
        df['is_earnings'] == True,
        (df['value_t+3'] - df['value']) / df['value'],
        np.nan
    )

    # 10-day
    df['ret_10d_from_earnings'] = np.where(
        df['is_earnings'] == True,
        (df['value_t+10'] - df['value']) / df['value'],
        np.nan
    )

    # Clean up helper cols
    df = df.drop(columns=['value_t+3', 'value_t+10'])
    return df


def daily_10d_drift_10d_vol_mom_3d(df):
    """ 1.9 Daily returns, 10d drift and volatility, 3-day momentum """
    # Daily returns
    df['daily_ret'] = df.groupby('stock')['value'].pct_change()

    # Rolling stats
    df['drift_10d'] = df.groupby('stock')['daily_ret'].transform(lambda x: x.rolling(10).mean().shift(1)) # 10-day drift
    df['vol_10d']   = df.groupby('stock')['daily_ret'].transform(lambda x: x.rolling(10).std().shift(1)) # 10-day volatility
    df['mom_3d']    = df.groupby('stock')['daily_ret'].transform(lambda x: x.rolling(3).sum().shift(1)) # 3-day momentum

    sector_features = (
        df.groupby(['sector', 'date'])
        .agg(
            sector_drift_10d=('drift_10d', 'mean'), # AVG of all drift_10d values across every stock in the same sector on that day, percentages
            sector_vol_10d=('vol_10d', 'mean') # Typical daily volatility in the sector in the last 10 days, percentages
        )
        .groupby('sector')
        .shift(1)
        .reset_index()
    )

    df = df.merge(sector_features, on=['sector', 'date'], how='left')

    return df

def add_reaction_3d_10d(earnings_df):
    """
        Adds reaction_3d, reaction_10d
        Classify post-earnings price reactions as Up / Down / No Change,
        with a threshold of 0.5%
    """
    earnings_df["reaction_3d"] = earnings_df["ret_3d_from_earnings"].apply(classify_reaction)
    earnings_df["reaction_10d"] = earnings_df["ret_10d_from_earnings"].apply(classify_reaction)

    return earnings_df


def add_surprise_bucket(earnings_df):
    """
        Adds surprise_bucket
        Analyze relationship with EPS surprise:
        Mean 10-day return conditional on positive vs. negative surprise.
        Reaction rates (Up, Down) for different surprise buckets
    """
    # Bucket surprises into terciles
    earnings_df["surprise_bucket"] = pd.qcut(earnings_df["surprise"], 3, labels=["Low", "Mid", "High"])
    # Mean returns by surprise bucket
    #print(earnings_df.groupby("surprise_bucket")[["ret_3d_from_earnings","ret_10d_from_earnings"]].mean())
    # Frequency of Up/Down by bucket
    #print(earnings_df.groupby(["surprise_bucket","reaction_10d"]).size().unstack(fill_value=0))

    return earnings_df

def add_is_up_down_nochange(earnings_df, reaction_days = 10):
    """
        Track consistency across 8 quarters
        Adds "is_up", "is_down", "is_nochange" columns
    """
    col = f"reaction_{reaction_days}d"
    # Sort earnings chronologically per stock
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])
    # Map reactions to numeric indicators
    earnings_df["is_up"]       = (earnings_df[col] == "Up").astype(int)
    earnings_df["is_down"]     = (earnings_df[col] == "Down").astype(int)
    earnings_df["is_nochange"] = (earnings_df[col] == "No Change").astype(int)
    # earnings_df["is_nochange"] = (earnings_df[f"reaction_{reaction_days}d"] == "No Change").astype(int)

    earnings_df.drop(columns = [f"reaction_{reaction_days}d"], inplace=True)

    return earnings_df

def add_rolling_frequencies(earnings_df):
    """ 
        Rolling frequencies over last 8 quarters (excluding current via shift)
        Adds "past_up_freq", "past_down_freq", "past_nochange_freq" columns
    """
    past_up = (
        earnings_df.groupby("stock")["is_up"]
        .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    earnings_df["past_up_freq"] = past_up

    past_down = (
        earnings_df.groupby("stock")["is_down"]
        .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    earnings_df["past_down_freq"] = past_down

    past_nochange = (
        earnings_df.groupby("stock")["is_nochange"]
        .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    earnings_df["past_nochange_freq"] = past_nochange

    # Fill empty values (first quarters) with zeros. is_first_report column flags them as first quarters so should be okay.
    earnings_df[["past_up_freq","past_down_freq","past_nochange_freq"]] = (
        earnings_df[["past_up_freq","past_down_freq","past_nochange_freq"]]
        .fillna(0.0)
    )

    return earnings_df

def add_stdev(earnings_df):
    """
        adds stdev_ret_3d, stdev_ret_10d:
        High stdev_ret_3d: stock tends to have unpredictable 3-day earnings moves → riskier trade
        Low stdev_ret_3d: stock reacts more consistently (either muted or directionally stable)
    """
    # Rolling 8-quarter standard deviation of past 3d returns
    earnings_df["stdev_ret_3d"] = (
        earnings_df.groupby("stock")["ret_3d_from_earnings"]
        .transform(lambda x: x.shift().rolling(8, min_periods=2).std())
    )

    # Rolling 8-quarter standard deviation of past 10d returns
    earnings_df["stdev_ret_10d"] = (
        earnings_df.groupby("stock")["ret_10d_from_earnings"]
        .transform(lambda x: x.shift().rolling(8, min_periods=2).std())
    )

    # Handle N/A: put 0
    earnings_df["stdev_ret_3d"] = earnings_df["stdev_ret_3d"].fillna(0.0)
    earnings_df["stdev_ret_10d"] = earnings_df["stdev_ret_10d"].fillna(0.0)

    return earnings_df

def add_beta_market_cap_buckets(earnings_df):
    """
        adds beta_bucket, cap_bucket
        Sector conditioning (e.g. tech small caps vs healthcare small caps).
        Use bucketed analysis (e.g. terciles by beta/size/volatility) since raw correlations might hide nonlinear effects.
    """
    earnings_df["beta_bucket"] = pd.qcut(earnings_df["beta_5y_monthly"], 3, labels=["Low Beta", "Mid Beta", "High Beta"])
    earnings_df["cap_bucket"] = pd.qcut(earnings_df["market_cap_log"], 3, labels=["Small Cap", "Mid Cap", "Large Cap"])
    return earnings_df

def add_sector_mean(earnings_df):
    # Sector average 3-day return per event
    earnings_df["sector_mean_3d"] = (
        earnings_df.groupby("sector")["ret_3d_from_earnings"]
        .apply(lambda x: x.shift().rolling(8, min_periods=2).mean())
        .reset_index(level=0, drop=True)
    )

    earnings_df["sector_mean_10d"] = (
        earnings_df.groupby("sector")["ret_10d_from_earnings"]
        .apply(lambda x: x.shift().rolling(8, min_periods=2).mean())
        .reset_index(level=0, drop=True)
    )
    return earnings_df

def add_relative_to_sector(earnings_df):
    # Relative to sector
    earnings_df["relative_to_sector"] = earnings_df["ret_3d_from_earnings"] - earnings_df["sector_mean_3d"]
    return earnings_df

def add_consistent_3d_10d(earnings_df):
    """
        Adds consistent_3d, consistent_10d columns
    """
    earnings_df["expected"] = earnings_df["surprise"].apply(expected_direction)
    """ 
        If surprise is above threshold: Returns 1
        If its below the negative threshold: -1
        If its without change: 0
    """
    # Consistency check for 3d and 10d reactions
    earnings_df["consistent_3d"] = (earnings_df["reaction_3d"] == earnings_df["expected"]).astype(int)
    earnings_df["consistent_10d"] = (earnings_df["reaction_10d"] == earnings_df["expected"]).astype(int)
    return earnings_df

def add_past_consistency_3d_10d(earnings_df):
    """
        How often a stock reacts in the same direction as expected based on earnings results.
        Takes last 8 earning report directions, returns mean.
    """
    earnings_df["past_consistency_10d"] = (
        earnings_df.groupby("stock")["consistent_10d"]
        .apply(lambda x: x.shift().rolling(8, min_periods=4).mean())
        .reset_index(level=0, drop=True)
    )

    earnings_df["past_consistency_3d"] = (
        earnings_df.groupby("stock")["consistent_3d"]
        .apply(lambda x: x.shift().rolling(8, min_periods=4).mean())
        .reset_index(level=0, drop=True)
    )
    return earnings_df

def add_confidence_score(earnings_df):
    """
        Adds confidence_score:
        Range: 0-1
        ≈ 1.0 → historically very reliable (high confidence)
        ≈ 0.5 → mixed history (medium confidence)
        ≈ 0.0 → usually wrong (low confidence)
    """
    # weighted combination of recent accuracies
    earnings_df["confidence_score"] = (
        0.4 * earnings_df["past_consistency_3d"].fillna(0) +
        0.6 * earnings_df["past_consistency_10d"].fillna(0)
    )
    return earnings_df

def add_neg_reaction_to_pos_surprise_10d(earnings_df):
    """
        Compares past 8 earnings reports
        adds neg_reaction_to_positive_surprise_10d, past_neg_reaction_to_positive_surprise_10d
    """
    # Flag negative reaction to positive surprise
    earnings_df["neg_reaction_to_positive_surprise_10d"] = (
        (earnings_df["surprise"] > 0) & (earnings_df["reaction_10d"] == -1)
    ).astype(int)

    # Rolling frequency over last 8 reports
    earnings_df["past_neg_reaction_to_positive_surprise_10d"] = (
        earnings_df.groupby("stock")["neg_reaction_to_positive_surprise_10d"]
        .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    # UNUSED 3-day version
    # earnings_df["neg_reaction_to_positive_surprise_3d"] = (
    #     (earnings_df["surprise"] > 0) & (earnings_df["reaction_3d"] == "Down")
    # ).astype(int)

    # # Rolling frequency over last 8 reports
    # earnings_df["past_neg_reaction_to_positive_surprise_3d"] = (
    #     earnings_df.groupby("stock")["neg_reaction_to_positive_surprise_3d"]
    #     .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
    #     .reset_index(level=0, drop=True)
    # )
    return earnings_df

    

def add_past_vol_10d(earnings_df):
    """
        Adds past_vol_10d
    """
    # Volatility around earnings
    # Rolling average volatility
    earnings_df["past_vol_10d"] = (
        earnings_df.groupby("stock")["stdev_ret_10d"]
        .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return earnings_df

def add_flag_volatility_3d_10d(earnings_df):
    """
        Flag cases where the move exceeds 2 x past STDEV
        returns 0/1
    """
    # For 3d moves
    earnings_df["flag_volatility_3d"] = (
        (earnings_df["ret_3d_from_earnings"].abs() >
        2 * earnings_df["stdev_ret_3d"]).astype(int)
    )

    # For 10d moves
    earnings_df["flag_volatility_10d"] = (
        (earnings_df["ret_10d_from_earnings"].abs() >
        2 * earnings_df["stdev_ret_10d"]).astype(int)
    )

    # Optional - create a single summary flag if you want just one flag regardless of horizon:
    # earnings_df["flag_volatility"] = (
    #     earnings_df[["flag_volatility_3d", "flag_volatility_10d"]].max(axis=1)
    # )
    return earnings_df

def add_sector_beta_features(earnings_df):
    """
        Adds sector_beta, beta_diff_sector
        Compares stock beta to sector beta
    """

    # Sector-level beta (per earnings date)
    earnings_df["sector_beta"] = (
        earnings_df.groupby(["sector", "earnings_date"])["beta_5y_monthly"]
        .transform("mean")
    )
    # Compute comparisons to the sector
    earnings_df["beta_diff_sector"]  = earnings_df["beta_5y_monthly"] - earnings_df["sector_beta"]
    return earnings_df

def add_relative_3d_10d(earnings_df):
    """
        Identify stocks that behave differently from peers post-earnings.
        Adds relative_10d (and 3d) - performance differential compared to 3d/10d sector mean
        """
    # Calculate relative performance
    earnings_df["relative_3d"] = earnings_df["ret_3d_from_earnings"] - earnings_df["sector_mean_3d"]
    earnings_df["relative_10d"] = earnings_df["ret_10d_from_earnings"] - earnings_df["sector_mean_10d"]
    return earnings_df

def add_flag_diff_3d_10d(earnings_df):
    """ Adds flag_diff_10d (and 3d)
    Flags unusually large post-earnings moves compared to their sector's typical volatility - detects outlier reactions """

    # Flag anomalies
    earnings_df["flag_diff_3d"] = (earnings_df["relative_3d"].abs() > VOLATILITY_THRESHOLD * earnings_df["sector_vol_10d"]).astype(int)
    earnings_df["flag_diff_10d"] = (earnings_df["relative_10d"].abs() > VOLATILITY_THRESHOLD * earnings_df["sector_vol_10d"]).astype(int)
    return earnings_df

def add_direction_mismatch(earnings_df):
    """ Adds direction_mismatch
    Meaning, returns are up while sector's returns are down and vice versa"""
    # Direction mismatch
    earnings_df["direction_mismatch"] = (
        ((earnings_df["ret_3d_from_earnings"] > 0) & (earnings_df["sector_mean_3d"] < 0)) |
        ((earnings_df["ret_3d_from_earnings"] < 0) & (earnings_df["sector_mean_3d"] > 0))
    ).astype(int)
    return earnings_df

