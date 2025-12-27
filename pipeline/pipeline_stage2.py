# Imports from other modules
from data_utilities.feature_engineering import (
    add_reaction_3d_10d,
    add_surprise_bucket,
    add_is_up_down_nochange,
    add_rolling_frequencies,
    add_stdev,
    add_beta_market_cap_buckets,
    add_sector_mean,
    add_relative_to_sector
  )
from data_utilities.data_processing import keep_earnings_dates_only
from config import STEP2_OUTPUT_FILE_PATH

def stage2(df):
    """ 2. Data Processing"""
    # Keep only rows of earnings report dates
    earnings_df = keep_earnings_dates_only(df)
    # Sort earnings chronologically per stock
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])
    """ 2.1 Earnings Pattern Analysis """
    earnings_df = add_reaction_3d_10d(earnings_df)
    earnings_df = add_surprise_bucket(earnings_df)
    earnings_df = add_is_up_down_nochange(earnings_df)
    earnings_df = add_rolling_frequencies(earnings_df)
    earnings_df = add_stdev(earnings_df)

    """ 2.2 Risk & Volatility Assessment """
    earnings_df = add_beta_market_cap_buckets(earnings_df)

    """ 2.3 Peer/Competitor Movements """
    earnings_df = add_sector_mean(earnings_df)
    earnings_df = add_relative_to_sector(earnings_df)

    """ Stage 2 Complete - Output CSV """
    earnings_df.to_csv(STEP2_OUTPUT_FILE_PATH, index = False)

    return earnings_df
