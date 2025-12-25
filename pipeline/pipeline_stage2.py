# Imports from other modules
from data_utilities.feature_engineering import (
    ret_3d_and_ret_10d_cols,
    daily_10d_drift_10d_vol_mom_3d,
    add_reaction_3d_10d,
    add_surprise_bucket,
    add_is_up_down_nochange,
    add_rolling_frequencies,
    add_stdev,
    add_beta_market_cap_buckets,
    add_sector_mean,
    add_relative_to_sector
  )
from data_utilities.merge_data import (
    merge_stocks_with_earnings,
    merge_eps_with_main_df,
    merge_sector_and_sub_sector_data,
    merge_market_cap_and_beta_with_df
  )
from data_utilities.formatting import (
    clean_column_names,
    format_dates,
    filter_min_history
    )
from data_utilities.data_loader import load_raw_data
from fetching_functions.api_fetch import fetch_EPS
from data_utilities.data_processing import keep_earnings_dates_only, handle_NA_values, remove_unuseful_features
from data_utilities.nlp import earnings_report_nlp_analysis
from risk_scoring.reccomendation import add_risk_recommendation

def stage2(df):
    """ 2. Data Processing"""
    # Keep only rows of earnings report dates
    earnings_df = keep_earnings_dates_only(df)
    earnings_df.to_csv("outputs/earnings_df.csv", index = False)

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
    #earnings_df.to_csv("outputs/earnings_df_step_2_complete.csv", index = False)

    return earnings_df
