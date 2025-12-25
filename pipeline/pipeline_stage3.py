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

def stage3(earnings_df):
    """ 3. Risk Assessment Methodology """
    """ 3A. Earnings-Based Risk Factors """

    """ Earnings Reaction Consistency Score """
    # Potential fix needed - min period = 8 Causes lots of empties in past_consistency!

    # Adds consistent_3d, consistent_10d columns: Stock direction - Actual vs Expected
    earnings_df = add_consistent_3d_10d(earnings_df)
    # How often a stock reacts in the same direction
    earnings_df = add_past_consistency_3d_10d(earnings_df)
    # Adds a confidence score - Range: 0-1
    earnings_df = add_confidence_score(earnings_df)

    """ Earnings Trend & Risk Indicator """
    earnings_df = add_neg_reaction_to_pos_surprise_10d(earnings_df)

    """ 3B. Volatility & Market Risk Factors """
    """ Historical Earnings Volatility Range """
    earnings_df = add_flag_volatility_3d_10d(earnings_df)
    """ Beta & Systemic Risk """
    earnings_df = add_sector_beta_features(earnings_df)
    """ Fetch and merge index betas. TODO: FIX to cache results, ADD server error handling. 
    TODO: PROBABBLY CHANGE to Sector betas.Takes a lot of time """
    # earnings_df = merge_index_beta_with_df(earnings_df)

    """ Identify stocks that behave differently from peers post-earnings """
    earnings_df = add_relative_3d_10d(earnings_df)
    earnings_df = add_flag_diff_3d_10d(earnings_df)
    earnings_df = add_direction_mismatch(earnings_df)

    """ Sector & Peer Performance Risk """
    earnings_df = add_sector_mean_3d_same_day(earnings_df)
    # Flag Sector & Peer Performance Risk cases
    earnings_df = add_sector_peer_risk_flag(earnings_df)
    
    """ IN TESTING - Earnings Call Sentiment Analysis """
    # processed_nlp = nlp()

    """ 3C. Risk Scoring System """
    earnings_df = handle_NA_values(earnings_df)

    """ Risk scoring per earning report """
    earnings_df = add_risk_score(earnings_df)

    """ TODO: NOT IMPLEMENTED YET. Risk score for each stock based on last 8 quarters """
    per_stock_score_df = per_stock_risk_score(earnings_df)

    """ Step 3 Complete CSV """
    earnings_df.to_csv("outputs/earnings_df_step_3_complete.csv", index = False)
