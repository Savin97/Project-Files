# Imports from other modules
from data_utilities.feature_engineering import (
    add_consistent_3d_10d,
    add_past_consistency_3d_10d,
    add_confidence_score,
    add_neg_reaction_to_pos_surprise_10d,
    add_flag_volatility_3d_10d,
    add_sector_beta_features,
    add_relative_3d_10d,
    add_flag_diff_3d_10d,
    add_direction_mismatch,
    add_sector_mean_3d_same_day,
    add_sector_peer_risk_flag,
    add_risk_score,
    add_past_vol_10d
  )

from data_utilities.data_processing import handle_NA_values
from risk_scoring.scoring import per_stock_risk_score
from config import STEP3_OUTPUT_FILE_PATH

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
    earnings_df = add_past_vol_10d(earnings_df)
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
    earnings_df.to_csv(STEP3_OUTPUT_FILE_PATH, index = False)

    return earnings_df
