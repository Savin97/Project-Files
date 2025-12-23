# main.py
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
from data_utilities.data_processing import keep_earnings_dates_only
import warnings

def main():
    warnings.filterwarnings('ignore')
    """ 1.1 Load datasets """
    stock_values, earning_dates = load_raw_data()
    print("\n\n\nData loaded successfully.\n")
    
    """ 1.2 Formatting"""
    stock_values = clean_column_names(stock_values)
    stock_values = format_dates(stock_values)

    # Standardize column names - stock, date in both
    earning_dates = clean_column_names(earning_dates)
    earning_dates = earning_dates.rename(columns={'earnings release date': 'earnings_date', 'symbol': 'stock'})
    earning_dates = format_dates(earning_dates, "earnings_date")

    """ 1.3 Only keep stocks that have at least 8 earnings report dates in BOTH files """
    stock_values,earning_dates = filter_min_history(stock_values,earning_dates) # This function doesn't do exactly what i want, fix to make sure last 8 quarters are present

    """ 1.4 Merge stock values and earning dates """
    df = merge_stocks_with_earnings(stock_values,earning_dates)
    
    """ 1.5 Fetch EPS with Alpha Vantage, Merge with main df """
    eps_df = fetch_EPS(df)
    df = merge_eps_with_main_df(df, eps_df)

    """ 1.6 3-day, 10-day Return Columns """
    df = ret_3d_and_ret_10d_cols(df)

    """ 1.7 Get Sector and Sub-sector values for each stock """
    df = merge_sector_and_sub_sector_data(df)

    """ 1.8 Adding market cap and beta """
    df = merge_market_cap_and_beta_with_df(df)

    """ 1.9 Daily returns, 10d drift and volatility, 3-day momentum """
    df = daily_10d_drift_10d_vol_mom_3d(df)

    """ Stage 1 Complete - Output CSV """
    #df.to_csv("DF_step_1_complete.csv", index = False)
    print("Stage 1 Complete! CSV created.")

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
    earnings_df.to_csv("outputs/earnings_df.csv", index = False)

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
    

    print("\nDone.\n\n\n")

if __name__ == "__main__":
    main()