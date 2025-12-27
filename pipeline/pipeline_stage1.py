import warnings
# Imports from other modules
from data_utilities.feature_engineering import (
    ret_3d_and_ret_10d_cols,
    daily_10d_drift_10d_vol_mom_3d
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
from config import STEP1_OUTPUT_FILE_PATH

def load_and_format_data():
    warnings.filterwarnings('ignore')
    
    """ Stage 1 """

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
    return df

def add_stage_1_features(df):
    """ 1.6 3-day, 10-day Return Columns """
    df = ret_3d_and_ret_10d_cols(df)

    """ 1.7 Get Sector and Sub-sector values for each stock """
    df = merge_sector_and_sub_sector_data(df)

    """ 1.8 Adding market cap and beta """
    df = merge_market_cap_and_beta_with_df(df)

    """ 1.9 Daily returns, 10d drift and volatility, 3-day momentum """
    df = daily_10d_drift_10d_vol_mom_3d(df)

    """ Stage 1 Complete - Output CSV """
    df.to_csv(STEP1_OUTPUT_FILE_PATH, index = False)
    # print("Stage 1 Complete! CSV created.")
    return df