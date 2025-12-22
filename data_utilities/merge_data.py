# data_utils/merge_data.py

import pandas as pd
import os
from config import CUTOFF_DATE, MIN_EARNINGS_HISTORY, MARKET_CAP_AND_BETA_FILE_PATH, SECTOR_FILE_PATH
from fetching_functions.api_fetch import fetch_sector_and_sub_sector_data, fetch_market_cap_and_beta
from data_utilities.formatting import clean_column_names, format_dates, filter_min_history


def merge_stocks_with_earnings(stock_values, earning_dates):
    """
        Performs merge_asof to attach next earnings_date to daily stock prices.
    """
    stock_values = stock_values.sort_values(['date']).reset_index(drop=True)
    earning_dates = earning_dates.sort_values(['earnings_date']).reset_index(drop=True)
    #earning_dates = earning_dates.dropna(subset=['earnings_date']) # Handle empty earnings_date values (only 5)

    df = pd.merge_asof(
        stock_values,
        earning_dates,
        by="stock",
        left_on="date",
        right_on="earnings_date",
        direction="forward"
    )

    return df.sort_values(['stock', 'earnings_date', 'date']).reset_index(drop=True)

def merge_eps_with_main_df(df,eps_df):
    # Clean surprisePercentage and convert to numeric
    # Make everything is str, strip % and commas etc.
    eps_df['surprisePercentage'] = (
        eps_df['surprisePercentage']
        .astype(str)
        .str.strip()
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .replace({'nan': None, 'NaN': None, '': None, '-': None})
    )

    eps_df['surprisePercentage'] = pd.to_numeric(
        eps_df['surprisePercentage'], errors='coerce'
    ) / 100.0

    # --- Make sure dates align ---
    # df's "earnings_date" looks like dd-mm-yy
    df['earnings_date'] = pd.to_datetime(df['earnings_date'], errors='coerce', dayfirst=True)
    eps_df['reportedDate'] = pd.to_datetime(eps_df['reportedDate'], errors='coerce')

    # --- Merge ---
    df_merged = df.merge(
        eps_df,
        left_on=["stock", "earnings_date"],
        right_on=["stock", "reportedDate"],
        how="left"
    )

    df_merged.drop(columns=["reportedDate"], inplace=True)
    print("DFs merged successfully!")

    return df_merged

def merge_sector_and_sub_sector_data(df):
    """## 1.7 Get Sector and Sub-sector values for each stock"""

    # Load existing mapping if available
    if os.path.exists(SECTOR_FILE_PATH):
        lookup_df = pd.read_csv(SECTOR_FILE_PATH, index_col=0)
        sector_map = lookup_df['sector'].to_dict()
        sub_sector_map = lookup_df['sub_sector'].to_dict()

    else:
        sector_map, sub_sector_map = fetch_sector_and_sub_sector_data(df)
        # Save back to disk
        lookup_df = pd.DataFrame({
            "sector": sector_map,
            "sub_sector": sub_sector_map
        })
        lookup_df.to_csv("./outputs/sector_lookup.csv")

    # # Probably redundant - CHECK! 
    # Save back to disk
    # lookup_df = pd.DataFrame({
    #     "sector": sector_map,
    #     "sub_sector": sub_sector_map
    # })

    # Map back to df
    df['sector'] = df['stock'].map(sector_map)
    df['sub_sector'] = df['stock'].map(sub_sector_map)

    # Check that length of sector and sub_sector is equal to number of stocks
    print("Number of stocks:", len(df['stock'] ) )
    print ("Empty sector values:", df['sector'].isna().sum() )
    # print( df['sector'].isna().sum()   ==  df['stock'].isna().sum()  )
    # print( df['sub_sector'].isna().sum()   ==  df['stock'].isna().sum()  )
    
    return df


def merge_market_cap_and_beta_with_df(df):
    
    if os.path.exists(MARKET_CAP_AND_BETA_FILE_PATH):
        stock_info_df = pd.read_csv(MARKET_CAP_AND_BETA_FILE_PATH)
    else:
        stock_info_df = fetch_market_cap_and_beta()
        stock_info_df.to_csv(MARKET_CAP_AND_BETA_FILE_PATH, index = False)
    # Merge back into your daily df (each row of df gets the right stock’s values)
    df = df.merge(stock_info_df, on='stock', how='left')

    return df
    