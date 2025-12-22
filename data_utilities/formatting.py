import pandas as pd
from config import CUTOFF_DATE, MIN_EARNINGS_HISTORY

def clean_column_names(df):
    """Standardizes column names (lowercase, stripped)."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def format_dates(df, date_column_name = "date"):
    """Convert date columns to datetime format."""
    df[date_column_name] = pd.to_datetime(df[date_column_name], dayfirst=True, errors='coerce')
    # stock_values['date'] = pd.to_datetime(stock_values['date'], dayfirst=True, errors='coerce')
    # earning_dates['earnings_date'] = pd.to_datetime(earning_dates['earnings_date'], dayfirst=True, errors='coerce')
    return df

def filter_min_history(stock_values, earning_dates):
    """Keep only stocks with at least MIN_EARNINGS_HISTORY earnings."""
    # Latest Earning report dates available
    cutoff = pd.to_datetime(CUTOFF_DATE)

    # count earnings before cutoff
    eligible = (
        earning_dates[earning_dates["earnings_date"] <= cutoff]
        .groupby("stock")
        .size()
    )

    # keep only stocks with >= MIN_EARNINGS_HISTORY earnings before cutoff
    valid_stocks = set(eligible[eligible >= MIN_EARNINGS_HISTORY].index)

    # filter both DataFrames
    stock_values = stock_values[stock_values["stock"].isin(valid_stocks)]
    earning_dates = earning_dates[earning_dates["stock"].isin(valid_stocks)]

    # First: sort both DataFrames by date (required for merge_asof)
    stock_values = stock_values.sort_values(['date']).reset_index(drop=True)
    earning_dates = earning_dates.sort_values(['earnings_date']).reset_index(drop=True)
    earning_dates = earning_dates.dropna(subset=['earnings_date']) # Handle empty earnings_date values (only 5)

    return stock_values, earning_dates
