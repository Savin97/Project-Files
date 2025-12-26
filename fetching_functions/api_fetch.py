"""
    Script for fetching EPS through Alpha Vantage.
    Outputs EPS_Alpha_Vantage.csv
"""
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import os
import yfinance as yf
from config import ALPHA_VANTAGE_API_KEY, EPS_DF_FILE_PATH, START_DATE, CUTOFF_DATE

def fetch_EPS(df):
    if os.path.exists(EPS_DF_FILE_PATH):
        print("Fetching CANCELED, Using existing EPS file")
        eps_df = pd.read_csv(EPS_DF_FILE_PATH)
        print("EPS df Shape:", eps_df.shape)
        return eps_df
    
    unique_stocks = df['stock'].unique()

    all_financials = []

    for i, ticker in enumerate(unique_stocks, start=1):
        print(f"Fetching {ticker} ({i}/{len(unique_stocks)})")
        try:
            url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            # Skip error messages
            if "quarterlyEarnings" not in data:
                print(f" No data for {ticker}: {data}")
                continue

            all_financials.append(data)

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        # --- Rate limit handling ---
        if i % 70 == 0:        # after ~70 calls, pause
            print("Pausing 60s to avoid throttling...")
            time.sleep(60)
        else:
            time.sleep(1)
        

    # --- Convert to dataframe ---
    records = []
    for stock in all_financials:
        symbol = stock.get("symbol", None)
        for q in stock.get("quarterlyEarnings", []):
            records.append({
                "stock": symbol,
                "fiscalDateEnding": q.get("fiscalDateEnding"),
                "reportedDate": q.get("reportedDate"),
                "reportedEPS": q.get("reportedEPS"),
                "estimatedEPS": q.get("estimatedEPS"),
                "surprise": q.get("surprise"),
                "surprisePercentage": q.get("surprisePercentage")
            })
    eps_df = pd.DataFrame(records)

    # Ensure reportedDate is in datetime format
    eps_df["reportedDate"] = pd.to_datetime(eps_df["reportedDate"], errors="coerce")

    # Define date range
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(CUTOFF_DATE)

    # Filter rows
    eps_df = eps_df[(eps_df["reportedDate"] >= start_date) & (eps_df["reportedDate"] <= end_date)]

    eps_df.to_csv(EPS_DF_FILE_PATH, index=False)
    print("Done! EPS CSV created. Shape:", eps_df.shape)

    return eps_df


def fetch_EPS_validation(df, eps_df):
    # Check which stocks dont appear in both
    set_eps = set(eps_df['stock'].unique())
    set_df = set(df['stock'].unique())

    print("In stock_values_file but not in earning_dates:", set_eps - set_df)
    print("In earning_dates but not in df:", len(set_df - set_eps), set_df - set_eps )

    remaining_stocks = list(set_df - set_eps)

    """ Alpha Vantage API """
    api_key = "XXX9W9E3YJASNQ2A"

    all_financials = []

    for i, ticker in enumerate(remaining_stocks, start=1):
        print(f"Fetching {ticker} ({i}/{len(remaining_stocks)})")
        try:
            url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}'
            r = requests.get(url)
            data = r.json()

            # Skip error messages
            if "quarterlyEarnings" not in data:
                print(f" No data for {ticker}: {data}")
                continue

            all_financials.append(data)

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        # --- Rate limit handling ---
        if i % 70 == 0:        # after ~70 calls, pause
            print("Pausing 60s to avoid throttling...")
            time.sleep(60)
        else:
            time.sleep(1)

    # --- Convert to dataframe ---
    records = []
    for stock in all_financials:
        symbol = stock.get("symbol", None)
        for q in stock.get("quarterlyEarnings", []):
            records.append({
                "stock": symbol,
                "fiscalDateEnding": q.get("fiscalDateEnding"),
                "reportedDate": q.get("reportedDate"),
                "reportedEPS": q.get("reportedEPS"),
                "estimatedEPS": q.get("estimatedEPS"),
                "surprise": q.get("surprise"),
                "surprisePercentage": q.get("surprisePercentage")
            })

    eps_remaining_df = pd.DataFrame(records)
    # Ensure reportedDate is in datetime format
    eps_remaining_df["reportedDate"] = pd.to_datetime(eps_remaining_df["reportedDate"], errors="coerce")
    # Filter rows
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(CUTOFF_DATE)
    eps_remaining_df = eps_remaining_df[(eps_remaining_df["reportedDate"] >= start_date) & (eps_remaining_df["reportedDate"] <= end_date)]
    eps_remaining_df.to_csv("EPS_Alpha_Vantage_remaining.csv", index=False)

    return eps_remaining_df

def fetch_sector_and_sub_sector_data(df):
    sector_map, sub_sector_map = {}, {}

    unique_stocks = df['stock'].unique()

    for symbol in unique_stocks:
        if symbol in sector_map:  # already cached
            continue
        print("Now fetching the sector for", symbol)
        try:
            info = yf.Ticker(symbol).info
            sector_map[symbol] = info.get('sector')
            sub_sector_map[symbol] = info.get('industry')
        except Exception as e:
            print(f"Failed for {symbol}: {e}")
            sector_map[symbol] = None
            sub_sector_map[symbol] = None

    return sector_map, sub_sector_map

def fetch_market_cap_and_beta(df):
    unique_stocks = df['stock'].unique()
    # Fetch stock-level data
    stock_info = []
    for stock in unique_stocks:
        t = yf.Ticker(stock)
        try:
            market_cap = t.info.get('marketCap', None)
            beta       = t.info.get('beta', None)

            stock_info.append({
                'stock': stock,
                'market_cap_log': np.log(market_cap) if market_cap else None,
                'beta_5y_monthly': beta
            })
        except Exception as e:
            print(f"Error fetching {stock}: {e}")

    #Create dataframe with stock-level features
    stock_info_df = pd.DataFrame(stock_info)

    return stock_info_df

def fetch_index_beta(earnings_df):
    """
        adds beta_vs_sp500, beta_vs_nasdaq, beta_diff_sp500, beta_diff_nasdaq
        takes a lot of time so turned off right now
    """

    def get_beta(stock_ticker, index_ticker = "SPY", start = START_DATE, end = CUTOFF_DATE):
        data = yf.download([stock_ticker, index_ticker], start=start, end=end)["Close"].pct_change().dropna()
        stock_ret = data[stock_ticker]
        index_ret = data[index_ticker]

        X = sm.add_constant(index_ret)
        model = sm.OLS(stock_ret, X).fit()
        return model.params[index_ticker]

    all_stocks = earnings_df["stock"].unique()
    betas_sp500 = {}
    betas_nasdaq = {}

    for s in all_stocks:
        try:
            betas_sp500[s] = get_beta(s, "SPY")
            betas_nasdaq[s] = get_beta(s, "QQQ")
        except:
            print(f"Error getting beta for {s}")
            continue
    return betas_sp500, betas_nasdaq