# config.py
from dotenv import load_dotenv
import os
load_dotenv()

# === File Paths ===
STOCK_VALUES_FILE_PATH = "data/Earning Stocks Values.csv"
EARNING_DATES_FILE_PATH = "data/Earning_2020_2025_filled.csv"
EPS_DF_FILE_PATH = "data/EPS_Alpha_Vantage.csv"
SECTOR_FILE_PATH = "data/sector_lookup.csv"
MARKET_CAP_AND_BETA_FILE_PATH = "data/market_cap_and_beta_fetched_from_yfinance.csv"

# === Global Parameters ===
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
START_DATE = "2020-01-01"
CUTOFF_DATE = "02-07-2025"
MIN_EARNINGS_HISTORY = 8
VOL_THRESHOLD = 0.06
SURPRISE_THRESHOLD = 0.005
RETURN_DAYS_CONFIG = 10