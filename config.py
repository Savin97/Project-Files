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
OUTPUT_DIR = "outputs"
STEP1_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "earnings_df_step_1_complete.csv")
STEP2_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "earnings_df_step_2_complete.csv")
STEP3_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "earnings_df_step_3_complete.csv")
OUTPUT_DF_FULL_DETAILED_FILE_PATH = os.path.join(OUTPUT_DIR, "output_full_detailed.csv")
DASHBOARD_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "output_dashboard_ready.csv")


# === Global Parameters ===
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
START_DATE = "2020-01-01"
CUTOFF_DATE = "2025-07-02" # changed from "02-07-2025"
MIN_EARNINGS_HISTORY = 8
SURPRISE_THRESHOLD = 0.005
RETURN_DAYS_CONFIG = 10
REACTION_THRESHOLD = 0.005
VOLATILITY_THRESHOLD = 1.5 # threshold * sector volatility
MISSING_FEATURE_VALUES_THRESHOLD = 300
SIGNIFICANT_EARNINGS_SURPRISE_THRESHOLD = 0.05 # +5% or more EPS surprise
LOW_REACTION_THRESHOLD = 0.01 # price move (3-day) < 1%      
POSITIVE_SURPRISE_THRESHOLD = 0.02 # +2% or more EPS surprise
NEGATIVE_SURPRISE_THRESHOLD = -0.02 # -2% or less EPS surprise
