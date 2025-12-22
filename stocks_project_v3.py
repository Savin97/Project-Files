# 0. Imports
import pandas as pd
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

earnings_df = []

"""# 3. Risk Assessment Methodology

## A. Earnings-Based Risk Factors

### Earnings Surprise Impact Analysis
"""

"""
    Outputs show a monotonic relationship - the higher the earnings surprise, the stronger the price reaction which supports that
    positive surprises lead to consistent positive price movements
"""
print("Actual EPS vs. analyst consensus:")
print(earnings_df[['reportedEPS', 'estimatedEPS']].describe())
print("\n--------------------------------------------------\n")
impact = earnings_df.groupby('surprise_bucket').agg({
    'ret_3d_from_earnings': ['mean','std'],
    'ret_10d_from_earnings': ['mean','std']
})
print("Stock pricing movement post-earnings:")
print(impact)
print("\n--------------------------------------------------\n")

# % of cases where stock went up after earnings
reaction = earnings_df.groupby('surprise_bucket')['ret_3d_from_earnings'].apply(
    lambda x: (x > 0).mean()
)
print("Upward reaction rate (3-day):")
print(reaction.head())

# plt.figure(figsize=(10,6), dpi=150)
# sns.boxplot(x='surprise_bucket', y='ret_3d_from_earnings', data=earnings_df)
# plt.title("3-Day Returns by Surprise Bucket")
# plt.show()

# plt.figure(figsize=(10,6), dpi=150)
# sns.boxplot(x='surprise_bucket', y='ret_10d_from_earnings', data=earnings_df)
# plt.title("10-Day Returns by Surprise Bucket")
# plt.show()

"""### Earnings Reaction Consistency Score

### !!min period = 8 Causes lots of empties in past_consistency!!
"""

"""
    adds consistent_3d, past_consistent... columns
    How often a stock reacts in the same direction as expected
    based on earnings results

    adds confidence_score:
    Range: 0-1
    ≈ 1.0 → historically very reliable (high confidence)
    ≈ 0.5 → mixed history (medium confidence)
    ≈ 0.0 → usually wrong (low confidence)
"""

# Expected reaction based on earnings surprise
def expected_direction(surprise, threshold=0.005):
    if surprise > threshold:
        return "Up"
    elif surprise < -threshold:
        return "Down"
    else:
        return "No Change"

earnings_df["expected"] = earnings_df["surprise"].apply(expected_direction)

# Consistency check for 3d and 10d reactions
earnings_df["consistent_3d"] = (earnings_df["reaction_3d"] == earnings_df["expected"]).astype(int)
earnings_df["consistent_10d"] = (earnings_df["reaction_10d"] == earnings_df["expected"]).astype(int)

earnings_df["past_consistency_10d"] = (
    earnings_df.groupby("stock")["consistent_10d"]
    .apply(lambda x: x.shift().rolling(8, min_periods=4).mean())
    .reset_index(level=0, drop=True)
)

earnings_df["past_consistency_3d"] = (
    earnings_df.groupby("stock")["consistent_3d"]
    .apply(lambda x: x.shift().rolling(8, min_periods=4).mean())
    .reset_index(level=0, drop=True)
)

# weighted combination of recent accuracies
earnings_df["confidence_score"] = (
    0.4 * earnings_df["past_consistency_3d"].fillna(0) +
    0.6 * earnings_df["past_consistency_10d"].fillna(0)
)

"""### Earnings Trend & Risk Indicator"""

"""
    Compares past 8 earnings reports
    adds neg_reaction_to_positive_surprise_10d, past_neg_reaction_to_positive_surprise_10d, past_vol_10d
"""

# Flag negative reaction to positive surprise
earnings_df["neg_reaction_to_positive_surprise_10d"] = (
    (earnings_df["surprise"] > 0) & (earnings_df["reaction_10d"] == "Down")
).astype(int)

# Rolling frequency over last 8 reports
earnings_df["past_neg_reaction_to_positive_surprise_10d"] = (
    earnings_df.groupby("stock")["neg_reaction_to_positive_surprise_10d"]
    .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)

# UNUSED 3-day version
# earnings_df["neg_reaction_to_positive_surprise_3d"] = (
#     (earnings_df["surprise"] > 0) & (earnings_df["reaction_3d"] == "Down")
# ).astype(int)

# # Rolling frequency over last 8 reports
# earnings_df["past_neg_reaction_to_positive_surprise_3d"] = (
#     earnings_df.groupby("stock")["neg_reaction_to_positive_surprise_3d"]
#     .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
#     .reset_index(level=0, drop=True)
# )

"""
    adds past_vol_10d
"""
# Volatility around earnings
# Rolling average volatility
earnings_df["past_vol_10d"] = (
    earnings_df.groupby("stock")["stdev_ret_10d"]
    .apply(lambda x: x.shift().rolling(8, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)

# Volatility trend (last vs. previous)
def vol_trend(series):
    if len(series) < 2:
        return 0
    return series.iloc[-1] - series.iloc[0]

vol_summary = (
    earnings_df.groupby("stock")
    .apply(lambda group: pd.Series({
        "vol_trend_10d": vol_trend(group.tail(8)["stdev_ret_10d"]),
        "avg_vol_10d": group.tail(8)["stdev_ret_10d"].mean()
    }))
)

trend_summary = earnings_df.groupby("stock").apply(
    lambda group: pd.Series({
        "Neg_to_PosSurpriseRate": group.tail(8)["neg_reaction_to_positive_surprise_10d"].mean(),
        "ConsistencyScore": group.tail(8)["consistent_10d"].mean(),
        "VolatilityTrend": vol_trend(group.tail(8)["stdev_ret_10d"]),
        "AvgVolatility": group.tail(8)["stdev_ret_10d"].mean()
    })
).reset_index()

"""## B. Volatility & Market Risk Factors

### Historical Earnings Volatility Range
"""

"""
    Flag cases where the move exceeds 2 x past STDEV
    returns 0/1
"""

# For 3d moves
earnings_df["flag_volatility_3d"] = (
    (earnings_df["ret_3d_from_earnings"].abs() >
     2 * earnings_df["stdev_ret_3d"]).astype(int)
)

# For 10d moves
earnings_df["flag_volatility_10d"] = (
    (earnings_df["ret_10d_from_earnings"].abs() >
     2 * earnings_df["stdev_ret_10d"]).astype(int)
)

# Optional - create a single summary flag if you want just one flag regardless of horizon:
# earnings_df["flag_volatility"] = (
#     earnings_df[["flag_volatility_3d", "flag_volatility_10d"]].max(axis=1)
# )

"""### Beta & Systemic Risk"""

"""
    adds sector_beta, beta_diff_sector
    Compares stock beta to sector beta
"""

# Sector-level beta (per earnings date)
earnings_df["sector_beta"] = (
    earnings_df.groupby(["sector", "earnings_date"])["beta_5y_monthly"]
    .transform("mean")
)
# Compute comparisons to the sector
earnings_df["beta_diff_sector"]  = earnings_df["beta_5y_monthly"] - earnings_df["sector_beta"]

"""Code for Fetching and Merging Sector Betas"""

"""
    adds beta_vs_sp500, beta_vs_nasdaq, beta_diff_sp500, beta_diff_nasdaq
    takes a lot of time so turned off right now
"""

# def get_beta(stock_ticker, index_ticker="SPY", start="2020-01-01", end="2025-07-02"):
#     data = yf.download([stock_ticker, index_ticker], start=start, end=end)["Close"].pct_change().dropna()
#     stock_ret = data[stock_ticker]
#     index_ret = data[index_ticker]

#     X = sm.add_constant(index_ret)
#     model = sm.OLS(stock_ret, X).fit()
#     return model.params[index_ticker]

# all_stocks = earnings_df["stock"].unique()
# betas_sp500 = {}
# betas_nasdaq = {}

# for s in all_stocks:
#     try:
#         betas_sp500[s] = get_beta(s, "SPY")
#         betas_nasdaq[s] = get_beta(s, "QQQ")
#     except:
#         print(f"Error getting beta for {s}")
#         continue

# # Merge into the DataFrame
# earnings_df["beta_vs_sp500"] = earnings_df["stock"].map(betas_sp500)
# earnings_df["beta_vs_nasdaq"] = earnings_df["stock"].map(betas_nasdaq)

# # Compute comparisons to the S&P500 and Nasdaq
# earnings_df["beta_diff_sp500"]   = earnings_df["beta_5y_monthly"] - earnings_df["beta_vs_sp500"]
# earnings_df["beta_diff_nasdaq"]  = earnings_df["beta_5y_monthly"] - earnings_df["beta_vs_nasdaq"]

"""
    Identify stocks that behave differently from peers post-earnings
    adds:
    relative_10d (and 3d) - performance differential
    flag_diff_10d (and 3d) - Flags unusually large post-earnings moves compared to their sector’s typical volatility - detects outlier reactions
    direction_mismatch - returns are up while sector's returns are down and vice versa
"""

# Calculate relative performance
earnings_df["relative_3d"] = earnings_df["ret_3d_from_earnings"] - earnings_df["sector_mean_3d"]
earnings_df["relative_10d"] = earnings_df["ret_10d_from_earnings"] - earnings_df["sector_mean_10d"]

# Flag anomalies
volatility_threshold = 1.5     # threshold * sector volatility
earnings_df["flag_diff_3d"] = (earnings_df["relative_3d"].abs() > volatility_threshold * earnings_df["sector_vol_10d"]).astype(int)
earnings_df["flag_diff_10d"] = (earnings_df["relative_10d"].abs() > volatility_threshold * earnings_df["sector_vol_10d"]).astype(int)

# Direction mismatch
earnings_df["direction_mismatch"] = (
    ((earnings_df["ret_3d_from_earnings"] > 0) & (earnings_df["sector_mean_3d"] < 0)) |
    ((earnings_df["ret_3d_from_earnings"] < 0) & (earnings_df["sector_mean_3d"] > 0))
).astype(int)

"""### Sector & Peer Performance Risk:

"""

"""
    adds sector_mean_3d_same_day - Average sector return per earnings date
    sector_peer_risk_flag -

    Flag Sector & Peer Performance Risk cases
    sector_peer_risk_flag - A “Sector & Peer Performance Risk” case happens when:
    - The stock is Down after earnings
    - But its sector peers gained (sector_mean_3d > 0).

    Gives a column identifying all stocks that fell while peers rose - 1 if Flagged, 0 if not.
"""

# Average sector return per earnings date
sector_performance = (
    earnings_df.groupby(['sector', 'earnings_date'])['ret_3d_from_earnings']
    .mean()
    .reset_index()
    .rename(columns={'ret_3d_from_earnings': 'sector_mean_3d_same_day'})
)

# Merge back into main DataFrame
earnings_df = earnings_df.merge(sector_performance, on=['sector', 'earnings_date'], how='left')

threshold=0.005
earnings_df["sector_peer_risk_flag"] = (
    (earnings_df["ret_3d_from_earnings"] < -threshold) &
    (earnings_df["sector_mean_3d_same_day"] > threshold)
).astype(int)

# Optional - Quantify severity - how much the stock underperformed peers
#earnings_df["sector_underperf"] = earnings_df["relative_to_sector"].apply(lambda x: -x if x < 0 else 0)

"""### TESTING - Earnings Call Sentiment Analysis"""

# """
#     Perform NLP analysis on earnings call transcripts.
#     Identify spikes in negative phrases that correlate with stock price drops.
# """
# with open("AAPL_2024Q1.txt", "r", encoding="utf-8") as f:
#     raw = f.read()


# def clean_earningscall_biz(text):
#     # Remove "speaker ..." labels entirely
#     text = re.sub(r"\bspeaker[\s\.\w]*?(?=\b[a-z]|$)", " ", text, flags=re.IGNORECASE)

#     # Remove job titles and company names after speaker lines
#     text = re.sub(r"\b(ceo|cfo|analyst|operator|director of investor relations|at|from)\b", "", text, flags=re.IGNORECASE)

#     # Remove leftover periods between single words (like "apple. vision. pro.")
#     text = re.sub(r"\b([a-zA-Z])\.\s(?=[a-zA-Z])", r"\1 ", text)
#     text = re.sub(r"(\b\w+)\.\s(\w+\b)", r"\1 \2", text)

#     # Collapse repeated periods or single-letter dots
#     text = re.sub(r"\s*\.\s*", ". ", text)
#     text = re.sub(r"\.{2,}", ".", text)

#     # Clean unwanted symbols and extra spaces
#     text = re.sub(r"[^a-zA-Z0-9,\.\?\!\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     return text
# with open("AAPL_2024Q1.txt", "r", encoding="utf-8") as f:
#     raw = f.read()

# cleaned = clean_earningscall_biz(raw)

# with open("AAPL_2024Q1_cleaned_fixed.txt", "w", encoding="utf-8") as f:
#     f.write(cleaned)

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')

# sia = SentimentIntensityAnalyzer()
# score = sia.polarity_scores(cleaned)
# print(score)

# sentences = cleaned.split(".")
# sentences_clean = [s.strip() for s in sentences if s.strip()]
# scores = [sia.polarity_scores(s)["compound"] for s in sentences if s.strip()]
# print(len(sentences))
# sent_df = pd.DataFrame({"sentence": sentences_clean, "score": scores})
# #sent_df.to_csv("report.csv", index = False)
# print("Min sentiment:", sent_df["score"].min())

# """
#     These two metrics (avg_sent, neg_ratio) are what i can later merge
#     into my earnings_df by stock/quarter and compare with ret_3d_from_earnings.
# """

# neg_ratio = (sent_df["score"] < -0.2).mean()
# avg_sent = sent_df["score"].mean()
# print(f"Average sentiment: {avg_sent:.3f}")
# print(f"Percent negative sentences: {neg_ratio*100:.1f}%")

"""## C. Risk Scoring System

### Handle N/A values
"""

# # Check N/A amount
# print("Columns that have N/A values:\n")
# for column in earnings_df.columns:
#     n_missing = earnings_df[column].isna().sum()
#     if n_missing > 0:
#         print(f"{column}: {n_missing}")

# 1. Columns to drop completely (too few or not useful)
drop_cols = ["company name", "fiscalDateEnding"]   # fill if needed

# 2. Categorical columns → fill with mode
categorical_cols = ["sector", "sub_sector", "surprise_bucket", "beta_bucket", "cap_bucket"]

# 3. Numeric columns → fill with group mean or global mean
num_cols = [
    "reportedEPS", "estimatedEPS", "surprise", "surprisePercentage",
    "ret_3d_from_earnings", "ret_10d_from_earnings", "market_cap_log",
    "beta_5y_monthly", "drift_10d", "vol_10d", "sector_drift_10d", "sector_vol_10d",
    "sector_mean_3d", "sector_mean_10d", "relative_to_sector",
    "past_neg_reaction_to_positive_surprise_10d", "past_vol_10d", "sector_beta",
    "relative_3d", "relative_10d", "sector_mean_3d_same_day","past_consistency_10d", "past_consistency_3d", "beta_diff_sector"
]

# Fill sector-based numeric columns using sector means
sector_fill_cols = [
    "sector_mean_3d", "sector_mean_10d", "sector_drift_10d",
    "sector_vol_10d", "sector_beta"
]

earnings_df = earnings_df.drop( columns = drop_cols )

# Fill categorical N/As with mode
for col in categorical_cols:
    if col in earnings_df.columns:
        mode_val = earnings_df[col].mode().iloc[0]
        earnings_df[col].fillna(mode_val, inplace=True)

# Fill sector columns with sector mean
for col in sector_fill_cols:
    if col in earnings_df.columns:
        earnings_df[col].fillna(
            earnings_df.groupby("sector")[col].transform("mean"),
            inplace=True
        )

# Fill remaining numeric columns with global mean
for col in num_cols:
    #if col in earnings_df.columns and earnings_df[col].isna().sum() > 0:
    earnings_df[col].fillna(earnings_df[col].mean(), inplace=True)

# # past_consistency_10d, past_consistency_3d - fill with the sector mean
# earnings_df["past_consistency_10d"].fillna(
#     earnings_df.groupby("sector")["past_consistency_10d"].transform("mean"),
#     inplace=True
# )
# earnings_df["past_consistency_3d"].fillna(
#     earnings_df.groupby("sector")["past_consistency_3d"].transform("mean"),
#     inplace=True
# )

#earnings_df["past_consistency_10d"].fillna(earnings_df["past_consistency_10d"].mean(), inplace=True)

# Removing columns if they have a large number of NaN values
# column_values_missing_threshold = 300
# for col in earnings_df.columns:
#     num_missing = earnings_df[col].isna().sum()
#     if num_missing > column_values_missing_threshold:
#     # Drop column entirely if there are more than 300 missing
#         earnings_df = earnings_df.drop(columns=[col])
#     # Drop rows with NaN if there are fewer than 300 missing
#     else:
#         earnings_df = earnings_df.dropna(subset=[col])

print("Columns that have N/A values:\n")
for column in earnings_df.columns:
    n_missing = earnings_df[column].isna().sum()
    if n_missing > 0:
        print(f"{column}: {n_missing}")

"""### Risk Score per earning report


"""

"""
    Risk	                                  Behavior	                                                        Quantitative rule

    1 (Very Low)             stable, low beta & volatility                    abs_ret < 0.02 and vol < 0.02 and beta < 0.8
    2 (Low)	                   stable, low beta & volatility                  	abs(ret_10d) < 0.04, beta < 0.8, vol_10d < 0.02
    3 (Moderate)    	    occasional inconsistency                        abs(ret_10d) between 2-5%, or moderate volatility (0.02-0.04)
    4 (High)	              frequent strong reactions	                        abs(ret_10d) > 0.05 or ret_10d opposite sign of surprise
    5 (Extreme)	                wild movements                                   abs(ret_10d) > 0.1 or very high volatility vol_10d > 0.06

"""

def risk_score(row):
    abs_ret = abs(row['ret_10d_from_earnings'])
    beta = row['beta_5y_monthly']
    vol = row['vol_10d']
    surp = row['surprisePercentage']
    ret10 = row['ret_10d_from_earnings']

    if abs_ret > 0.10 or vol > 0.06:
        return 5  # Extreme
    if (surp > 0 and ret10 < 0) or (surp < 0 and ret10 > 0):
        return 4  # High – opposite reaction to surprise
    if (0.02 <= abs_ret <= 0.05) or (0.02 <= vol <= 0.04):
        return 3  # Moderate
    if abs_ret < 0.02 and vol < 0.02 and beta < 0.8:
        return 1  # Very Low
    if 0.02 <= abs_ret < 0.04 and beta < 0.8 and vol < 0.02:
        return 2  # Low
    return 3


# earnings_df["reaction_consistency"] = (
#     earnings_df.groupby("stock")["ret_3d_from_earnings"]
#     .transform(lambda x: x.rolling(4).std())
# )

earnings_df["risk_score"] = earnings_df.apply(risk_score, axis=1)

"""### Risk score for each stock based on last 8 quarters"""

"""
    Compute one current risk score per stock, based on n_quarters( 8 ) recent quarters of data.
    Allows this type of analysis, for example:
    AAPL usually moves about 4% after earnings: risk = 3 (moderate)
    TSLA swings wildly: risk = 5 (extreme)
"""

def per_stock_risk_score(df, n_quarters=8):
    results = []

    for stock, group in df.groupby("stock"):
        # Sort by earnings date (oldest → newest)
        group = group.sort_values("earnings_date").tail(n_quarters)

        if len(group) < 3:  # Only keep stocks with enough history
            continue

        # Magnitude and variability of post-earnings moves
        mean_abs_move = group["ret_3d_from_earnings"].abs().mean()
        std_abs_move = group["ret_3d_from_earnings"].abs().std()

        # Consistency between surprise direction and price move

        direction_corr = np.corrcoef(
            group["surprisePercentage"].fillna(0),
            group["ret_3d_from_earnings"].fillna(0)
        )[0, 1] if len(group) > 2 else 0

        # Typical volatility and beta
        avg_vol = group["vol_10d"].mean()
        latest_beta = group["beta_5y_monthly"].iloc[-1]

        # Weighted composite raw score
        raw_score = (
            3.0 * mean_abs_move +
            1.5 * std_abs_move +
            2.0 * avg_vol +
            1.0 * latest_beta -
            1.5 * max(direction_corr, -1)  # negative correlation → higher risk
        )

        results.append({"stock": stock, "raw_score": raw_score})

    result_df = pd.DataFrame(results).dropna(subset=["raw_score"])
    result_df = result_df[result_df["raw_score"].notna() & np.isfinite(result_df["raw_score"])]

    # rank-based quantiles instead of qcut directly to avoid bin-edge issues
    try:
        result_df["risk_score"] = pd.qcut(
            result_df["raw_score"], q=5, labels=[1, 2, 3, 4, 5]
        )
    except ValueError:
        result_df["risk_score"] = 3

    result_df["risk_score"] = result_df["risk_score"].astype(float).fillna(3).astype(int)

    return result_df

# Example usage:
risk_df = per_stock_risk_score(earnings_df)
#risk_df.to_csv("risk_scores.csv", index=False)

"""## Step 3 Complete CSV"""

earnings_df.to_csv("./outputs/DF_step_3_complete.csv", index = False)

"""# 4. Outputs - blocks here appear twice - the second ones are old and may be redundant

## Pre-Earnings Insights

###  Recommendations based on risk score
"""

#earnings_df = pd.read_csv("DF_step_3_complete.csv")

"""
    Sell / Sell 50% / Hold
"""
def recommendation_from_risk(score):
    if score >= 4.0:
        return "SELL"
    elif score >= 3.0:
        return "SELL 50%"
    else:
        return "HOLD"

# Filter for Upcoming Earnings Only
# today = datetime.today()
# future_window = timedelta(days=100)  # or whatever window you want
# upcoming = earnings_df[
#     (earnings_df['earnings_date'] >= today) &
#     (earnings_df['earnings_date'] <= today + future_window)
# ]

# Select relevant columns from earnings_df
output_cols = [
    'stock',
    'earnings_date',
    'risk_score',
    'sector',
    'sub_sector',
    'beta_5y_monthly',
    'vol_10d',
    'sector_vol_10d',
    'sector_peer_risk_flag'
]

output_df = earnings_df[output_cols].copy()

output_df['risk_recommendation'] = output_df['risk_score'].apply(recommendation_from_risk)
output_df = output_df.sort_values(['earnings_date', 'stock'])

"""### Explanation of Decision

#### Competitor Earnings Influence
"""

# --- Define grouping key ---
group_key = 'sub_sector'

# --- Derive competitor metrics (without editing earnings_df) ---
tmp = earnings_df.copy()

# Peer 3-day avg after earnings
tmp['peer_avg_3d'] = (
    tmp.groupby([group_key, 'earnings_date'])['ret_3d_from_earnings']
       .transform('mean')
)

# Rolling competitor trend (8-event window)
tmp['competitor_trend'] = (
    tmp.groupby(group_key)['peer_avg_3d']
       .transform(lambda x: x.rolling(8, min_periods=2).mean())
)

# Relative performance vs competitors
tmp['relative_to_competitors'] = tmp['ret_3d_from_earnings'] - tmp['competitor_trend']

# Sensitivity: correlation between own reaction and peer trend
def competitor_sensitivity(group):
    if len(group) < 4:
        return np.nan
    return group['ret_3d_from_earnings'].corr(group['competitor_trend'])

competitor_sensitivity_df = (
    tmp.groupby('stock', group_keys=False)
       .apply(competitor_sensitivity)
       .rename('competitor_sensitivity')
       .reset_index()
)

# Most recent competitor trend per stock
latest_comp_trend = (
    tmp.sort_values('earnings_date')
       .groupby('stock', as_index=False)
       .last()[['stock', 'competitor_trend']]
)

# Merge the two
competitor_output = competitor_sensitivity_df.merge(
    latest_comp_trend, on='stock', how='left'
)

# --- Flag competitor influence ---
competitor_output['competitor_influence'] = np.select(
    [
        (competitor_output['competitor_sensitivity'] > 0.4) &
        (competitor_output['competitor_trend'] < 0),

        (competitor_output['competitor_sensitivity'] > 0.4) &
        (competitor_output['competitor_trend'] > 0)
    ],
    [
        "High - Peers recently weak, stock tends to follow their downside.",
        "Moderate - Peers recently strong, stock tends to benefit."
    ],
    default="Low - Weak or inverse linkage to competitors."
)

# --- Join with recommendation and risk score for final output ---
output_df = earnings_df[['stock', 'earnings_date', 'risk_score']].drop_duplicates()
output_df['recommendation'] = output_df['risk_score'].apply(recommendation_from_risk)

output_df = output_df.merge(competitor_output, on='stock', how='left')

# Optional: reorder for readability
output_df = output_df[
    ['stock', 'earnings_date', 'risk_score', 'recommendation',
     'competitor_sensitivity', 'competitor_trend', 'competitor_influence']
].sort_values(['earnings_date', 'stock'], ascending=[False, True] )

# group_key = 'sub_sector'

# # Peer 3-day avg after earnings
# earnings_df['peer_avg_3d'] = (
#     earnings_df.groupby([group_key, 'earnings_date'])['ret_3d_from_earnings']
#     .transform('mean')
# )

# # Rolling competitor trend (8-event window)
# earnings_df['competitor_trend'] = (
#     earnings_df.groupby(group_key)['peer_avg_3d']
#     .transform(lambda x: x.rolling(8, min_periods=2).mean())
# )

# # Relative performance vs competitors
# earnings_df['relative_to_competitors'] = (
#     earnings_df['ret_3d_from_earnings'] - earnings_df['competitor_trend']
# )

# # Sensitivity (correlation between own reaction and peer trend)
# def competitor_sensitivity(group):
#     if len(group) < 4:
#         return np.nan
#     return group['ret_3d_from_earnings'].corr(group['competitor_trend'])

# competitor_sensitivity_df = (
#     earnings_df.groupby('stock', group_keys=False)
#     .apply(competitor_sensitivity)
#     .rename('competitor_sensitivity')
#     .reset_index()
# )

# # Join with most recent competitor trend per stock
# latest_comp_trend = (
#     earnings_df.sort_values('earnings_date')
#     .groupby('stock', as_index=False)
#     .last()[['stock', 'competitor_trend']]
# )

# competitor_output = competitor_sensitivity_df.merge(
#     latest_comp_trend, on='stock', how='left'
# )

# # change to output['competitor
# competitor_output['competitor_risk_flag'] = np.select(
#     [
#         (competitor_output['competitor_sensitivity'] > 0.4) &
#         (competitor_output['competitor_trend'] < 0),
#         (competitor_output['competitor_sensitivity'] > 0.4) &
#         (competitor_output['competitor_trend'] > 0)
#     ],
#     [
#         'High - Peers recently weak, stock tends to follow',
#         'Moderate - Peers strong, may benefit'
#     ],
#     default='Low - Weak or inverse competitor linkage'
# )

"""#### Pre-Earnings Risk Indicator"""

# --- 1. Temporary working copy ---
tmp = earnings_df.copy()

# Positive surprise & negative post-earnings return
tmp['positive_surprise'] = tmp['surprise'] > 0
tmp['drop_after_positive'] = tmp['positive_surprise'] & (tmp['ret_3d_from_earnings'] < 0)

# --- 2. Compute per-stock frequency ---
pre_risk = (
    tmp.groupby('stock')['drop_after_positive']
       .mean()
       .rename('pre_earnings_risk_score')
       .reset_index()
)

# --- 3. Descriptive label for interpretability ---
def pre_risk_label(x):
    if x > 0.5:
        return "High - Often drops despite positive earnings."
    elif x > 0.2:
        return "Moderate - Mixed post-earnings reactions."
    else:
        return "Low - Typically rewarded for good earnings."

pre_risk['pre_earnings_risk_flag'] = pre_risk['pre_earnings_risk_score'].apply(pre_risk_label)

# --- 4. Merge with existing output (recommendation + competitor influence) ---
output_df = (
    output_df
    .merge(pre_risk, on='stock', how='left')
    .sort_values(['earnings_date', 'stock'], ascending=[False, True])
)

# # Positive surprise & negative post-earnings return
# earnings_df['positive_surprise'] = earnings_df['surprise'] > 0
# earnings_df['drop_after_positive'] = (
#     (earnings_df['positive_surprise']) &
#     (earnings_df['ret_3d_from_earnings'] < 0)
# )

# # Frequency of drops despite positive earnings
# pre_risk = (
#     earnings_df.groupby('stock')['drop_after_positive']
#     .mean()
#     .rename('pre_earnings_risk_score')
#     .reset_index()
# )

# # Add descriptive label
# def risk_label(x):
#     if x > 0.5:
#         return "High - Often drops despite positive earnings"
#     elif x > 0.2:
#         return "Moderate - Mixed post-earnings reactions"
#     else:
#         return "Low - Typically rewarded for good earnings"

# pre_risk['pre_earnings_risk_flag'] = pre_risk['pre_earnings_risk_score'].apply(risk_label)

"""#### Sector/Sub-Sector Risk"""

# --- 1. Temporary working copy ---
tmp = earnings_df.copy()

# --- 2. Compute sector-level rolling drift & volatility averages (10-event window) ---
sector_trend = (
    tmp.groupby('sector', group_keys=False)
       .agg({
           'sector_drift_10d': 'mean',
           'sector_vol_10d': 'mean',
           'sector_beta': 'mean'
       })
       .rename(columns={
           'sector_drift_10d': 'sector_avg_drift',
           'sector_vol_10d': 'sector_avg_vol',
           'sector_beta': 'sector_avg_beta'
       })
       .reset_index()
)

# --- 3. Label risk based on drift/volatility pattern ---
def sector_risk_label(row):
    # Compute 30th, 70th percentiles for drift and vol to set cutoffs
    drift_low, drift_high = sector_trend['sector_avg_drift'].quantile([0.3, 0.7])
    vol_high = sector_trend['sector_avg_vol'].quantile(0.7)

    if row['sector_avg_drift'] <= drift_low and row['sector_avg_vol'] >= vol_high:
        return "High - Sector under pressure with elevated volatility."
    elif row['sector_avg_drift'] <= drift_low:
        return "Moderate - Sector showing mild weakness."
    elif row['sector_avg_drift'] >= drift_high and row['sector_avg_vol'] < vol_high:
        return "Low - Sector performing strongly with stable volatility."
    else:
        return "Neutral - No strong sector trend detected."

sector_trend['sector_risk_flag'] = sector_trend.apply(sector_risk_label, axis=1)

# --- 4. Bring sector info to output_df temporarily ---
# (earnings_df may have multiple rows per stock, so take latest known sector)
latest_sector = (
    tmp.sort_values('earnings_date')
       .groupby('stock', as_index=False)
       .last()[['stock', 'sector']]
)

output_df = output_df.merge(latest_sector, on='stock', how='left')

# --- 5. Merge sector-level risk flags ---
output_df = output_df.merge(sector_trend[['sector', 'sector_risk_flag']], on='sector', how='left')

output_df = output_df.drop(columns=['sector']).sort_values(['earnings_date', 'stock'])

"""## Post-Earnings Insights

### Excessive Price Move Alert
"""

tmp = earnings_df.sort_values(['stock', 'earnings_date']).copy()

# --- 2. Compute rolling average of absolute 3-day post-earnings moves (last 8 quarters) ---
tmp['avg_reaction_8q'] = (
    tmp.groupby('stock')['ret_3d_from_earnings']
       .transform(lambda x: x.abs().shift().rolling(8, min_periods=2).mean())
)

# --- 3. Compare current reaction to historical average ---
tmp['excessive_move_flag'] = np.where(
    tmp['ret_3d_from_earnings'].abs() > tmp['avg_reaction_8q'],
    1, 0
)

# --- 4. Create human-readable label for easier interpretation ---
tmp['excessive_move_label'] = np.select(
    [
        tmp['excessive_move_flag'] == 1,
        tmp['excessive_move_flag'] == 0
    ],
    [
        "Yes - Move exceeds 8-quarter average.",
        "No - Within normal range."
    ],
    default="No data"
)

# --- 5. Keep only relevant fields ---
move_alert = tmp[['stock', 'earnings_date', 'ret_3d_from_earnings',
                  'avg_reaction_8q', 'excessive_move_label']].copy()

# --- 6. Merge into your main output_df ---
output_df = output_df.merge(move_alert, on=['stock', 'earnings_date'], how='left')

# --- 7. Save updated output ---
output_df = output_df.sort_values(['earnings_date', 'stock'])

# """
#     Flag a stock when its current post-earnings move (3/10-day return)
#     is larger in absolute terms than its average earnings reaction over the last 8 quarters.
# """

# earnings_df = earnings_df.sort_values(['stock', 'earnings_date'])

# # Step 2: Compute the rolling average of absolute past earnings reactions
# earnings_df['avg_reaction_8q'] = (
#     earnings_df
#     .groupby('stock')['ret_3d_from_earnings']  # or 'ret_10d_from_earnings'
#     .transform(lambda x: x.abs().shift().rolling(8, min_periods=2).mean())
# )

# # Step 3: Compare current reaction to average
# output['excessive_move_flag'] = (
#     earnings_df['ret_3d_from_earnings'].abs() > earnings_df['avg_reaction_8q']
# )

# # # Options
# # Create an alert magnitude or ratio
# # output['move_ratio'] = (
# #     earnings_df['ret_3d_from_earnings'].abs() / earnings_df['avg_reaction_8q']
# # )

# # # Alert text for dashboard / CSV output
# # output['excessive_move_alert'] = np.where(
# #     output['excessive_move_flag'],
# #     'Excessive move vs last 8 quarters average',
# #     ''
# # )

# # # Weigh more recent quarters higher:
# # output['avg_reaction_8q'] = (
# #     earnings_df.groupby('stock')['ret_3d_from_earnings']
# #     .transform(lambda x: x.abs().shift().ewm(span=8, min_periods=2).mean())
# # )

"""### Earnings Surprise with No Reaction"""

tmp = earnings_df.copy()

# --- 2. Thresholds ---
significant_earnings_surprise_threshold = 5.0   # +5% or more EPS surprise
low_reaction_threshold = 0.5                    # |price move| < 0.5%

# --- 3. Create component flags ---
tmp['strong_positive_surprise'] = tmp['surprisePercentage'] >= significant_earnings_surprise_threshold
tmp['no_reaction'] = tmp['ret_3d_from_earnings'].abs() < low_reaction_threshold / 100

# --- 4. Combine both to form the alert condition ---
tmp['earnings_surprise_no_reaction'] = (
    tmp['strong_positive_surprise'] & tmp['no_reaction']
)

# --- 5. Human-readable alert label ---
tmp['surprise_no_reaction_alert'] = np.where(
    tmp['earnings_surprise_no_reaction'],
    "Strong earnings but muted price response.",
    "None"
)

# --- 6. Keep only relevant columns for merge ---
no_reaction_alert = tmp[['stock', 'earnings_date', 'surprise_no_reaction_alert']].copy()

# --- 7. Merge into the main output_df ---
output_df = output_df.merge(no_reaction_alert, on=['stock', 'earnings_date'], how='left')

# --- 8. Save updated output ---
output_df = output_df.sort_values(['earnings_date', 'stock'])

# """
#     Identify cases where:
#     The earnings surprise is positive and significant,
#     But the post-earnings move is near zero.

#     This can signal:
#     Market skepticism, or
#     A potential delayed move (lagging reaction).
# """
# # +5% or more surprise
# significant_earnings_surprise_threshold = 5.0

# output['no_reaction_flag'] = (
#     (earnings_df['surprise'] > significant_earnings_surprise_threshold) &
#     (earnings_df['ret_3d_from_earnings'] < 0.01)
# )

# # Step 1: Define what counts as a "strong earnings surprise"
# low_reaction_threshold = 0.5    # |price move| less than 0.5%

# # Step 2: Create a flag for positive strong surprises
# earnings_df['strong_positive_surprise'] = earnings_df['surprisePercentage'] >= significant_earnings_surprise_threshold

# # Step 3: Create a flag for low/no reaction
# earnings_df['no_reaction'] = earnings_df['ret_3d_from_earnings'].abs() < low_reaction_threshold / 100

# # Step 4: Combine both to form the final alert
# earnings_df['earnings_surprise_no_reaction'] = (
#     earnings_df['strong_positive_surprise'] & earnings_df['no_reaction']
# )

# # Step 5: Optional alert text for dashboard / output CSV
# # output['surprise_no_reaction_alert'] = np.where(
# #     earnings_df['earnings_surprise_no_reaction'],
# #     'Strong earnings but muted price response',
# #     'None'
# # )

# # Optional: Dynamic thresholds:
# # Compute the median or standard deviation of past earnings surprises per stock to define “strong surprise” contextually.

"""### Implied vs. Actual Reaction Divergence"""

tmp = earnings_df.copy()

# --- 2. Thresholds for defining expected vs. actual direction ---
positive_surprise_threshold = 2.0   # EPS beat > +2%
negative_surprise_threshold = -2.0  # EPS miss < –2%

# --- 3. Expected direction based on earnings surprise ---
tmp['expected_positive'] = tmp['surprisePercentage'] > positive_surprise_threshold
tmp['expected_negative'] = tmp['surprisePercentage'] < negative_surprise_threshold

# --- 4. Actual direction based on price reaction ---
tmp['actual_up']   = tmp['ret_3d_from_earnings'] > 0
tmp['actual_down'] = tmp['ret_3d_from_earnings'] < 0

# --- 5. Flag divergence (earnings beat but price falls, or miss but price rises) ---
tmp['reaction_divergence'] = (
    (tmp['expected_positive'] & tmp['actual_down']) |
    (tmp['expected_negative'] & tmp['actual_up'])
)

# --- 6. Descriptive alert text ---
tmp['divergence_alert'] = np.select(
    [
        tmp['expected_positive'] & tmp['actual_down'],
        tmp['expected_negative'] & tmp['actual_up']
    ],
    [
        "EPS BEAT but price FELL.",
        "EPS MISS but price ROSE."
    ],
    default="None"
)

# --- 7. Keep only needed columns ---
divergence_alert = tmp[['stock', 'earnings_date', 'reaction_divergence', 'divergence_alert']].copy()

# --- 8. Merge with existing output ---
output_df = output_df.merge(divergence_alert, on=['stock', 'earnings_date'], how='left')

# --- 9. Save updated file ---
output_df = output_df.sort_values(['earnings_date', 'stock'])

# """
#     Flag cases where the stock’s post-earnings move contradicts its earnings result -
#     EPS beat, but price drops or EPS missed, but price rises.

#     This can suggest:
#     Market disbelief or future guidance concerns,
#     Mispricing or lagging reaction,
#     Or insider/expectation effects.
# """

# positive_surprise_threshold = 2.0   # e.g., > +2% = beat
# negative_surprise_threshold = -2.0  # e.g., < -2% = miss

# # Step 2: Define divergence conditions
# earnings_df['expected_positive'] = earnings_df['surprisePercentage'] > positive_surprise_threshold
# earnings_df['expected_negative'] = earnings_df['surprisePercentage'] < negative_surprise_threshold

# # Step 3: Define actual reaction direction
# earnings_df['actual_up']   = earnings_df['ret_3d_from_earnings'] > 0
# earnings_df['actual_down'] = earnings_df['ret_3d_from_earnings'] < 0

# # Step 4: Flag divergence (opposite direction)
# output['reaction_divergence'] = (
#     (earnings_df['expected_positive'] & earnings_df['actual_down']) |
#     (earnings_df['expected_negative'] & earnings_df['actual_up'])
# )

# # Step 5: Optional text label for dashboards or CSVs

# output['divergence_alert'] = np.select(
#     [
#         earnings_df['expected_positive'] & earnings_df['actual_down'],
#         earnings_df['expected_negative'] & earnings_df['actual_up']
#     ],
#     [
#         'EPS BEAT but price FELL',
#         'EPS MISSED but price ROSE'
#     ],
#     default=''
# )

"""### Negative Sentiment Alert

Flag cases where a company’s latest earnings call transcript shows a spike in negative sentiment relative to its past 6-8 quarters.
    This will help identify potential management pessimism, guidance downgrades, or hidden bad news not yet reflected in the numbers.


    Finish 3B Earnings Call Sentiment Analysis First!

### Muted Stock Response to Earnings Beat/Miss
"""

tmp = earnings_df.copy()

# --- 2. Define thresholds (tunable) ---
big_surprise_threshold = 5.0    # ±5% EPS or revenue surprise
small_move_threshold   = 1.0     # ±1% price move (3-day)

# --- 3. Flag strong beat or miss ---
tmp['big_beat'] = tmp['surprisePercentage'] >= big_surprise_threshold
tmp['big_miss'] = tmp['surprisePercentage'] <= -big_surprise_threshold

# --- 4. Flag muted stock movement ---
tmp['muted_move'] = tmp['ret_3d_from_earnings'].abs() < (small_move_threshold / 100)

# --- 5. Combine conditions into a single alert flag ---
tmp['muted_response_alert_flag'] = (
    (tmp['big_beat'] | tmp['big_miss']) & tmp['muted_move']
)

# --- 6. Human-readable alert label ---
tmp['muted_response_alert'] = np.select(
    [
        tmp['big_beat'] & tmp['muted_move'],
        tmp['big_miss'] & tmp['muted_move']
    ],
    [
        "Strong EPS beat (≥ +10%) but muted price reaction (±1%).",
        "Big EPS miss (≤ −10%) but muted price reaction (±1%)."
    ],
    default="None"
)

# --- 7. Keep only relevant columns for merging ---
muted_alert = tmp[['stock', 'earnings_date', 'muted_response_alert_flag',
                   'muted_response_alert']].copy()

# --- 8. Merge into main output_df ---
output_df = output_df.merge(muted_alert, on=['stock', 'earnings_date'], how='left')

# --- 9. Save updated output ---
output_df = output_df.sort_values(['earnings_date', 'stock'])

# """
#     Detect cases where:
#     The earnings surprise (EPS or revenue) is large (≥ ±10%),
#     But the post-earnings move is small (within ±1%).
#     This flags potential disbelief, delayed reaction, or priced-in results.
# """
# # Step 1: define thresholds (you can tune these)
# big_surprise_threshold = 10.0    # ±10% EPS or revenue surprise
# small_move_threshold   = 1.0     # ±1% price move (3-day)

# # Step 2: flag strong beat or miss
# earnings_df['big_beat'] = earnings_df['surprisePercentage'] >= big_surprise_threshold
# earnings_df['big_miss'] = earnings_df['surprisePercentage'] <= -big_surprise_threshold

# # Step 3: flag small price reaction
# earnings_df['muted_move'] = earnings_df['ret_3d_from_earnings'].abs() < (small_move_threshold / 100)

# # Step 4: combine conditions
# output['muted_response_alert_flag'] = (
#     (earnings_df['big_beat'] | earnings_df['big_miss']) &
#     earnings_df['muted_move']
# )

# # Step 5: descriptive alert text
# output['muted_response_alert'] = np.select(
#     [
#         earnings_df['big_beat'] & earnings_df['muted_move'],
#         earnings_df['big_miss'] & earnings_df['muted_move']
#     ],
#     [
#         'Strong EPS beat (+>=10%) but muted price reaction (+-1%)',
#         'Big EPS miss (->=10%) but muted price reaction (+-1%)'
#     ],
#     default='None'
# )

"""### Extreme Volatility Alert"""

tmp = earnings_df.sort_values(['stock', 'earnings_date']).copy()

# --- 2) Parameters (tunable) ---
vol_floor = 1e-6          # avoid division by zero
ratio_threshold = 3.0      # "exceeds 3x implied range"
use_return_cols = ['ret_3d_from_earnings', 'ret_10d_from_earnings']

# --- 3) Realized absolute moves ---
tmp['realized_move_3d']  = tmp['ret_3d_from_earnings'].abs()
tmp['realized_move_10d'] = tmp['ret_10d_from_earnings'].abs()

# --- 4) Ratios vs (implied) volatility proxy (vol_10d) ---
# If vol_10d is NaN or zero, replace with floor so ratios stay defined
denom = (tmp['vol_10d'].fillna(0) + vol_floor)
tmp['move_vs_vol_3d']  = tmp['realized_move_3d']  / denom
tmp['move_vs_vol_10d'] = tmp['realized_move_10d'] / denom

# --- 5) Flag & human label ---
tmp['extreme_volatility_alert_flag'] = (
    (tmp['move_vs_vol_3d']  > ratio_threshold) |
    (tmp['move_vs_vol_10d'] > ratio_threshold)
).astype(int)

tmp['extreme_volatility_alert'] = np.where(
    tmp['extreme_volatility_alert_flag'] == 1,
    "Extreme post-earnings volatility: move > 2× normal 10-day range.",
    "None"
)

# Optional: concise numeric context for dashboards/CSV
# (rounded to bps/% for readability)
tmp['move_ctx'] = (
    "3d:" + (tmp['realized_move_3d']*100).round(2).astype(str) + "% "
    + "(x_sigma:" + tmp['move_vs_vol_3d'].round(2).astype(str) + "); "
    + "10d:" + (tmp['realized_move_10d']*100).round(2).astype(str) + "% "
    + "(x_sigma:" + tmp['move_vs_vol_10d'].round(2).astype(str) + ")"
)

# --- 6) Keep only needed columns for merge ---
extreme_alert = tmp[[
    'stock', 'earnings_date',
    'realized_move_3d', 'realized_move_10d',
    'move_vs_vol_3d', 'move_vs_vol_10d',
    'extreme_volatility_alert_flag', 'extreme_volatility_alert',
    'move_ctx'
]].copy()

# --- 7) Merge into your existing output_df and export ---
output_df = output_df.merge(extreme_alert, on=['stock', 'earnings_date'], how='left')
output_df = output_df.sort_values(['earnings_date', 'stock'])
output_df.to_csv('./outputs/output.csv', index=False)

# """
#     Identify stocks where:
#     The actual post-earnings move (3/10-day return)
#     exceeds 2x their implied volatility range,
#     Signaling shock, repricing, or model-breaking behavior.
# """

# # Absolute realized return
# earnings_df['realized_move_3d']  = earnings_df['ret_3d_from_earnings'].abs()
# earnings_df['realized_move_10d'] = earnings_df['ret_10d_from_earnings'].abs()

# # Ratio of realized vs expected
# # 1e-6 - tiny constant added to the denominator to avoid division by zero
# earnings_df['move_vs_vol_3d']  = earnings_df['realized_move_3d']  / (earnings_df['vol_10d'] + 1e-6)
# earnings_df['move_vs_vol_10d'] = earnings_df['realized_move_10d'] / (earnings_df['vol_10d'] + 1e-6)

# output['extreme_volatility_alert_flag'] = (
#     (earnings_df['move_vs_vol_3d']  > 2.0) |
#     (earnings_df['move_vs_vol_10d'] > 2.0)
# )

# # Step 4: add readable alert text
# output['extreme_volatility_alert'] = np.where(
#     output['extreme_volatility_alert_flag'],
#     'Extreme post-earnings volatility: move > 2 x normal 10-day range',
#     'None'
# )

"""## Clean and prepare output for the Dashboard"""

output_df.columns

# Optional: keep only key columns for now
cols_to_keep = [
    "earnings_date", "stock", "risk_score", "recommendation",
    "excessive_move_label", "surprise_no_reaction_alert", "reaction_divergence",
    "muted_response_alert_flag", "extreme_volatility_alert_flag"
]

dashboard_df = output_df[cols_to_keep]
dashboard_df = dashboard_df.dropna()

# Rename for clean display
dashboard_df = dashboard_df.rename(columns={
    "stock": "Stock",
    "earnings_date": "Date",
    "risk_score": "Risk Score",
    "pre_earnings_recommendation": "Recommendation",
    "excessive_move_flag": "Excessive Move",
    "no_reaction_flag": "No Reaction",
    "reaction_divergence": "Reaction Divergence",
    "muted_response_alert_flag": "Muted Response",
    "extreme_volatility_alert_flag": "Extreme Volatility"
})

dashboard_df.to_csv("./outputs/output_dashboard_ready.csv", index=False)

"""# Machine Learning

## Feature Selection
"""

# Load df
df_for_ml = earnings_df.copy()

# One hot econde sector columns
df_for_ml = pd.get_dummies(df_for_ml, columns=['sector','sub_sector'])

# Choose features for ML model

all_columns = list(df_for_ml.columns)

['value',
 'reportedEPS',
 'market_cap_log',
 'beta_5y_monthly',
 'daily_ret',
 'drift_10d',
 'vol_10d',
 'mom_3d',
 'sector_drift_10d',
 'sector_vol_10d',
 'surprise_bucket',
 'is_nochange',
 'past_up_freq',
 'past_down_freq',
 'past_nochange_freq',
 'stdev_ret_3d',
 'stdev_ret_10d',
 'beta_bucket',
 'cap_bucket',
 'sector_mean_3d',
 'sector_mean_10d',
 'relative_to_sector',
 'sector_beta',
 'risk_score']

# features_to_exclude = [
#     # Identifiers / time info
#     'date',
#     'stock',
#     'year',
#     'quarter',
#     'earnings_date',
#     'fiscalDateEnding',

#     # Target / label or derived from label
#     'ret_3d_from_earnings',
#     'ret_10d_from_earnings',
#     'reaction_3d',
#     'reaction_10d',
#     'relative_3d',
#     'relative_10d',
#     'flag_diff_3d',
#     'flag_diff_10d',
#     'direction_mismatch',
#     'is_up',
#     'is_down',
#     'is_nochange',

#     # Label or risk outputs (post-model)
#     'risk_score',
#     'sector_peer_risk_flag',

#     # Columns that directly encode future info or engineered summaries that include it
#     'expected',
#     'consistent_3d',
#     'consistent_10d',
#     'neg_reaction_to_positive_surprise_3d',
#     'neg_reaction_to_positive_surprise_10d',
#     'flag_volatility_3d',
#     'flag_volatility_10d',
#     'sector_mean_3d_same_day',
#     'relative_to_sector',

#     # Core fundamentals that are already captured by surprise features
#     'reportedEPS',
#     'estimatedEPS',
#     'surprise',
#     'surprisePercentage',
#     'surprise_bucket'

#     # Non-numerical or constant after one-hot encoding
#     'value',
# ]


# features = [col for col in all_columns if col not in features_to_exclude] # return the features that aren't in features_to_exclude

# Building for ML
# Choose Label - return 3 days from earning report date
df_for_ml['label_10d'] = (df_for_ml['ret_10d_from_earnings'] > 0).astype(int)

earnings_df_for_ml = df_for_ml.copy()


features = [
    "estimatedEPS",
    "market_cap_log",
    "beta_5y_monthly",
    "daily_ret",
    "drift_10d",
    "vol_10d",
    "mom_3d",
    "sector_drift_10d",
    "sector_vol_10d",
    "past_up_freq",
    "past_down_freq",
    "past_nochange_freq",
    "stdev_ret_3d",
    "stdev_ret_10d"
]


features_with_label_list = features.copy()
features_with_label_list.append('label_10d')

#Handling missing values in feature columns
# 1. Impute rolling stats and sector-level features
for col in ['drift_10d', 'vol_10d', 'mom_3d', 'sector_drift_10d', 'sector_vol_10d']:
    if col in earnings_df_for_ml.columns:
        earnings_df_for_ml[col] = earnings_df_for_ml[col].fillna(earnings_df_for_ml[col].mean())

# 2. Impute stock-level descriptors
for col in ['market_cap_log', 'beta_5y_monthly']:
    if col in earnings_df_for_ml.columns:
        earnings_df_for_ml[col] = earnings_df_for_ml[col].fillna(earnings_df_for_ml[col].mean())

#earnings_df_for_ml.to_csv("./outputs/Earnings_DF_for_ml.csv", index=False)

# One-hot encode quarter?
#earnings_df_for_ml = pd.get_dummies(earnings_df_for_ml, columns=['quarter'])

"""## Train / Test Split"""

# Get rid of NaN values
earnings_df_for_ml = earnings_df_for_ml.dropna(subset = features)

# Drop rows where the label is missing (only critical column)
earnings_df_for_ml = earnings_df_for_ml.dropna(subset=['label_10d'])
earnings_df_for_ml['label_10d'] = (df_for_ml['ret_10d_from_earnings'] > 0).astype(int)

# earnings_df_for_ml = earnings_df_for_ml.dropna() # DELETE THIS
#earnings_df_for_ml.to_csv("./outputs/test.csv",index = False)
train = earnings_df_for_ml[earnings_df_for_ml['date'] < '2023-07-01']
test  = earnings_df_for_ml[earnings_df_for_ml['date'] >= '2023-07-01']

# numeric_cols = [
#     "value", "market_cap_log", "beta_5y_monthly",
#     "daily_ret", "drift_10d", "vol_10d", "mom_3d",
#     "sector_drift_10d", "sector_vol_10d",
#     "estimatedEPS", "past_up_freq", "past_down_freq",
#     "past_nochange_freq", "stdev_ret_3d", "stdev_ret_10d","sector_mean_3d"
# ]

# numeric_cols = [
#     "market_cap_log", "beta_5y_monthly",
#     "daily_ret", "drift_10d", "vol_10d", "mom_3d",
#     "sector_drift_10d", "sector_vol_10d",
#     "estimatedEPS", "past_up_freq", "past_down_freq",
#     "past_nochange_freq", "stdev_ret_3d", "stdev_ret_10d"]

scaler = StandardScaler()

X_train = train[ features ]
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
X_train = scaler.fit_transform(X_train[numeric_cols])
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

y_train = train['label_10d']

X_test  = test[ features ]
X_test = scaler.transform(X_test[numeric_cols])
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

y_test  = test['label_10d']

#
# X_train = train[features].dropna()
# y_train = train.loc[X_train.index, 'label_10d']   # align labels with kept rows

# X_test = test[features].dropna()
# y_test = test.loc[X_test.index, 'label_10d']

"""## Defining Models"""

# Logistic Regression
Log_Reg_model = LogisticRegression(max_iter=1000)

# Random Forest
Rand_Forest_model = RandomForestClassifier(n_estimators=200, random_state=42)

# MLP
MLP_model = Pipeline([
    ('mlp', MLPClassifier(hidden_layer_sizes=(50,),      # simpler architecture
                          activation='relu',
                          solver='adam',
                          max_iter=100,                   # fewer iterations
                          early_stopping=True,            # stops when validation loss stops improving
                          n_iter_no_change=5,             # patience for early stopping
                          random_state=42))
])

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)

# Adaboost
ada_model = AdaBoostClassifier(random_state=42)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,       # number of boosting rounds
    learning_rate=0.1,      # step size shrinkage
    max_depth=3,            # depth of each tree
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'   # avoids deprecation warning
)

"""## Running Models"""

# models = [Log_Reg_model, Rand_Forest_model, MLP_model, knn_model, ada_model, xgb_model]

# for model in models:
#     if model == MLP_model:
#         print(f"Training MLP...")
#     else:
#         print(f"Training {model.__class__.__name__}...")

#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)
#     probs = model.predict_proba(X_test)[:,1]

#     accuracy = accuracy_score(y_test, preds)
#     print(f"Accuracy: {accuracy:.4f}")

#     roc_auc = roc_auc_score(y_test, probs)
#     print(f"ROC AUC: {roc_auc:.4f}")

#     # Cross Evaluation - Time dependent
#     tscv = TimeSeriesSplit(n_splits=5)
#     scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
#     # scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
#     print(f"CV ROC AUC: {scores.mean():.4f}")
#     print("\n")

"""## Cross K-Fold Evaluation"""

# from sklearn.metrics import roc_curve, auc

# # Convert to numpy arrays for slicing
# X_sorted = X_test.values
# y_sorted = y_test.values

# # Use time-based cross-validation
# n_splits = 5
# tscv = TimeSeriesSplit(n_splits=n_splits)

# # Initialize plots
# cols = 3   # number of columns
# rows = int(np.ceil(len(models) / cols))
# fig, ax = plt.subplots(rows, cols, figsize=(22, 10))  # much larger
# ax = ax.flatten()

# for i, model in enumerate(models):
#     model_name = model.__class__.__name__
#     print(f"\nRunning {model_name}\n")

#     mean_fpr = np.linspace(0, 1, 100)
#     tprs, aucs = [], []

#     # Time-based cross-validation
#     for fold, (train_index, test_index) in enumerate(tscv.split(X_sorted)):
#         X_train_fold, X_test_fold = X_sorted[train_index], X_sorted[test_index]
#         y_train_fold, y_test_fold = y_sorted[train_index], y_sorted[test_index]

#         # Train
#         model.fit(X_train_fold, y_train_fold)

#         # Predict
#         y_prob = model.predict_proba(X_test_fold)[:, 1]

#         # ROC
#         fpr, tpr, _ = roc_curve(y_test_fold, y_prob)
#         roc_auc = auc(fpr, tpr)

#         # Interpolate for mean ROC
#         tprs.append(np.interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         aucs.append(roc_auc)

#         ax[i].plot(fpr, tpr, lw=1, alpha=0.3,
#                    label=f'Fold {fold+1} (AUC = {roc_auc:.4f})')

#     # Mean ROC
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)

#     ax[i].plot(mean_fpr, mean_tpr, color='b',
#                label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})',
#                lw=2, alpha=0.8)

#     ax[i].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)

#     ax[i].set_xlim([0.0, 1.0])
#     ax[i].set_ylim([0.0, 1.05])
#     ax[i].set_xlabel('False Positive Rate')
#     ax[i].set_ylabel('True Positive Rate')
#     ax[i].set_title(f'{model_name} ROC')
#     ax[i].legend(loc='lower right')

# plt.suptitle('ROC Curves for Models with Time-Series Cross Validation', fontsize=16)
# plt.tight_layout()
# plt.show()

"""# Back testing to validate accuracy

Prepare the dataset
"""

# --- 1. Copy and clean base data ---
backtesting_df = earnings_df.copy()
backtesting_df = backtesting_df.sort_values(['stock', 'earnings_date']).reset_index(drop=True)
backtesting_df['earnings_date'] = pd.to_datetime(backtesting_df['earnings_date'])
# Create a sector volatility flag based on sector volatility above a threshold
backtesting_df['sector_volatility_score'] = backtesting_df['sector_vol_10d'] / backtesting_df['sector_vol_10d'].rolling(120, min_periods=20).median()# Create a flag for stocks with a history of high volatility reactions
backtesting_df['reaction_volatility'] =( (backtesting_df['past_up_freq'] + backtesting_df['past_down_freq']) > backtesting_df['past_nochange_freq'] ).astype(int)
# Calculate risk-adjusted return (return / volatility)
# backtesting_df['risk_adjusted_return'] = backtesting_df['ret_10d_from_earnings'] / backtesting_df['stdev_ret_10d']
# # Create an interaction term between surprise and volatility
# backtesting_df['surprise_volatility_interaction'] = backtesting_df['surprisePercentage'] * backtesting_df['vol_10d']

# --- Parameters ---
model_threshold = 0.55       # probability cutoff for "long"
transaction_cost = 0.001     # 0.1% per trade
strategy_records = []        # will collect per-year test predictions
all_importances = []         # will collect feature importances

# --- Normalize key features by year to remove 'year' leakage ---

features_to_normalize = [
    'drift_10d', 'vol_10d', 'mom_3d',
    'sector_drift_10d', 'sector_vol_10d',
    'stdev_ret_3d', 'stdev_ret_10d',
    'sector_beta', 'past_vol_10d',
    'sector_volatility_score'
]

def zscore_by_year(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df.groupby('year')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    return df

backtesting_df = zscore_by_year(backtesting_df, features_to_normalize)

# Clean any potential inf/nan left by zero std years
backtesting_df = backtesting_df.replace([np.inf, -np.inf], 0).fillna(0)


# # Replace inf and -inf with NaN
# backtesting_df = backtesting_df.replace([np.inf, -np.inf], np.nan)

# # Drop or fill NaN (better: fill with 0 for standardized features)
# backtesting_df = backtesting_df.fillna(0)

# --- 2. Encode sector information ---
backtesting_df = pd.get_dummies(backtesting_df, columns=['sector'], prefix='sec')

# --- 3. Create target label: positive 10-day return after earnings ---
backtesting_df['label_10d'] = (backtesting_df['ret_10d_from_earnings'] > 0).astype(int)
backtesting_df['relative_to_sector'] = backtesting_df['drift_10d'] - backtesting_df['sector_drift_10d']

# --- 4. Simplify past-earnings direction ---
# combine is_up / is_down / is_nochange into one variable to reduce dimensionality
backtesting_df['prev_earnings_direction'] = (
    backtesting_df.groupby('stock')['is_up'].shift(1)
    - backtesting_df.groupby('stock')['is_down'].shift(1)
) # +1 = up, −1 = down, 0 = flat

cat_cols = ['surprise_bucket', 'beta_bucket', 'cap_bucket']
backtesting_df = pd.get_dummies(backtesting_df, columns=cat_cols, prefix=cat_cols)

# --- 5. Drop unnecessary columns for modeling ---
X_cols = [
    'year', 'market_cap_log', 'beta_5y_monthly', 'daily_ret', 'drift_10d', 'vol_10d', 'mom_3d',
    'sector_drift_10d', 'sector_vol_10d',
    'past_up_freq', 'past_down_freq', 'past_nochange_freq',
    'stdev_ret_3d', 'stdev_ret_10d',
    'sector_mean_3d', 'sector_mean_10d', 'relative_to_sector',
    'past_consistency_3d', 'past_consistency_10d', 'past_vol_10d',
    'sector_beta', 'beta_diff_sector', 'prev_earnings_direction',
    'sector_volatility_score', 'reaction_volatility'
] + [c for c in backtesting_df.columns if c.startswith(('sec_', 'surprise_bucket_', 'beta_bucket_', 'cap_bucket_'))]

"""Rolling Quarterly Backtest"""

# # --- Rolling retrain every quarter (approx. 3-month window) ---

# def rolling_retrain_quarterly(df, model_class=XGBClassifier, lookback_years=2):
#     df = df.copy()
#     df['quarter'] = df['earnings_date'].dt.to_period('Q').astype(str)
#     quarters = sorted(df['quarter'].unique())
#     results = []

#     for i in range(lookback_years * 4, len(quarters)):
#         train_quarters = quarters[i - lookback_years * 4:i]
#         test_quarter   = quarters[i]

#         train = df[df['quarter'].isin(train_quarters)]
#         test  = df[df['quarter'] == test_quarter].copy()

#         if len(train) < 1000 or len(test) == 0:
#             continue

#         X_train, y_train = train[X_cols], train['label_10d']
#         X_test,  y_test  = test[X_cols],  test['label_10d']

#         # model = model_class(
#         #     n_estimators=300, learning_rate=0.05, max_depth=4,
#         #     subsample=0.8, colsample_bytree=0.8,
#         #     random_state=42, eval_metric='logloss'
#         # )
#         model = XGBClassifier(
#             n_estimators=200,          # was 300
#             learning_rate=0.05,
#             max_depth=3,               # was 4
#             subsample=0.7,
#             colsample_bytree=0.7,
#             reg_lambda=2.0,            # added
#             reg_alpha=0.5,             # added
#             random_state=42,
#             eval_metric='logloss'
#         )
#         model.fit(X_train, y_train)

#         probs = model.predict_proba(X_test)[:, 1]
#         test['proba_up'] = probs
#         test['signal_strength'] = probs - 0.5
#         test['weight'] = test.groupby('earnings_date')['signal_strength'].transform(
#             lambda x: x / x.abs().sum()
#         ).fillna(0)
#         test['weight'] = test['weight'].clip(-0.3, 0.3)
#         test['net_ret'] = test['weight'] * (
#             test['ret_10d_from_earnings'] - transaction_cost * (test['signal_strength'] != 0)
#         )

#         results.append(test[['earnings_date','stock','ret_10d_from_earnings','net_ret','proba_up']])

#     return pd.concat(results).sort_values('earnings_date')



# print("Prepared dataset shape:", backtesting_df.shape)
# print("Feature count:", len(X_cols))
# print("Target distribution:\n", backtesting_df['label_10d'].value_counts(normalize=True).round(3))

# # --- Run quarterly retraining backtest ---
# quarterly_backtest = rolling_retrain_quarterly(backtesting_df)

# # --- Combine all quarterly test predictions ---
# quarterly_backtest = quarterly_backtest.sort_values("earnings_date").reset_index(drop=True)

# # --- Compute strategy performance ---
# daily_perf = (
#     quarterly_backtest.groupby("earnings_date")["net_ret"]
#     .sum()
#     .to_frame("strategy_ret")
# )
# daily_perf["cumulative_strategy"] = (1 + daily_perf["strategy_ret"]).cumprod()

# # --- Basic stats ---
# end_capital = daily_perf["cumulative_strategy"].iloc[-1]
# mean_ret = daily_perf["strategy_ret"].mean()
# std_ret = daily_perf["strategy_ret"].std()
# sharpe = mean_ret / std_ret * np.sqrt(252 / 10)   # 10-day horizon ≈ 25 periods/year

# # Adjust CAGR scaling: about 25 ten-day periods per year
# periods_per_year = 252 / 10
# cagr = (end_capital ** (periods_per_year / len(daily_perf))) - 1


# print(f"Cumulative end capital: {end_capital:.3f}")
# print(f"Mean daily return: {mean_ret:.6f}")
# print(f"Std daily return: {std_ret:.6f}")
# print("\n=== Quarterly Rolling Strategy Metrics ===")
# print(f"CAGR           : {cagr:.4f}")
# print(f"Sharpe         : {sharpe:.4f}")
# print(f"Max Drawdown   : {daily_perf['cumulative_strategy'].div(daily_perf['cumulative_strategy'].cummax()).min()-1:.4f}")
# print(f"Hit Rate       : {(daily_perf['strategy_ret']>0).mean():.4f}")
# print(f"Total Return   : {(end_capital-1)*100:.2f}")

# plt.figure(figsize=(10,4))
# plt.plot(daily_perf.index, daily_perf["cumulative_strategy"], label="Quarterly retrain strategy")
# plt.title("Rolling Quarterly Backtest — Cumulative Performance")
# plt.xlabel("Date")
# plt.ylabel("Cumulative Return")
# plt.grid(True)
# plt.legend()
# plt.show()

"""yearly walk-forward loop"""

# --- Parameters ---
model_threshold = 0.55       # probability cutoff for "long"
strategy_records = []        # will collect per-year test predictions
all_importances = []         # will collect feature importances

# --- Walk-forward years setup ---
years = sorted(backtesting_df['earnings_date'].dt.year.unique())
print("Years in data:", years)

# --- Walk-forward backtest loop (skeleton) ---
for i in range(len(years) - 1):
    train_years = years[:i + 1]
    test_year   = years[i + 1]

    train = backtesting_df[backtesting_df['earnings_date'].dt.year.isin(train_years)]
    test  = backtesting_df[backtesting_df['earnings_date'].dt.year == test_year].copy()

    if len(test) == 0 or len(train) < 100:
        continue  # skip if insufficient data

    print(f"\nTraining on {train_years} → Testing {test_year} | "
          f"Train: {len(train)}, Test: {len(test)}")

    # --- Train/Test placeholders (next step will fill) ---
    X_train, y_train = train[X_cols], train['label_10d']
    X_test,  y_test  = test[X_cols],  test['label_10d']

    # numeric-only safety filter
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test[X_train.columns]

"""Train model yearly walk-forward and generate trading signals"""

# --- Walk-forward years setup ---
years = sorted(backtesting_df['earnings_date'].dt.year.unique())

for i in range(len(years) - 1):
    train_years = years[:i + 1]
    test_year   = years[i + 1]

    train = backtesting_df[backtesting_df['earnings_date'].dt.year.isin(train_years)]
    test  = backtesting_df[backtesting_df['earnings_date'].dt.year == test_year].copy()

    if len(test) == 0 or len(train) < 100:
        continue

    print(f"\nTraining on {train_years} → Testing {test_year}")

    # --- Prepare inputs ---
    X_train, y_train = train[X_cols], train['label_10d']
    X_test,  y_test  = test[X_cols],  test['label_10d']

    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test[X_train.columns]

    # --- Train classifier ---
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # --- Predict probabilities ---
    test['proba_up'] = model.predict_proba(X_test)[:, 1]
    # 1. Compute signal strength from probabilities
    test['signal_strength'] = test['proba_up'] - 0.5   #positions are scaled by confidence (proba_up - 0.5), so stronger signals get larger positions  range −0.5 … +0.5

    # 2. Turn into weights proportional to confidence, normalized per date
    test['weight'] = test.groupby('earnings_date')['signal_strength'].transform(
        lambda x: x / x.abs().sum()
    )
    test['weight'] = test['weight'].fillna(0)

    # 3. Apply weighted returns
    test['net_ret'] = test['weight'] * (
        test['ret_10d_from_earnings'] - transaction_cost * (test['signal_strength'] != 0)
    )


    # test['signal_strength'] = test['proba_up'] - 0.5        # −0.5 … +0.5
    # # --- Portfolio weights per earnings date (normalize within-day) ---
    # test['weight'] = test.groupby('earnings_date')['signal_strength'].transform(
    #     lambda x: x / x.abs().sum()
    # ).fillna(0)

    # # --- Net return after transaction cost ---
    # test['net_ret'] = test['weight'] * (
    #     test['ret_10d_from_earnings'] - transaction_cost * (test['signal_strength'] != 0)
    # )

    # Save for later aggregation
    strategy_records.append(test[['earnings_date','stock','ret_10d_from_earnings','net_ret','proba_up']] )
    # --- Feature importances for diagnostics ---
    imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
        'year': test_year
    })
    all_importances.append(imp)

# --- 1. Combine all yearly test outputs ---
backtest = pd.concat(strategy_records).sort_values('earnings_date').reset_index(drop=True)
backtest.rename(columns={'net_ret': 'strategy_ret'}, inplace=True)

# --- 2. Simulate 10-day overlapping positions ---
capital = 1.0
active_positions = []        # list of dicts with {'days_left', 'ret'}
equity_curve = []

for date, group in backtest.groupby('earnings_date'):
    # open new 10-day positions today
    new_positions = [{'days_left': 10, 'ret': r} for r in group['strategy_ret']]
    active_positions += new_positions

    # compute daily average pnl from all open positions
    if active_positions:
        daily_pnl = np.mean([p['ret'] / 10 for p in active_positions])
    else:
        daily_pnl = 0.0

    # update capital
    capital *= (1 + daily_pnl)
    equity_curve.append((date, capital))

    # age and remove expired positions
    for p in active_positions:
        p['days_left'] -= 1
    active_positions = [p for p in active_positions if p['days_left'] > 0]

# --- 3. Build daily dataframe ---
daily = pd.DataFrame(equity_curve, columns=['earnings_date', 'cumulative_strategy'])
daily.set_index('earnings_date', inplace=True)
daily['strategy_ret'] = daily['cumulative_strategy'].pct_change().fillna(0)

print(daily.head())
print("\nCumulative end capital:", round(daily['cumulative_strategy'].iloc[-1], 3))
print("Mean daily return:", daily['strategy_ret'].mean().round(6))
print("Std daily return:", np.round(daily['strategy_ret'].std(), 6))

"""## 10 day Metrics"""

# --- 1. Helper: compute performance metrics ---
def compute_metrics(returns):
    """returns = daily percentage returns (e.g. 0.0013 = 0.13%)"""
    mean, std = returns.mean(), returns.std()
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max - 1)
    max_dd = drawdown.min()

    total_ret = cumulative.iloc[-1] - 1
    cagr = (1 + total_ret) ** (252 / len(returns)) - 1
    hit_rate = (returns > 0).mean()

    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Hit Rate': hit_rate,
        'Total Return': total_ret
    }

# --- 2. Compute metrics for strategy ---
ten_day_ret = backtest.groupby('earnings_date')['strategy_ret'].mean()
metrics_10d = compute_metrics(ten_day_ret)
print("\n=== 10-Day Horizon Metrics ===")
for k,v in metrics_10d.items():
    print(f"{k:15s}: {v:.4f}")


# metrics = compute_metrics(daily['strategy_ret'])
# print("\n=== Strategy Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:15s}: {v:.4f}")

# --- 3. Compare to "market" baseline ---
# Approximate market as mean 10-day return across all stocks
# --- Correct market returns to daily scale (10-day → daily) ---
market_10d = (
    backtest.groupby('earnings_date')['ret_10d_from_earnings']
    .mean()
    .reindex(daily.index, fill_value=0)
)
# Convert each 10-day return to an approximate daily equivalent
daily['market_ret'] = (1 + market_10d) ** (1/10) - 1

# Re-compute cumulative market performance
daily['cumulative_market'] = (1 + daily['market_ret']).cumprod()

# --- 4. Plot cumulative curves ---
plt.figure(figsize=(12,6))
plt.plot(daily.index, daily['cumulative_strategy'], label='Strategy')
plt.plot(daily.index, daily['cumulative_market'], label='Market (mean 10d)', linestyle='--')
plt.title("Walk-Forward Backtest: Strategy vs Market")
plt.legend()
plt.grid(True)
plt.show()

daily['strategy_ret'].describe()

"""## Feature Importance"""

# --- 1. Aggregate feature importances across years ---
fi = pd.concat(all_importances, ignore_index=True)
fi_mean = (
    fi.groupby('feature')['importance']
      .mean()
      .sort_values(ascending=False)
      .head(20)
)

# --- 2. Plot mean feature importance ---
plt.figure(figsize=(8,6))
fi_mean.plot(kind='barh')
plt.title('Average Feature Importance (XGBoost, Walk-Forward)')
plt.xlabel('Mean Importance')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()

# --- 3. Optional: per-year variability check ---
fi_pivot = fi.pivot_table(index='feature', columns='year', values='importance')
print("\nTop 10 features by average importance:\n", fi_mean.head(10))

validation_records = []

for i in range(len(years) - 1):
    test_year = years[i + 1]
    test = backtesting_df[backtesting_df['earnings_date'].dt.year == test_year].copy()

    # Skip if that year's predictions not stored
    if i >= len(strategy_records):
        continue

    preds = strategy_records[i].copy()
    preds = preds.drop_duplicates(subset=['stock', 'earnings_date'])
    preds = preds[['stock', 'earnings_date', 'proba_up']].copy()
    # get ground truth from df
    truth = backtesting_df[backtesting_df['earnings_date'].dt.year == test_year][
        ['stock', 'earnings_date', 'label_10d']
    ].drop_duplicates(subset=['stock', 'earnings_date'])

    # merge to align perfectly
    merged = pd.merge(truth, preds, on=['stock', 'earnings_date'], how='inner')

    if merged.empty:
        continue

    y_true = merged['label_10d']
    y_prob = merged['proba_up']
    y_pred = (y_prob >= 0.5).astype(int)

    metrics_year = {
        'year': int(test_year),
        'n_samples': len(merged),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }
    validation_records.append(metrics_year)

val_df = pd.DataFrame(validation_records)
print("\n=== Year-by-Year Validation Metrics ===")
print(val_df.round(3))

plt.figure(figsize=(8,5))
for col in ['accuracy','precision','recall','auc']:
    plt.plot(val_df['year'], val_df[col], marker='o', label=col)
plt.title("Year-by-Year Predictive Performance")
plt.xlabel("Test Year")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

"""## Using top-confidence subset top 20%"""

# --- Parameters ---
confidence_cut = 0.80        # trade only top 20 % of predictions each earnings_date
capital = 1.0
active_positions = []
equity_curve_top = []

# --- 1. Merge all test-year predictions ---
backtest_conf = pd.concat(strategy_records).sort_values('earnings_date').reset_index(drop=True)
backtest_conf.rename(columns={'net_ret': 'strategy_ret'}, inplace=True)

# --- 2. Filter to top-confidence predictions per earnings_date ---
def select_top_conf(group):
    threshold = group['proba_up'].quantile(confidence_cut)
    return group[group['proba_up'] >= threshold]

backtest_top = backtest_conf.groupby('earnings_date', group_keys=False).apply(select_top_conf)
print(f"Filtered to {len(backtest_top)} trades ({len(backtest_top)/len(backtest_conf):.1%} of total)")

# --- 3. Simulate overlapping 10-day positions only for those trades ---
for date, group in backtest_top.groupby('earnings_date'):
    new_positions = [{'days_left':10, 'ret': r} for r in group['strategy_ret']]
    active_positions += new_positions

    if active_positions:
        weights = 1 / len(active_positions)
        daily_pnl = sum(weights * ((1 + p['ret'])**(1/10) - 1) for p in active_positions)
    else:
        daily_pnl = 0.0

    capital *= (1 + daily_pnl)
    equity_curve_top.append((date, capital))

    for p in active_positions:
        p['days_left'] -= 1
    active_positions = [p for p in active_positions if p['days_left'] > 0]

# --- 4. Build equity curve + metrics ---
daily_top = pd.DataFrame(equity_curve_top, columns=['earnings_date','cumulative_strategy']).set_index('earnings_date')
daily_top['strategy_ret'] = daily_top['cumulative_strategy'].pct_change().fillna(0)

metrics_top = compute_metrics(daily_top['strategy_ret'])
print("\n=== Top-Confidence Strategy Metrics ===")
for k,v in metrics_top.items():
    print(f"{k:15s}: {v:.4f}")

plt.figure(figsize=(12,6))
plt.plot(daily_top.index, daily_top['cumulative_strategy'], label='Top-Confidence Strategy')
plt.plot(daily.index, daily['cumulative_strategy'], label='Full Strategy', linestyle='--')
plt.plot(daily.index, daily['cumulative_market'], label='Market Benchmark', linestyle=':')
plt.title("Top-Confidence vs Full Strategy vs Market")
#plt.title("Top-Confidence vs Full Strategy")
plt.legend()
plt.grid(True)
plt.show()

# --- 1. Define the rolling retraining function ---
def rolling_retrain(df, window_size=2, step_size=1, model_class=XGBClassifier):
    years = sorted(df['earnings_date'].dt.year.unique())
    results = []

    # Iterate over the years, training on a rolling window
    for i in range(window_size, len(years) - step_size + 1, step_size):
        train_years = years[i - window_size:i]  # training on the last `window_size` years
        test_year = years[i]  # test on the next year

        # Filter the data for train and test years
        train = df[df['earnings_date'].dt.year.isin(train_years)]
        test = df[df['earnings_date'].dt.year == test_year].copy()

        if len(train) == 0 or len(test) == 0:
            continue  # Skip if no data in the training or testing year

        # Prepare features and target for training
        X_train, y_train = train[features], train['label_10d']
        X_test, y_test = test[features], test['label_10d']

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
        X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

        # Train the model
        model = model_class(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate performance
        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        results.append({
            'train_years': train_years,
            'test_year': test_year,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        })

    return pd.DataFrame(results)

# --- 2. Run rolling retraining ---
results = rolling_retrain(backtesting_df)

# Display the results
print(results)
