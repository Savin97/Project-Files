from config import REACTION_THRESHOLD

def keep_earnings_dates_only(df):
    earnings_df = df[df['is_earnings']].copy()
    earnings_df = earnings_df.drop(columns=['is_earnings'])
    return earnings_df

def classify_reaction(reaction):
    """
        REACTION THRESHOLD  set by default to 0.5%
        returns 1 if >0.5% (Up)
        returns -1 if <0.5% (Down)
        returns 0 otherwse (No Change)
    """
    
    if reaction > REACTION_THRESHOLD:   # more than +0.5%
        return 1 # Up
    elif reaction < -REACTION_THRESHOLD:  # less than -0.5%
        return -1 # Down
    else:
        return 0 # No Change

# Expected reaction based on earnings surprise
def expected_direction(surprise):
    """ 
        If surprise is above threshold: Returns 1
        If its below the negative threshold: -1
        If its without change: 0
    """
    if surprise > REACTION_THRESHOLD:
        return 1 # Up
    elif surprise < -REACTION_THRESHOLD:
        return -1 # Down
    else:
        return 0 # No Change

def handle_NA_values(earnings_df):
    """ Handle N/A values """
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
    # ) # fills missing values with the mean of that column within each sector

    # earnings_df["past_consistency_3d"].fillna(
    #     earnings_df.groupby("sector")["past_consistency_3d"].transform("mean"),
    #     inplace=True
    # )

    # fill missing values with the mean of the entire column across all rows
    #earnings_df["past_consistency_10d"].fillna(earnings_df["past_consistency_10d"].mean(), inplace=True) # Might need to be removed

    return earnings_df

def remove_unuseful_features(earnings_df):
    print("Columns before:\n", earnings_df.columns)
    # Removing columns if they have a large number of NaN values
    for col in earnings_df.columns:
        num_missing = earnings_df[col].isna().sum()
        if num_missing > MISSING_FEATURE_VALUES_THRESHOLD:
        # Drop column entirely if there are more than 300 missing
            earnings_df = earnings_df.drop(columns=[col])
        # Drop rows with NaN if there are fewer than 300 missing
        else:
            earnings_df = earnings_df.dropna(subset=[col])
    print("Columns after:\n", earnings_df.columns)
    return earnings_df