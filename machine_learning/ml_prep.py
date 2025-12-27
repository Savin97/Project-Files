import pandas as pd
from config import REACTION_THRESHOLD

""" Feature Selection """
def select_features_for_ML(df: pd.DataFrame) -> pd.DataFrame:
    """
        Selects features for ML model from earnings dataframe.
        Args:
            df (pd.DataFrame): DataFrame containing earnings data with engineered features.
        Returns:
            pd.DataFrame: DataFrame with selected features for ML model.
    """
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

    # Check that all columns in df_for_ml are in features
    for col in features:
        if col not in df_for_ml.columns:
            return ValueError(f"Feature {col} not recognized in feature selection.")
        

    # Possible future features to consider
    # ['value',
    # 'reportedEPS',
    # 'market_cap_log',
    # 'beta_5y_monthly',
    # 'daily_ret',
    # 'drift_10d',
    # 'vol_10d',
    # 'mom_3d',
    # 'sector_drift_10d',
    # 'sector_vol_10d',
    # 'surprise_bucket',
    # 'is_nochange',
    # 'past_up_freq',
    # 'past_down_freq',
    # 'past_nochange_freq',
    # 'stdev_ret_3d',
    # 'stdev_ret_10d',
    # 'beta_bucket',
    # 'cap_bucket',
    # 'sector_mean_3d',
    # 'sector_mean_10d',
    # 'relative_to_sector',
    # 'sector_beta',
    # 'risk_score']

    # One hot encode sector columns
    df_for_ml = pd.get_dummies(df_for_ml, columns=['sector','sub_sector'])

    return df_for_ml[features]

def build_features_and_label(df_for_ml: pd.DataFrame) -> pd.DataFrame:
    """
        Builds features and label for ML model from earnings dataframe.
        Args:
            df_for_ml (pd.DataFrame): DataFrame containing earnings data with engineered features.
        Returns:
            pd.DataFrame: DataFrame with features and label for ML model.
    """
    # Select features
    df_for_ml = select_features_for_ML(df_for_ml)
    features = df_for_ml.columns.tolist()

    
# Building for ML
# Choose Label - return 10 days from earning report date
df_for_ml['label_10d'] = (df_for_ml['ret_10d_from_earnings'] > REACTION_THRESHOLD).astype(int)



features_with_label_list = features.copy()
features_with_label_list.append('label_10d')

def handle_missing_values(df_for_ml: pd.DataFrame) -> pd.DataFrame:
    """
        Handles missing values in the DataFrame for ML model.
        Args:
            df_for_ml (pd.DataFrame): DataFrame containing earnings data with engineered features.
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
    """
    #Handling missing values in feature columns
    # 1. Impute rolling stats and sector-level features
    for col in ['drift_10d', 'vol_10d', 'mom_3d', 'sector_drift_10d', 'sector_vol_10d']:
        if col in df_for_ml.columns:
            df_for_ml[col] = df_for_ml[col].fillna(df_for_ml[col].mean())

    # 2. Impute stock-level descriptors
    for col in ['market_cap_log', 'beta_5y_monthly']:
        if col in df_for_ml.columns:
            df_for_ml[col] = df_for_ml[col].fillna(df_for_ml[col].mean())

#df_for_ml.to_csv("df_for_ml.csv", index=False)

# One-hot encode quarter?
#df_for_ml = pd.get_dummies(df_for_ml, columns=['quarter'])