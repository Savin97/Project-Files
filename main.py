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
from data_utilities.data_processing import keep_earnings_dates_only, handle_NA_values, remove_unuseful_features
from data_utilities.nlp import earnings_report_nlp_analysis
from risk_scoring.reccomendation import add_risk_recommendation


def main():

  data = load_and_format_data()
  data_stage_1 = add_stage_1_features(df)
  feature_engineering = stage2(data_stage_1)
  risk_scoring_finished = stage3(feature_engineering)
  
  output = stage4(risk_scoring_finished)

  """ 4. Outputs """
  """ Pre-Earnings Insights """
  """ Recommendations based on risk score """
  output_df = add_risk_recommendation(earnings_df)

  """ Explanations of Decisions """
  """ Competitor Earnings Influence """
  # TODO: CHECK, Might overwrite output_df instead of adding to it
  output_df = add_reccomendation_explanations(earnings_df)

  """ Pre-Earnings Risk Indicator """
  # TODO: CHECK, Might overwrite output_df instead of adding to it
  output_df = add_pre_earnings_risk_flag(earnings_df, output_df)

  """ Sector/Sub-Sector Risk """
  output_df = add_sector_level_risk_flags(earnings_df, output_df)

  print("\nDone.\n\n\n")

if __name__ == "__main__":
    main()