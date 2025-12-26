# main.py
# Imports from other modules
from pipeline.pipeline_stage1 import load_and_format_data, add_stage_1_features
from pipeline.pipeline_stage2 import stage2
from pipeline.pipeline_stage3 import stage3
from risk_scoring.reccomendation import (add_risk_recommendation,
                                         add_reccomendation_explanations,
                                         add_pre_earnings_risk_flag,
                                         add_sector_level_risk_flags,
                                         add_excessive_price_move_alert)


def main():
  data = load_and_format_data()
  data_stage_1 = add_stage_1_features(data)
  feature_engineering = stage2(data_stage_1)
  earnings_df = stage3(feature_engineering)

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

  """ Post-Earnings Insights """
  """ Excessive Price Move Alert """
  output_df = add_excessive_price_move_alert(earnings_df, output_df)

  print("\nDone.\n\n\n")

if __name__ == "__main__":
    main()