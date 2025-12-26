# Imports from other modules
from risk_scoring.reccomendation import (add_risk_recommendation,
                                         add_reccomendation_explanations,
                                         add_pre_earnings_risk_flag,
                                         add_sector_level_risk_flags,
                                         add_excessive_price_move_alert,
                                         add_surprise_no_reaction_alert,
                                         add_eps_reaction_divergence_alert,
                                         add_muted_response_alert,
                                         add_extreme_volatility_alert,
                                         prepare_df_for_dashboard)
def stage4(earnings_df):
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

  """ Earnings Surprise with No Reaction """
  output_df = add_surprise_no_reaction_alert(earnings_df, output_df)

  """ Implied vs. Actual Reaction Divergence """
  output_df = add_eps_reaction_divergence_alert(earnings_df, output_df)

  # Not implemented yet
  # add_negative_sentiment_alert(earnings_df, output_df)
  
  """ Muted Stock Response to Earnings Beat/Miss """
  output_df = add_muted_response_alert(earnings_df, output_df)

  """ Extreme Volatility Alert """
  output_df = add_extreme_volatility_alert(earnings_df, output_df)
  
  """ Final Output DataFrame for Dashboard """
  dashboard_df = prepare_df_for_dashboard(output_df)

  dashboard_df.to_csv("outputs/output_dashboard_ready.csv", index=False)

  return dashboard_df