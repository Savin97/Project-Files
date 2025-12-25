# Function to check data integrity, implemenation isnt great
def check_data(stock_values_file,earning_dates,df,eps_df):    
    set_values = set(stock_values_file['Stock'].unique())
    set_earnings = set(earning_dates['Symbol'].unique())
    print(len(set_values), len(set_earnings))
    print("In stock_values_file but not in earning_dates:", set_values - set_earnings)
    print("In earning_dates but not in stock_values_file:", set_earnings - set_values)

    # Check which stocks dont appear in both EPS and df
    set_eps = set(eps_df['stock'].unique())
    set_df = set(df['stock'].unique())

    print("In earning_dates but not in original df:", set_eps - set_df)
    print(f"{len(set_df - set_eps)} in original df but not in earning_dates:", set_df - set_eps )

def check_NA_values_in_cols(earnings_df):
    print("Columns that have N/A values:\n")
    for column in earnings_df.columns:
        n_missing = earnings_df[column].isna().sum()
        if n_missing > 0:
            print(f"{column}: {n_missing}")
