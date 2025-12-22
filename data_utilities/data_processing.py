def keep_earnings_dates_only(df):
    earnings_df = df[df['is_earnings']].copy()
    earnings_df = earnings_df.drop(columns=['is_earnings'])
    return earnings_df

def classify_reaction(reaction, threshold=0.005):
    """
        Threshold set by default to 0.5%
        returns 1 if >0.5% (Up)
        returns -1 if <0.5% (Down)
        returns 0 otherwse (No Change)
    """
    
    if reaction > threshold:   # more than +0.5%
        return "Up" # Up
    elif reaction < -threshold:  # less than -0.5%
        return "Down"
    else:
        return "No Change"
