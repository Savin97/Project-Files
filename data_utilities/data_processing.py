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
