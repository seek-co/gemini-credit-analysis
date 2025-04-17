import pandas as pd

def altman_z_prime_score(df: pd.DataFrame) -> pd.Series:
    """    
    Required columns in df:
      - current_assets
      - current_liabilities
      - total_assets
      - total_liabilities
      - total_equity  (assumed as "book value of equity")
      - retained_earnings
      - ebit or operating profit
      - sales (or revenue)
    
    Returns:
      A pandas Series with the Z′-score for each row.
    """
    # Copy to avoid modifying original
    df = df.copy()
    
    # 1) Derive Working Capital
    df['working_capital'] = df['current_assets'] - df['current_liabilities']
    
    # 2) Book Value of Equity (for private firms, we can use total_equity directly)
    df['book_value_equity'] = df['total_equity']
    
    # 3) Calculate Z′
    z_prime = (
         0.717 * (df['working_capital']     / df['total_assets'])
       + 0.847 * (df['retained_earnings']   / df['total_assets'])
       + 3.107 * (df['ebit']               / df['total_assets'])
       + 0.420 * (df['book_value_equity']   / df['total_liabilities'])
       + 0.998 * (df['sales']              / df['total_assets'])
    )
    
    return z_prime

