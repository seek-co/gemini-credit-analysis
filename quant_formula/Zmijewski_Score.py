import pandas as pd

def zmijewski_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes the Zmijewski Score for each row.
    
    Required columns in df:
      - net_income
      - total_assets
      - total_liabilities
      - current_assets
      - current_liabilities
    
    Returns:
      A pandas Series with the Zmijewski score for each row.
    """
    df = df.copy()

    x_score = (
          -4.336
        - 4.513 * (df['net_income'] / df['total_assets'])
        + 5.679 * (df['total_liabilities'] / df['total_assets'])
        + 0.004 * (df['current_assets'] / df['current_liabilities'])
    )
    
    return x_score

