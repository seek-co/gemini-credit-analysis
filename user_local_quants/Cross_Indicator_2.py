import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    #Compute EMA of a Series
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    
    # Standard RSI calculation.

    delta = series.diff()
    # Gains (positive) and losses (negative)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Use EWMA to calculate average gain/loss
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def franken_indicator(df: pd.DataFrame,
                      rsi_period: int = 14,
                      bb_period: int = 20,
                      bb_std_dev: float = 2.0) -> pd.DataFrame:
   
    df = df.copy()  

    # 1) Volume-Weighted Price (VWPrice)
    #    typical_price = (H + L + C) / 3
    #    volume_weighted = typical_price * volume
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    df['VWPrice'] = typical_price * df['volume']

    # 2) Compute RSI on the volume-weighted price
    df['RSI'] = calc_rsi(df['VWPrice'], period=rsi_period)

    # 3) Triple-smooth the RSI with EMA
    df['RSI_EMA1'] = ema(df['RSI'], period=5)
    df['RSI_EMA2'] = ema(df['RSI_EMA1'], period=5)
    df['RSI_EMA3'] = ema(df['RSI_EMA2'], period=5)

    # 4) Bollinger Bands on the triple-smoothed RSI
   
    df['BB_Mid'] = df['RSI_EMA3'].rolling(bb_period).mean()
    df['BB_Std'] = df['RSI_EMA3'].rolling(bb_period).std(ddof=0)  # population std
    df['BB_Upper'] = df['BB_Mid'] + bb_std_dev * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - bb_std_dev * df['BB_Std']

    # 5) Generate signals: crossing above/below Bollinger middle band
    #    - +1 if the RSI_EMA3 crosses from below to above the BB_Mid
    #    - -1 if it crosses from above to below the BB_Mid
    #    - 0 otherwise
    df['Signal'] = 0
    for i in range(1, len(df)):
        prev_rsi = df.at[df.index[i-1], 'RSI_EMA3']
        curr_rsi = df.at[df.index[i], 'RSI_EMA3']
        prev_mid = df.at[df.index[i-1], 'BB_Mid']
        curr_mid = df.at[df.index[i], 'BB_Mid']

        # crossing from below to above
        if (prev_rsi < prev_mid) and (curr_rsi > curr_mid):
            df.at[df.index[i], 'Signal'] = +1
        # crossing from above to below
        elif (prev_rsi > prev_mid) and (curr_rsi < curr_mid):
            df.at[df.index[i], 'Signal'] = -1

    # Cleanup temporary std column
    df.drop(columns=['BB_Std'], inplace=True)

    return df