import pandas as pd
import numpy as np

def calculate_laguerre_filter(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
  
    # Assuming data frame has columns: time, open, high, low, close, volume (generally standard)
    source = (df['high_price'] + df['low_price']) / 2

    length = len(source)
    L0 = np.zeros(length)
    L1 = np.zeros(length)
    L2 = np.zeros(length)
    L3 = np.zeros(length)
    LagF = np.zeros(length)

    gamma = 1.0 - alpha

    for i in range(length):
        if i == 0:
            # Initialize for the very first row
            L0[i] = source.iloc[i]
            L1[i] = source.iloc[i]
            L2[i] = source.iloc[i]
            L3[i] = source.iloc[i]
        else:
            L0[i] = (1 - gamma) * source.iloc[i] + gamma * L0[i - 1]
            L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
            L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
            L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]
        
        # Final Laguerre Filter value for this row
        LagF[i] = (L0[i] + 2.0 * L1[i] + 2.0 * L2[i] + L3[i]) / 6.0

    df['LaguerreFilter'] = LagF

    # --- 4) Generate a simple signal based on whether LagF is rising or falling ---
    # +1 if LagF > previous, -1 if LagF < previous, 0 if no previous or no change
    signal = np.sign(df['LaguerreFilter'].diff())
    # For the first row, there's no previous data:
    signal.iloc[0] = 0  
    df['LaguerreSignal'] = signal

    return df