import numpy as np

def add_volatility_regime(df):
    threshold = df["volatility"].median()

    df["regime"] = np.where(df["volatility"] > threshold, 1, 0)

    return df