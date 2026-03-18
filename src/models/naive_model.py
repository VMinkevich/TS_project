import pandas as pd

def naive_forecast(df):
    """
    Прогноз: y_t = y_{t-1}
    """
    df = df.copy()
    df["prediction"] = df["return"].shift(1)
    df.dropna(inplace=True)

    return df