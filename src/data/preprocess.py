import os
import pandas as pd
import numpy as np

from src.config import PATHS, FEATURE_CONFIG


def preprocess_data(file_name):
    file_path = os.path.join(PATHS["raw_data"], file_name)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # лог-доходности
    df["return"] = np.log(df["price"]).diff()

    # лаги
    for lag in FEATURE_CONFIG["return_lags"]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    # rolling volatility
    window = FEATURE_CONFIG["rolling_window"]
    df["volatility"] = df["return"].rolling(window).std()

    df.dropna(inplace=True)

    return df


def save_processed(df, file_name):
    os.makedirs(PATHS["processed_data"], exist_ok=True)

    file_path = os.path.join(PATHS["processed_data"], file_name)
    df.to_csv(file_path)

    print(f"Processed data saved to {file_path}")