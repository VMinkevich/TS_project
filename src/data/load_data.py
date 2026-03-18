import os
import yfinance as yf
import pandas as pd

from src.config import DATA_CONFIG, PATHS


def download_data():
    os.makedirs(PATHS["raw_data"], exist_ok=True)

    for name, ticker in DATA_CONFIG["tickers"].items():
        print(f"Downloading {name} ({ticker})...")

        df = yf.download(
            ticker,
            start=DATA_CONFIG["start_date"],
            end=DATA_CONFIG["end_date"]
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Adj Close" in df.columns:
            df = df[["Adj Close"]].rename(columns={"Adj Close": "price"})
        else:
            df = df[["Close"]].rename(columns={"Close": "price"})
        df.dropna(inplace=True)

        file_path = os.path.join(PATHS["raw_data"], f"{name}.csv")
        df.to_csv(file_path)

        print(f"Saved to {file_path}")


if __name__ == "__main__":
    download_data()