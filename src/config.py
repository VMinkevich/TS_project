DATA_CONFIG = {
    "tickers": {
        "sp500": "^GSPC",
        "eurusd": "EURUSD=X"
    },
    "start_date": "2023-01-01",
    "end_date": "2025-12-31"
}

FEATURE_CONFIG = {
    "return_lags": [1, 5, 10],
    "rolling_window": 20
}

VOLATILITY_CONFIG = {
    "method": "rolling_std",
    "threshold": "median"
}

PATHS = {
    "raw_data": "data/raw/",
    "processed_data": "data/processed/"
}

RANDOM_STATE = 42