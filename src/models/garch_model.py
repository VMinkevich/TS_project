import pandas as pd
from arch import arch_model


def garch_forecast(train_df, test_df):
    returns = train_df["return"] * 100
    model = arch_model(returns, vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    forecasts = res.forecast(horizon=len(test_df))
    variance = forecasts.variance.values[-1, :]  # последняя доступная строка
    test_df = test_df.copy()
    test_df["predicted_volatility"] = variance ** 0.5
    test_df["prediction"] = 0.0
    return test_df