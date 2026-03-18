from src.evaluation.metrics import rmse, mae, directional_accuracy

def evaluate_by_regime(df):
    results = {}

    for regime_value, regime_name in [(0, "low_vol"), (1, "high_vol")]:
        subset = df[df["regime"] == regime_value]

        y_true = subset["return"]
        y_pred = subset["prediction"]

        results[regime_name] = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred)
        }

    # разница
    results["delta"] = {
        "rmse_diff": results["high_vol"]["rmse"] - results["low_vol"]["rmse"],
        "mae_diff": results["high_vol"]["mae"] - results["low_vol"]["mae"],
    }

    return results