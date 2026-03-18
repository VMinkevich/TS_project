import os
import pandas as pd
from datetime import datetime

from src.data.load_data import download_data
from src.data.preprocess import preprocess_data, save_processed
from src.models.naive_model import naive_forecast
from src.models.garch_model import garch_forecast
from src.models.xgboost_model import xgb_forecast
from src.models.mlp_model import mlp_forecast

from src.evaluation.evaluate_regimes import evaluate_by_regime
from src.utils.logger import log_results
from src.config import PATHS

# Потом вынесу в конфиг, но пока так для удобства
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

def run_pipeline():
    # Скачиваем данные
    download_data()

    # Папка для предсказаний
    os.makedirs("results/predictions", exist_ok=True)

    for file_name in os.listdir(PATHS["raw_data"]):
        print(f"\nProcessing {file_name}...")

        # препроцесс
        df = preprocess_data(file_name)
        save_processed(df, file_name)

        # делим на трейн и тест
        df_train = df[df.index <= TRAIN_END]
        df_test = df[df.index >= TEST_START]

        print(f"Train: {df_train.index[0]} → {df_train.index[-1]}")
        print(f"Test: {df_test.index[0]} → {df_test.index[-1]}")

        # формируем regime только на тесте
        from src.data.volatility_regimes import add_volatility_regime
        df_test = add_volatility_regime(df_test)

        # Модели
        # 1) Naive
        df_test_naive = naive_forecast(df_test)
        results_naive = evaluate_by_regime(df_test_naive)
        log_results(results_naive, file_name, model_name="naive")
        print("Naive Results:")
        print(results_naive)

        # 2) GJR-GARCH
        df_test_garch = garch_forecast(df_train, df_test)
        results_garch = evaluate_by_regime(df_test_garch)
        log_results(results_garch, file_name, model_name="gjr_garch")
        print("GARCH Results:")
        print(results_garch)

        # 3) XGBoost
        df_test_xgb = xgb_forecast(df_train, df_test)
        results_xgb = evaluate_by_regime(df_test_xgb)
        log_results(results_xgb, file_name, model_name="xgboost")
        print("XGBoost Results:")
        print(results_xgb)

        # 4) MLP (взял самый примитивный вариант, нет гпу для тестирования и запуска локально всего проекта)
        df_test_mlp = mlp_forecast(df_train, df_test)
        results_mlp = evaluate_by_regime(df_test_mlp)
        log_results(results_mlp, file_name, model_name="mlp")
        print("MLP Results:")
        print(results_mlp)

        # Сохраняем все предсказания в один CSV
        df_preds = df_test[['return']].copy()
        df_preds['regime'] = df_test['regime']
        df_preds['naive'] = df_test_naive['prediction']
        df_preds['garch'] = df_test_garch['prediction']
        df_preds['xgboost'] = df_test_xgb['prediction']
        df_preds['mlp'] = df_test_mlp['prediction']

        pred_file = os.path.join("results/predictions", f"{file_name.replace('.csv','')}_predictions.csv")
        df_preds.to_csv(pred_file)
        print(f"Predictions saved to {pred_file}")

    print("\n Pipeline finished. Metrics logged to JSON and predictions saved.")
    return

if __name__ == "__main__":
    run_pipeline()