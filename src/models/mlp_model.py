import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

def create_lag_features(df, lags=[1,2,3,4,5], windows=[5,10]):
    """
    Генерация признаков: лаги, скользящие средние, std, momentum.
    df: DataFrame с колонкой 'return'
    """
    df = df.copy()
    
    # Лаги
    for lag in lags:
        df[f"lag_{lag}"] = df["return"].shift(lag)
    
    # Скользящие признаки
    for w in windows:
        df[f"rolling_mean_{w}"] = df["return"].rolling(w).mean()
        df[f"rolling_std_{w}"] = df["return"].rolling(w).std()
        df[f"momentum_{w}"] = df["return"].rolling(w).sum()
    
    return df

def mlp_forecast(train_df, test_df):
    """
    Обучение MLPRegressor на train_df и прогноз на test_df.
    Возвращает test_df с колонкой 'prediction'.
    """
    # Признаки для train
    train_features = create_lag_features(train_df).dropna()
    feature_cols = [c for c in train_features.columns if c != "return"]
    X_train = train_features[feature_cols]
    y_train = train_features["return"]

    # Признаки для test
    n_window = 10
    combined = pd.concat([train_df.tail(n_window), test_df])
    test_features = create_lag_features(combined).iloc[n_window:]
    X_test = test_features[feature_cols]

    # Обучение MLP
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Прогноз
    test_df = test_df.copy()
    test_df["prediction"] = model.predict(X_test)
    
    return test_df