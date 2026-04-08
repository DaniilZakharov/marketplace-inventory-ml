import pandas as pd
from catboost import CatBoostRegressor
import os

MODEL_PATH = "models/catboost_model.cbm"

if not os.path.exists(MODEL_PATH):
    print(f"Файл не найден по пути: {MODEL_PATH}")
else:
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    print("Модель успешно загружена!")

    # Создаем одну строку данных (те же фичи, что в train.py)
    test_data = pd.DataFrame([{
        "store": 1, "item": 1, "day_of_week": 1, "month": 5, "year": 2018,
        "is_holiday": 0, "is_shopping_day": 0,
        "sales_lag_1": 15, "sales_lag_7": 15, "sales_lag_30": 15,
        "sales_rolling_mean_7": 15, "sales_rolling_mean_30": 15
    }])

    pred = model.predict(test_data)
    print(f"Прогноз модели: {pred}")