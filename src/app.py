from fastapi import FastAPI
import pandas as pd
from catboost import CatBoostRegressor
import os

app = FastAPI(title="Store Item Forecasting API")
MODEL_PATH = "models/catboost_model.cbm"

# Загружаем модель
model = CatBoostRegressor()
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
else:
    print("Файл модели не найден!")

@app.get("/predict")
def predict(store: int, item: int, day_of_week: int, month: int, year: int):
    # Создаем DataFrame СТРОГО в том порядке, который выдал model.feature_names_
    features = pd.DataFrame([{
        "store": store,
        "item": item,
        "day_of_week": day_of_week,
        "month": month,
        "year": year,
        "is_holiday": 0,
        "is_shopping_day": 0,
        "sales_lag_1": 0,
        "sales_lag_7": 0,
        "sales_rolling_mean_7": 0,
        "sales_rolling_mean_30": 0,
        "sales_lag_30": 0
    }])

    # Делаем предсказание
    prediction = model.predict(features)

    # Так как у нас MultiQuantile (0.05, 0.6, 0.98), 
    # результат придет в виде массива [[q1, q2, q3]]
    # Нам нужна медиана (второе число, индекс 1)
    
    if len(prediction.shape) > 1:
        res = prediction[0][1] # Берем 0-ю строку, 1-й столбец (медиана)
    else:
        res = prediction[1]    # Если вдруг вернулся плоский массив

    return {
        "store": store,
        "item": item,
        "forecasted_sales": round(float(res), 2)
    }