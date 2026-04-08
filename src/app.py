import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor

app = FastAPI(title="WB Inventory AI Agent")

# Константы
MODEL_PATH = "models/catboost_model.cbm"
DATA_PATH = "data/train_cleaned.csv" 

# Глобальные переменные для кэширования
model = CatBoostRegressor()
history_df = None

@app.on_event("startup")
def load_assets():
    global history_df
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("Модель не найдена!")
    

    if os.path.exists(DATA_PATH):
        history_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        history_df = history_df.sort_values('date')
    else:
        print("ВНИМАНИЕ: Файл данных не найден. Лаги будут нулевыми.")

class PredictRequest(BaseModel):
    date: str  # YYYY-MM-DD
    store: int
    item: int
    current_stock: int = 0 # Запасы на складе сейчас

def get_features(date_str, store, item):
    target_date = pd.to_datetime(date_str)
    
    # 1. Календарные фичи
    day_of_week = target_date.dayofweek
    month = target_date.month
    year = target_date.year
    
    manual_holidays = [
        (1, 1), (12, 31), (12, 30), (2, 14), (2, 13), (7, 4), 
        (7, 3), (10, 31), (10, 30), (11, 24), (11, 23), (12, 25), (12, 24), (12, 23)
    ]
    is_holiday = 1 if (month, target_date.day) in manual_holidays else 0
    is_shopping_day = 1 if day_of_week in [4, 5] else 0 # Пятница, Суббота
    
    # 2. Динамические лаги
    mask = (history_df['store'] == store) & (history_df['item'] == item) & (history_df['date'] < target_date)
    item_history = history_df[mask].tail(30)
    
    if item_history.empty:
        # Если товара нет, берем средние продажи по всему магазину за этот период
        store_mask = (history_df['store'] == store) & (history_df['date'] < target_date)
        store_history = history_df[store_mask].tail(30)
        
        if store_history.empty:
            return None # Совсем нет данных по магазину
            
        # Заполняем лаги средними по магазину
        avg_sales = store_history['sales'].mean()
        sales_lag_1 = sales_lag_7 = sales_lag_30 = avg_sales
        rolling_7 = rolling_30 = avg_sales
    else:
        sales_lag_1 = item_history['sales'].iloc[-1] if len(item_history) >= 1 else 0
        sales_lag_7 = item_history['sales'].iloc[-7] if len(item_history) >= 7 else sales_lag_1
        sales_lag_30 = item_history['sales'].iloc[-30] if len(item_history) >= 30 else sales_lag_7
        rolling_7 = item_history['sales'].tail(7).mean()
        rolling_30 = item_history['sales'].tail(30).mean()

    return pd.DataFrame([{
        "store": store,
        "item": item,
        "day_of_week": day_of_week,
        "month": month,
        "year": year,
        "is_holiday": is_holiday,
        "is_shopping_day": is_shopping_day,
        "sales_lag_1": sales_lag_1,
        "sales_lag_7": sales_lag_7,
        "sales_rolling_mean_7": rolling_7,
        "sales_rolling_mean_30": rolling_30,
        "sales_lag_30": sales_lag_30
    }])

@app.post("/predict_stock")
def predict_stock(req: PredictRequest):
    features = get_features(req.date, req.store, req.item)
    
    if features is None:
        raise HTTPException(status_code=404, detail="No history found for this store/item")

    prediction = model.predict(features)
    
    # MultiQuantile: 0.05 (пессимист), 0.6 (медиана), 0.98 (верхняя граница)
    q_low, q_med, q_high = prediction[0]
    
    # Бизнес-логика (Reorder Point)
    recommendation = "OK"
    suggested_order = 0
    
    if req.current_stock < q_high * 3: # Если остатка меньше, чем сожрут по верхней границе за 3 дня
        recommendation = "REPLENISHMENT REQUIRED"
        suggested_order = int((q_high * 7) - req.current_stock) # Заказываем на неделю вперед
    
    return {
        "forecast": {
            "p5_low": round(q_low, 2),
            "p60_median": round(q_med, 2),
            "p98_high": round(q_high, 2)
        },
        "business_decision": {
            "status": recommendation,
            "suggested_order_quantity": max(0, suggested_order),
            "reason": f"Current stock {req.current_stock} is below safety buffer based on p98 forecast."
        }
    }