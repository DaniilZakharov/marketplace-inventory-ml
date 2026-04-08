import pandas as pd
import numpy as np  
from catboost import CatBoostRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import os


train = pd.read_csv('data/train_cleaned.csv', parse_dates=['date'])

cat_features = ['store', 'item', 'day_of_week', 'month', 'year', 'is_holiday', 'is_shopping_day']
FEATURES = cat_features + ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7', "sales_rolling_mean_30", "sales_lag_30"]



def recursive_forecast_month(model, history_df, forecast_dates_df):
    """
    Версия с поддержкой MultiQuantile (3 колонки прогноза).
    """
    # Рабочий буфер: история продаж
    buffer = history_df[["date", "store", "item", "sales"]].copy()
    
    # Уникальные пары магазин-товар
    pairs = forecast_dates_df[["store", "item"]].drop_duplicates()
    results = []
    forecast_days = sorted(forecast_dates_df["date"].unique())

    for day in forecast_days:
        day_rows = forecast_dates_df[forecast_dates_df["date"] == day].copy()

       
        cutoff_date = day - pd.Timedelta(days=31)
        recent = buffer[buffer["date"] >= cutoff_date].copy()

        lag_rows = []
        for _, pair_row in pairs.iterrows():
            s, i = pair_row["store"], pair_row["item"]
            hist = recent[(recent["store"] == s) & (recent["item"] == i)].sort_values("date")
            
            # Вспомогательная функция для лага
            def get_lag(n):
                target = day - pd.Timedelta(days=n)
                val = hist[hist["date"] == target]["sales"].values
                return val[0] if len(val) > 0 else np.nan

            l1, l7, l30 = get_lag(1), get_lag(7), get_lag(30)
            
            # Rolling mean
            w7 = hist[(hist["date"] >= day - pd.Timedelta(days=7)) & (hist["date"] < day)]["sales"].values
            w30 = hist[(hist["date"] >= day - pd.Timedelta(days=30)) & (hist["date"] < day)]["sales"].values
            
            lag_rows.append({
                "store": s, "item": i,
                "sales_lag_1": l1, "sales_lag_7": l7, "sales_lag_30": l30,
                "sales_rolling_mean_7": np.mean(w7) if len(w7) > 0 else np.nan,
                "sales_rolling_mean_30": np.mean(w30) if len(w30) > 0 else np.nan
            })

        lag_df = pd.DataFrame(lag_rows)
        day_rows = day_rows.merge(lag_df, on=["store", "item"], how="left")

       
        X_day = day_rows[FEATURES]
        preds = model.predict(X_day)
        preds = np.maximum(preds, 0)

        day_rows = day_rows.copy()
        # Раскладываем 3 колонки квантилей
        day_rows["lower_ci"] = preds[:, 0]       # q=0.05
        day_rows["predicted_sales"] = preds[:, 1] # медиана
        day_rows["upper_ci"] = preds[:, 2]       # q=0.95

        
        pred_buffer = day_rows[["date", "store", "item"]].copy()
        pred_buffer["sales"] = preds[:, 1] # Для лагов используем медиану
        buffer = pd.concat([buffer, pred_buffer], ignore_index=True)

        results.append(day_rows)

    return pd.concat(results, ignore_index=True)


def add_lag_features(df):
    df = df.copy()
    # Создаем лаги
    df['sales_lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    df['sales_lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
    df['sales_lag_30'] = df.groupby(['store', 'item'])['sales'].shift(30)
    
    # Создаем скользящие средние
    # Используем shift(1), чтобы не было утечки данных 
    df['sales_rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df['sales_rolling_mean_30'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    return df








# Выбираем последнюю точку для теста (декабрь 2017)
cutoff = train["date"].max() - pd.DateOffset(days=30) 
cutoff = cutoff.replace(day=1) # Округляем до начала месяца

print(f"Запуск...")
print(f"Train: до {cutoff.date()} | Test: Декабрь 2017")

# 1. Разделение
train_fold = train[train["date"] < cutoff].copy()
test_fold = train[train["date"] >= cutoff].copy()

# 2. Пересчет признаков (Честная статика)
cols_to_drop = ["store_avg_sales", "item_avg_sales", "sales_lag_1", "sales_lag_7", 
                "sales_lag_30", "sales_rolling_mean_7", "sales_rolling_mean_30"]

train_fold = train_fold.drop(columns=[c for c in cols_to_drop if c in train_fold.columns])
test_fold = test_fold.drop(columns=[c for c in cols_to_drop if c in test_fold.columns])

s_avg = train_fold.groupby("store")["sales"].mean().rename("store_avg_sales")
i_avg = train_fold.groupby("item")["sales"].mean().rename("item_avg_sales")

train_fold = train_fold.join(s_avg, on="store").join(i_avg, on="item")
test_fold = test_fold.join(s_avg, on="store").join(i_avg, on="item")

# Лаги для обучения
train_fold = add_lag_features(train_fold).dropna()





# 3. Обучение
X_tr, y_tr = train_fold[FEATURES], train_fold["sales"]
val_cutoff = cutoff - pd.DateOffset(months=1)
X_val = train_fold[train_fold["date"] >= val_cutoff][FEATURES]
y_val = train_fold[train_fold["date"] >= val_cutoff]["sales"]

model = CatBoostRegressor(
    iterations=1000,
    # Мы ставим 0.6 как "базовый" прогноз (медиану), чтобы модель была чуть оптимистичнее
    # И 0.98 для верхней границы, чтобы почти исключить дефицит
    loss_function='MultiQuantile:alpha=0.05,0.6,0.98', 
    verbose=0
)

model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

# 4. Рекурсивный прогноз
history = train_fold[["date", "store", "item", "sales"]].copy()
forecast_df = recursive_forecast_month(model, history, test_fold)

# 5. Метрики (с исправлением MAPE)
y_true = forecast_df["sales"]
y_pred = forecast_df["predicted_sales"]

m_mae = mean_absolute_error(y_true, y_pred)
m_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# Заменяем MAPE на "безопасный" вариант, чтобы не делить на 0
m_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100 

print(f"\nРезультаты фолда:")
print(f"MAE: {m_mae:.2f} | RMSE: {m_rmse:.2f} | MAPE: {m_mape:.2f}%")

# Сохраняем модель, чтобы Docker мог её "отдать"
os.makedirs('models', exist_ok=True)
model.save_model('models/catboost_model.cbm')
print("Модель сохранена в models/catboost_model.cbm")


