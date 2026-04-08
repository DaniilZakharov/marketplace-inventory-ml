import pandas as pd
import numpy as np
from src.app import get_features  # Импортируем твою функцию

def test_no_data_leakage():
    # 1. Создаем фейковую историю
    # Пусть мы хотим предсказать продажи на 2024-01-10
    target_date_str = "2024-01-10"
    
    # Создаем данные, включая саму дату прогноза и даже будущее
    dates = pd.date_range(start="2023-12-01", end="2024-01-15")
    data = {
        'date': dates,
        'store': [1] * len(dates),
        'item': [1] * len(dates),
        'sales': [float(i) for i in range(len(dates))] # Продажи растут каждый день: 0, 1, 2...
    }
    mock_history = pd.DataFrame(data)
    
    # Подменяем глобальную переменную history_df в модуле app (monkeypatching)
    import src.app
    src.app.history_df = mock_history
    
    # 2. Вызываем расчет признаков
    features_df = get_features(target_date_str, store=1, item=1)
    
    # Находим значение продаж в "день прогноза" (target_date) в нашей фейковой истории
    today_sales = mock_history[mock_history['date'] == pd.to_datetime(target_date_str)]['sales'].values[0]
    # Находим значение за "завтра" (будущее)
    tomorrow_sales = mock_history[mock_history['date'] > pd.to_datetime(target_date_str)]['sales'].values[0]
    
    # 3. ПРОВЕРКИ (Asserts)
    
    # Проверяем лаг 1: он должен быть равен продажам за ВЧЕРА (т.е. today_sales - 1)
    # Если там значение today_sales — значит у нас утечка!
    assert features_df['sales_lag_1'].iloc[0] < today_sales, "Утечка! sales_lag_1 использует данные сегодняшнего дня"
    
    # Проверяем скользящее среднее: оно не должно включать в себя сегодняшние или будущие продажи
    all_feature_values = [
        features_df['sales_lag_1'].iloc[0],
        features_df['sales_lag_7'].iloc[0],
        features_df['sales_lag_30'].iloc[0],
        features_df['sales_rolling_mean_7'].iloc[0],
        features_df['sales_rolling_mean_30'].iloc[0]
    ]
    
    for val in all_feature_values:
        assert val != today_sales, f"Критическая ошибка: значение {val} совпадает с продажами за сегодня!"
        assert val != tomorrow_sales, f"Критическая ошибка: значение {val} совпадает с данными из будущего!"

    print("✅ Тест пройден: Утечек данных в признаках не обнаружено!")

if __name__ == "__main__":
    test_no_data_leakage()