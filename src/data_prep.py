import pandas as pd
import os

def prepare_data():
    # Используем пути от корня проекта (важно для Docker)
    input_path = 'data/train.csv'
    output_path = 'data/train_cleaned.csv'

    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден!")
        return

    print("--- Шаг 1: Загрузка данных ---")
    train = pd.read_csv(input_path, parse_dates=['date'])

    # Извлекаем компоненты даты
    train['day'] = train['date'].dt.day # Добавили, чтобы работала функция праздников
    train['day_of_week'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year

    print("--- Шаг 2: Создание лагов и признаков ---")
    # Группировка и сдвиги
    group_cols = ['store', 'item']
    train['sales_lag_1'] = train.groupby(group_cols)['sales'].shift(1)
    train['sales_lag_7'] = train.groupby(group_cols)['sales'].shift(7)
    train['sales_lag_30'] = train.groupby(group_cols)['sales'].shift(30)
    
    train['sales_rolling_mean_7'] = train.groupby(group_cols)['sales'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    train['sales_rolling_mean_30'] = train.groupby(group_cols)['sales'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )

    # Профили магазина и товара
    item_avg = train.groupby('item')['sales'].mean().rename('item_avg_sales')
    store_avg = train.groupby('store')['sales'].mean().rename('store_avg_sales')
    
    train = train.merge(item_avg, on='item', how='left')
    train = train.merge(store_avg, on='store', how='left')

    print("--- Шаг 3: Праздники и выходные ---")
    manual_holidays = [
        (1, 1), (12, 31), (12, 30), (2, 14), (2, 13), (7, 4), 
        (7, 3), (10, 31), (10, 30), (11, 24), (11, 23), (12, 25), (12, 24), (12, 23)
    ]

    train['is_holiday'] = train.apply(
        lambda row: 1 if (row.month, row.day) in manual_holidays else 0, axis=1
    )
    train['is_shopping_day'] = train['day_of_week'].isin([4, 5]).astype(int)

    # Чистим пустые строки (появившиеся из-за лагов)
    train.dropna(inplace=True)

    # Сохранение
    os.makedirs('data', exist_ok=True)
    train.to_csv(output_path, index=False)
    print(f"Файл сохранен: {output_path}")

if __name__ == "__main__":
    prepare_data()