# Demand Forecasting Project 

Проект по прогнозированию спроса на товары https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview

Цель: минимизация out-of-stock на складах.

## Основной функционал
- [ ] EDA и анализ сезонности (день недели, месяц).
- [ ] Обучение модели CatBoost для прогноза продаж.
- [ ] Генерация бизнес-рекомендаций: "Сколько и когда везти на склад".

## Технологии
- Python, Pandas, CatBoost.
- Git, Docker.
- LLM для интерпретации результатов модели.

## Как запустить
    Убедитесь, что у вас установлены драйверы NVIDIA и CUDA.

    Скачайте данные с https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview и положите в папку /data.

    Установите зависимости: pip install catboost pandas matplotlib.

    Запустите ноутбук 01_eda.ipynb

## Доделать
 Применить Детрендинг (Detrending):

    Сначала ты вычитаешь из продаж общий тренд (линейный рост).

    Обучаешь CatBoost предсказывать "остатки" (колебания вокруг этого тренда).

    При предсказании складываешь результат.