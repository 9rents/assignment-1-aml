# Анализ выбросов загрязняющих веществ в атмосферу (Казахстан)

## Описание проекта

Проект по курсу **Advanced Machine Learning (AML)** для анализа и прогнозирования объемов выбросов загрязняющих веществ в атмосферу от стационарных источников в Республике Казахстан.

**Датасет:** Статистические данные по выбросам в атмосферу (stat.gov.kz)

**Цель:** Сравнить производительность 10 различных алгоритмов регрессии для предсказания объема выбросов.

---

## Структура проекта

```
aml_assignment/
├── data/
│   └── dataset.csv                  # Исходный датасет
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook для анализа
├── src/
│   ├── preprocessing.py             # Предобработка данных
│   ├── train_models.py              # Обучение моделей
│   └── evaluate.py                  # Оценка результатов
├── results/
│   ├── table1_results.csv           # Итоговая таблица результатов
│   └── plots/                       # Графики визуализации
├── report/
│   └── ОТЧЕТ.md                     # Финальный отчет
├── requirements.txt                 # Зависимости Python
└── README.md                        # Этот файл
```

---

## Установка зависимостей

### Вариант 1: через pip
```bash
pip install -r requirements.txt
```

### Вариант 2: через uv (быстрее)
```bash
uv pip install -r requirements.txt
```

---

## Запуск проекта

### Шаг 1: Предобработка данных
```bash
cd aml_assignment
python src/preprocessing.py
```

**Что происходит:**
- Загрузка датасета из `data/dataset.csv`
- Очистка данных (удаление пропусков, выбросов)
- Кодирование категориальных признаков
- Разделение на обучающую и тестовую выборки
- Сохранение обработанных данных

### Шаг 2: Обучение моделей
```bash
python src/train_models.py
```

**Что происходит:**
- Обучение 10 алгоритмов регрессии:
  1. Ridge Regression
  2. Lasso Regression
  3. Elastic Net
  4. K-Nearest Neighbors (KNN)
  5. Extra Trees Regression
  6. Adaptive Boosting (AdaBoost)
  7. Gradient Boosting
  8. XGBoost
  9. LightGBM
  10. CatBoost
  11. HistGradientBoosting
- 10-fold кросс-валидация для каждой модели
- Сохранение обученных моделей

### Шаг 3: Оценка результатов
```bash
python src/evaluate.py
```

**Что происходит:**
- Расчет метрик (RMSE, R²) для каждой модели
- Создание таблицы результатов
- Генерация графиков сравнения
- Сохранение в `results/table1_results.csv`

### Шаг 4: Генерация отчета
```bash
python src/generate_report.py
```

**Что происходит:**
- Автоматическая генерация отчета в формате Markdown
- Включение таблиц и графиков
- Сохранение в `report/ОТЧЕТ.md`

---

## Альтернативный запуск: Jupyter Notebook

Для интерактивного анализа:

```bash
jupyter notebook notebooks/analysis.ipynb
```

В notebook выполни все ячейки последовательно.

---

## Используемые алгоритмы

| № | Алгоритм | Библиотека |
|---|----------|------------|
| 1 | Ridge Regression | scikit-learn |
| 2 | Lasso Regression | scikit-learn |
| 3 | Elastic Net | scikit-learn |
| 4 | KNN Regression | scikit-learn |
| 5 | Extra Trees | scikit-learn |
| 6 | AdaBoost | scikit-learn |
| 7 | Gradient Boosting | scikit-learn |
| 8 | XGBoost | xgboost |
| 9 | LightGBM | lightgbm |
| 10 | CatBoost | catboost |
| 11 | HistGradientBoosting | scikit-learn |

---

## Метрики оценки

- **RMSE (Root Mean Squared Error)** - среднеквадратичная ошибка
- **R² (R-squared)** - коэффициент детерминации

**Интерпретация:**
- RMSE: чем меньше, тем лучше (показывает среднюю ошибку предсказания)
- R²: чем ближе к 1, тем лучше (показывает долю объясненной дисперсии)

---

## Ожидаемый результат

После выполнения всех шагов получишь:

1. **Таблица результатов** (`results/table1_results.csv`) - заполненная Table 1 из задания
2. **Графики сравнения** (`results/plots/`) - визуализация производительности моделей
3. **Отчет** (`report/ОТЧЕТ.md`) - полный отчет с анализом

---

## Автор

**Студент:** [Grents Artem]  
**Курс:** Advanced Machine Learning (AML)  
**Дата:** January 2026