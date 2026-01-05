import pandas as pd
import pickle
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        self.report_lines = []
        
    def add(self, text=""):
        self.report_lines.append(text)
        
    def add_header(self, text, level=1):
        self.add("#" * level + " " + text)
        self.add()
        
    def load_data(self):
        with open('results/processed_data.pkl', 'rb') as f:
            self.data_info = pickle.load(f)
        
        self.results_df = pd.read_csv('results/table1_results.csv')
        self.data_stats = pd.read_csv('results/data_info.csv')
        
    def generate_report(self):
        self.add_header("Отчет по машинному обучению", 1)
        self.add_header("Анализ выбросов загрязняющих веществ в атмосферу (Казахстан)", 2)
        self.add()
        self.add(f"**Дата:** {datetime.now().strftime('%d.%m.%Y')}")
        self.add()
        self.add("---")
        self.add()
        
        self.add_header("1. Описание датасета", 2)
        self.add(f"**Источник данных:** stat.gov.kz")
        self.add(f"**Целевая переменная:** {self.data_info['target_name']}")
        self.add()
        
        stats = self.data_stats.iloc[0]
        self.add(f"- Всего образцов: {stats['total_samples']}")
        self.add(f"- Обучающая выборка: {stats['train_samples']}")
        self.add(f"- Тестовая выборка: {stats['test_samples']}")
        self.add(f"- Количество признаков: {stats['num_features']}")
        self.add(f"  - Категориальных: {stats['categorical_features_count']}")
        self.add(f"  - Числовых: {stats['numeric_features_count']}")
        self.add()
        
        self.add_header("2. Методология", 2)
        self.add("**Используемые алгоритмы:**")
        self.add()
        for i, row in self.results_df.iterrows():
            self.add(f"{i+1}. {row['Algorithm']}")
        self.add()
        self.add(f"**Валидация:** {self.results_df.iloc[0]['k-fold validation']}-fold кросс-валидация")
        self.add()
        self.add("**Метрики оценки:**")
        self.add("- RMSE (Root Mean Squared Error) - среднеквадратичная ошибка")
        self.add("- R² (Coefficient of Determination) - коэффициент детерминации")
        self.add()
        
        self.add_header("3. Результаты", 2)
        self.add()
        
        self.add("| Алгоритм | Признаков | Целей | K-fold | CV RMSE | Test RMSE | R² |")
        self.add("|----------|-----------|-------|--------|---------|-----------|-----|")
        
        for _, row in self.results_df.iterrows():
            self.add(f"| {row['Algorithm']} | {row['Number of features']} | {row['Number of targets']} | "
                    f"{row['k-fold validation']} | {row['CV RMSE']:.2f} | {row['Test RMSE']:.2f} | {row['R2']:.4f} |")
        
        self.add()
        
        self.add_header("4. Анализ результатов", 2)
        
        best_model = self.results_df.iloc[0]
        worst_model = self.results_df.iloc[-1]
        
        self.add(f"**Лучшая модель:** {best_model['Algorithm']}")
        self.add(f"- Test RMSE: {best_model['Test RMSE']:.2f}")
        self.add(f"- R²: {best_model['R2']:.4f}")
        self.add()
        
        self.add(f"**Худшая модель:** {worst_model['Algorithm']}")
        self.add(f"- Test RMSE: {worst_model['Test RMSE']:.2f}")
        self.add(f"- R²: {worst_model['R2']:.4f}")
        self.add()
        
        avg_r2 = self.results_df['R2'].mean()
        self.add(f"**Средний R² по всем моделям:** {avg_r2:.4f}")
        self.add()
        
        top3 = self.results_df.head(3)
        self.add("**ТОП-3 модели:**")
        for i, row in top3.iterrows():
            self.add(f"{i+1}. {row['Algorithm']} (RMSE={row['Test RMSE']:.2f}, R²={row['R2']:.4f})")
        self.add()
        
        self.add_header("5. Выводы", 2)
        
        if best_model['R2'] > 0.8:
            quality = "отличное"
        elif best_model['R2'] > 0.6:
            quality = "хорошее"
        elif best_model['R2'] > 0.4:
            quality = "удовлетворительное"
        else:
            quality = "низкое"
            
        self.add(f"1. Лучший результат показала модель **{best_model['Algorithm']}** с {quality} качеством прогнозирования (R²={best_model['R2']:.4f}).")
        self.add()
        
        ensemble_models = self.results_df[self.results_df['Algorithm'].str.contains('Boost|Trees')]['R2'].mean()
        linear_models = self.results_df[self.results_df['Algorithm'].str.contains('Ridge|Lasso|Elastic')]['R2'].mean()
        
        if ensemble_models > linear_models:
            self.add(f"2. Ансамблевые методы (средний R²={ensemble_models:.4f}) показали лучшие результаты по сравнению с линейными моделями (средний R²={linear_models:.4f}).")
        else:
            self.add(f"2. Линейные модели (средний R²={linear_models:.4f}) показали результаты сопоставимые с ансамблевыми методами (средний R²={ensemble_models:.4f}).")
        self.add()
        
        self.add(f"3. Использование {self.results_df.iloc[0]['k-fold validation']}-fold кросс-валидации обеспечило надежную оценку обобщающей способности моделей.")
        self.add()
        
        self.add("---")
        self.add()
        self.add("*Отчет сгенерирован автоматически*")
        
    def save_report(self, filename='report/ОТЧЕТ.md'):
        import os
        os.makedirs('report', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        
        print(f"Отчет сохранен: {filename}")
        
    def run(self):
        print("\n=== ГЕНЕРАЦИЯ ОТЧЕТА ===\n")
        self.load_data()
        self.generate_report()
        self.save_report()
        print("\n=== ГОТОВО ===\n")

if __name__ == '__main__':
    generator = ReportGenerator()
    generator.run()