from src.preprocessing import DataPreprocessor
from src.train_models import ModelTrainer
from src.generate_report import ReportGenerator

def main():
    print("\n" + "="*70)
    print("АНАЛИЗ ВЫБРОСОВ ЗАГРЯЗНЯЮЩИХ ВЕЩЕСТВ - МАШИННОЕ ОБУЧЕНИЕ")
    print("="*70 + "\n")
    
    preprocessor = DataPreprocessor()
    preprocessor.run()
    
    trainer = ModelTrainer()
    trainer.run()
    
    generator = ReportGenerator()
    generator.run()
    
    print("="*70)
    print("ВЕСЬ ПАЙПЛАЙН ВЫПОЛНЕН УСПЕШНО")
    print("="*70)
    print("\nПроверьте результаты:")
    print("  - results/table1_results.csv - таблица результатов")
    print("  - report/ОТЧЕТ.md - итоговый отчет")
    print()

if __name__ == '__main__':
    main()