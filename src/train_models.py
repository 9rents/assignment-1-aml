import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = []
        
    def load_data(self):
        with open('results/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']
        self.target_name = data['target_name']
        
        print(f"Загружены данные: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
        
    def initialize_models(self):
        self.models = {
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'Elastic Net': ElasticNet(random_state=42),
            'KNN Regression': KNeighborsRegressor(n_neighbors=5),
            'Extra Trees Regression': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Adaptive Boosting': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
            'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
            'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=42)
        }
        
        print(f"Инициализировано {len(self.models)} моделей")
        
    def train_and_evaluate(self, k_folds=10):
        print(f"\n=== ОБУЧЕНИЕ И ОЦЕНКА ({k_folds}-fold CV) ===\n")
        
        for name, model in self.models.items():
            print(f"{name}...", end=" ")
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=k_folds, scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_r2 = r2_score(self.y_test, y_pred)
            
            self.results.append({
                'Algorithm': name,
                'Number of features': self.X_train.shape[1],
                'Number of targets': 1,
                'k-fold validation': k_folds,
                'CV RMSE': cv_rmse,
                'Test RMSE': test_rmse,
                'R2': test_r2
            })
            
            print(f"RMSE={test_rmse:.2f}, R2={test_r2:.4f}")
            
    def save_results(self):
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values('Test RMSE')
        df_results.to_csv('results/table1_results.csv', index=False)
        
        print(f"\nРезультаты сохранены в results/table1_results.csv")
        
    def run(self):
        print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===\n")
        self.load_data()
        self.initialize_models()
        self.train_and_evaluate(k_folds=10)
        self.save_results()
        print("\n=== ГОТОВО ===\n")

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()