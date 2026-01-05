import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path='data/dataset.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        try:
            self.df = pd.read_csv(
                self.data_path, 
                sep='\t',
                quotechar='"',
                encoding='utf-8',
                thousands=' '
            )
            print(f"Загружено: {self.df.shape[0]} строк, {self.df.shape[1]} колонок")
        except:
            try:
                self.df = pd.read_csv(
                    self.data_path, 
                    sep='\t',
                    encoding='cp1251',
                    thousands=' '
                )
                print(f"Загружено: {self.df.shape[0]} строк, {self.df.shape[1]} колонок")
            except Exception as e:
                raise Exception(f"Не удалось загрузить данные: {e}")
    
    def clean_data(self):
        initial_rows = len(self.df)
        self.df.dropna(how='all', inplace=True)
        self.df.drop_duplicates(inplace=True)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        val_cols = [col for col in self.df.columns if 'VAL' in str(col).upper()]
        
        if val_cols:
            self.target_col = val_cols[0]
        elif len(numeric_cols) > 0:
            self.target_col = numeric_cols[-1]
        else:
            raise Exception("Не найдена целевая переменная")
        
        self.df = self.df[self.df[self.target_col].notna()]
        self.df = self.df[self.df[self.target_col] > 0]
        
        print(f"Целевая переменная: {self.target_col}")
        print(f"После очистки: {len(self.df)} строк")
        
    def prepare_features(self):
        exclude_cols = [self.target_col, 'NAM', 'DAT']
        
        self.categorical_features = []
        for col in self.df.columns:
            if col not in exclude_cols and self.df[col].dtype == 'object':
                self.categorical_features.append(col)
        
        self.numeric_features = []
        for col in self.df.columns:
            if col not in exclude_cols and col != self.target_col:
                if self.df[col].dtype in [np.int64, np.float64]:
                    self.numeric_features.append(col)
        
        for col in self.categorical_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        feature_cols = self.categorical_features + self.numeric_features
        self.X = self.df[feature_cols].copy()
        self.y = self.df[self.target_col].copy()
        
        print(f"Признаков: {self.X.shape[1]} ({len(self.categorical_features)} категор. + {len(self.numeric_features)} числ.)")
        
    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        
    def scale_features(self):
        if len(self.numeric_features) > 0:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
            
            self.X_train_scaled[self.numeric_features] = self.scaler.fit_transform(
                self.X_train[self.numeric_features]
            )
            self.X_test_scaled[self.numeric_features] = self.scaler.transform(
                self.X_test[self.numeric_features]
            )
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        
    def save_processed_data(self):
        os.makedirs('results', exist_ok=True)
        
        data_dict = {
            'X_train': self.X_train_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': list(self.X.columns),
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'target_name': self.target_col
        }
        
        with open('results/processed_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        
        info = {
            'total_samples': len(self.df),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'num_features': len(self.X.columns),
            'categorical_features_count': len(self.categorical_features),
            'numeric_features_count': len(self.numeric_features),
            'target_variable': self.target_col
        }
        
        pd.DataFrame([info]).to_csv('results/data_info.csv', index=False)
        print("Данные сохранены в results/")
        
    def run(self):
        print("\n=== ПРЕДОБРАБОТКА ДАННЫХ ===\n")
        self.load_data()
        self.clean_data()
        self.prepare_features()
        self.split_data()
        self.scale_features()
        self.save_processed_data()
        print("\n=== ГОТОВО ===\n")