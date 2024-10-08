# Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Exploración y visualización de datos
class DataExplorer:
    @staticmethod
    def explore_data(data):
        print(data.head().T)
        print(data.describe())
        print(data.info())
    
    @staticmethod
    def plot_histograms(data):
        data.hist(bins=15, figsize=(15, 10))
        plt.show()

    @staticmethod
    def plot_correlation_matrix(data):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.show()
        
    @staticmethod
    def plot_feature_relationships(data, target):
        for column in data.columns[:-1]:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=target, y=column, data=data)
            plt.title(f'Relationship between Cervical cancer and {column}')
            plt.show()

# Preprocesamiento: Normalización y PCA
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target, threshold=0.90):
        self.target = target
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.pca = None
    
    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.threshold).fit(X_scaled)
        return self
    
    def transform(self, X, y=None):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
    
    def delete_outliers(self, data):
        numeric_columns = data.select_dtypes(include='number').columns
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound))
        return data[~outliers.any(axis=1)]
    
    def normalization(self, data):
        skew = data.skew()
        log_transform_columns = [index for index, value in skew.items() if value <= -1]
        sqrt_transform_columns = [index for index, value in skew.items() if -1 < value < -0.5]
        
        normalized = data.copy()
        normalized[log_transform_columns] = normalized[log_transform_columns].apply(np.log1p)
        normalized[sqrt_transform_columns] = normalized[sqrt_transform_columns].apply(np.sqrt)
        return normalized

# Clase del Modelo
class CervicalCancerModel:
    def __init__(self, filepath, target='ca_cervix'):
        self.filepath = filepath
        self.target = target
        self.model_pipeline = Pipeline([
            ('preprocessor', DataPreprocessor(target=self.target)),
            ('classifier', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=10))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        DataExplorer.explore_data(self.data)
        return self

    def preprocess_data(self):
        # Removing outliers and normalizing the data
        preprocessor = DataPreprocessor(target=self.target)
        self.data = preprocessor.delete_outliers(self.data)
        self.data = preprocessor.normalization(self.data)
        
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self

    def train_model(self):
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self
    
    def evaluate_model(self):
        y_pred = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)
        return self
    
    def cross_validate_model(self):
        scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
        print("Average Accuracy with CV:", np.mean(scores))
        return self

# Función principal para ejecutar el pipeline
def main():
    filepath = 'sobar-72.csv'
    model = CervicalCancerModel(filepath)
    (model.load_data()
          .preprocess_data()
          .train_model()
          .evaluate_model()
          .cross_validate_model())

if __name__ == '__main__':
    main()
