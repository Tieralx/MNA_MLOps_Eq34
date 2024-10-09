import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
#importar clase regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Creating the classes

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

class preprocess_data:
    @staticmethod
    def delete_outliers(self):
        print("Aplicando operación: Borrado de outliers")
        numeric_columns = self.select_dtypes(include='number').columns
        # calcualr el IQR para cada columna
        Q1 = self[numeric_columns].quantile(0.25)
        Q3 = self[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1

        # calcula los limites inferiores y superiores
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # se determina una df the valores booleanos: True si es un outlier, False si no
        outliers = ((self[numeric_columns] < lower_bound) | (self[numeric_columns] > upper_bound))

        #se usa el df de booleanos para filtrar los outliers
        cleaned_data = self[~outliers.any(axis=1)]
        return cleaned_data
    
    @staticmethod
    def normalization(self):
        print("Aplicando operación: Normalización logaritmica y raiz cuadrada")
        skew = self.skew()
        log_transform_columns = []
        sqrt_transform_columns = []
        for index, value in skew.items():
            if value <= -1: 
                log_transform_columns.append(index)
            elif value > -1 and value < -0.5:
                sqrt_transform_columns.append(index)

        normalized = self.copy()

        # Transformación logarítmica: Reduce el impacto de valores extremos. Ideal para variables con sesgo positivo.
        normalized[log_transform_columns] = normalized[log_transform_columns].apply(np.log1p)

        # Transformación de raíz cuadrada: Similar a la logarítmica, pero menos agresiva. Funciona bien para variables con valores más pequeños o negativos con sesgo positivo o negativo.
        normalized[sqrt_transform_columns] = normalized[sqrt_transform_columns].apply(np.sqrt)

        return normalized
    
    @staticmethod
    def aplicar_pca(self, target, threshold = 0.90):
        print("Aplicando operación: PCA")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.drop(columns=[target]))  # Excluir la variable objetivo
        # Aplicar PCA
        pca = PCA().fit(X_scaled)
        # Calcular la varianza explicada acumulada
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= threshold) + 1
        pca = PCA(n_components=num_components)  # Número de componentes principales
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]  # Nombrar los componentes como PC1, PC2, etc.
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)
        df_pca['y'] = self[target].values
        return df_pca



class cervical_cancer_model:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model_pipeline = Pipeline(
            [
           
             ('regressor', LogisticRegression(random_state= 10, solver = 'liblinear', multi_class='ovr'))   
            ])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        df = pd.read_csv(self.filepath)
        DataExplorer.explore_data(df)
        self.data = df
        return self
    
    def preprocess_pipe(self):
        del_outliers = preprocess_data.delete_outliers(self.data)
        normalized = preprocess_data.normalization(del_outliers)
        applied_pca = preprocess_data.aplicar_pca(normalized,'ca_cervix',0.9)
        X = applied_pca.drop('y', axis=1)
        y = applied_pca['y']       
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return self
    
    def train_model(self):
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self
    
    def evaluate_model(self):
        print("Model Evaluation:")
        y_pred = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_test))
        disp.plot(cmap='Blues', colorbar=False)
        plt.show()
        
        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)
        return self
    
    def cross_validate_model(self):
        scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
        print("Average Accuracy with CV:", np.mean(scores))
        return self
    
def main():
    filepath=r'MNA_MLOps_Eq34\data\raw\sobar-72.csv'
    model =cervical_cancer_model(filepath)
    (model.load_data()
          .preprocess_pipe()
          .train_model()
          .evaluate_model()
          .cross_validate_model())

if __name__ == '__main__':
    main()