import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

# Clase del Modelo
class CervicalCancerModel:
    def __init__(self, filepath, target='ca_cervix'):
        self.filepath = filepath
        self.target = target
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),  # Aplicar PCA para reducir dimensiones, basado en un umbral del 90%
            ('classifier', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=10))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        # Cargar los datos
        self.data = pd.read_csv(self.filepath)
        return self

    def preprocess_data(self):
        # Preprocesar los datos: eliminar outliers y normalizar
        numeric_columns = self.data.select_dtypes(include='number').columns
        Q1 = self.data[numeric_columns].quantile(0.25)
        Q3 = self.data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((self.data[numeric_columns] < lower_bound) | (self.data[numeric_columns] > upper_bound))
        self.data = self.data[~outliers.any(axis=1)]
        
        skew = self.data.skew()
        log_transform_columns = [index for index, value in skew.items() if value <= -1]
        sqrt_transform_columns = [index for index, value in skew.items() if -1 < value < -0.5]
        
        self.data[log_transform_columns] = self.data[log_transform_columns].apply(np.log1p)
        self.data[sqrt_transform_columns] = self.data[sqrt_transform_columns].apply(np.sqrt)
        
        # Dividir los datos en características (X) y objetivo (y)
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        # Dividir el dataset en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return self

    def train_model(self):

        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        # Create a new MLflow Experiment
        mlflow.set_experiment("Prueba Piloto MLFlow")
        with mlflow.start_run():
            # Registrar parámetros del modelo
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_param("pca_n_components", 0.90)
            mlflow.log_param("classifier", "LogisticRegression")
            
            # Entrenar el modelo
            self.model_pipeline.fit(self.X_train, self.y_train)
            
            # Imprimir los componentes principales seleccionados por PCA
            pca = self.model_pipeline.named_steps['pca']
            num_components = pca.n_components_
            mlflow.log_param("n_components", num_components)
            print(f"Número total de componentes seleccionados por PCA: {num_components}")
            
            # Predecir los valores del conjunto de prueba
            y_pred = self.model_pipeline.predict(self.X_test)
            
            # Registrar accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy: {accuracy}")
            
            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['without cervical cancer', 'with cervical cancer'], columns=['without cervical cancer', 'with cervical cancer'])
            print("Matriz de Confusión:")
            print(cm_df)
            # Guardar la matriz de confusión como una imagen
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            
            # Reporte de clasificación
            report = classification_report(self.y_test, y_pred, target_names=['without cervical cancer', 'with cervical cancer'], zero_division=0.0)
            print("Classification Report:")
            print(report)
            mlflow.log_text(report, "classification_report.txt")

    def cross_validate_model(self):
        with mlflow.start_run():
            # Validación cruzada para evaluar la generalización del modelo
            scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
            avg_score = np.mean(scores)
            print("Average Accuracy with CV:", avg_score)
            mlflow.log_metric("average_accuracy_cv", avg_score)
        return self

# Función principal para ejecutar el pipeline
def main():
    filepath = 'sobar-72.csv'
    model = CervicalCancerModel(filepath)
    (model.load_data()
          .preprocess_data()  # Aqui se eliminan outliers y se normaliza el dataset completo antes del split
          .train_model()  # Entrenar el modelo usando el pipeline que incluye escalado y PCA
          .cross_validate_model())  # Validar el modelo con validación cruzada

if __name__ == '__main__':
    main()