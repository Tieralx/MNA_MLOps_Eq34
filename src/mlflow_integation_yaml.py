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
from mlflow.models import infer_signature
import yaml

#yaml file load
with open("MNA_MLOps_Eq34\src\params.yaml") as config_file:
    config_params = yaml.safe_load(config_file)

# Clase del Modelo
class CervicalCancerModel:
    def __init__(self, filepath, target= config_params["data"]["target_column"]):
        self.filepath = filepath
        self.target = target
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=config_params["preprocessing"]["PCA_threshold"])),  # Aplicar PCA para reducir dimensiones, basado en un umbral del 90%
            ('classifier', LogisticRegression(solver=config_params["train"]["solver"], multi_class=config_params["train"]["multi_class"], random_state=config_params["train"]["random_state"], max_iter=config_params["train"]["max_iter"] ))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        # Cargar los datos
        self.data = pd.read_csv(self.filepath)
             
        mlflow.log_input(mlflow.data.from_pandas(self.data), context="raw_data")
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
        
        
        mlflow.log_input(mlflow.data.from_pandas(self.data), context="preprocessed_data")
        # Dividir los datos en características (X) y objetivo (y)
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        # Dividir el dataset en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=config_params["data_split"]["test_size"], random_state=config_params["data_split"]["random_state"], shuffle=config_params["data_split"]["shuffle"])
        return self

    def train_model(self):

        
        
            # Registrar parámetros del modelo
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_param("pca_n_components", config_params["preprocessing"]["PCA_threshold"])
            mlflow.log_param("classifier", "LogisticRegression")
            mlflow.log_param("model_solver" ,config_params["train"]["solver"])
            mlflow.log_param("model_multi_class", config_params["train"]["multi_class"]) 
            mlflow.log_param("model_random_state",config_params["train"]["random_state"])
            mlflow.log_param("model_max_iter", config_params["train"]["max_iter"])
            
            # Entrenar el modelo
            self.model_pipeline.fit(self.X_train, self.y_train)
            
            # Imprimir los componentes principales seleccionados por PCA
            pca = self.model_pipeline.named_steps['pca']
            num_components = pca.n_components_
            mlflow.log_param("n_components", num_components)
            print(f"Número total de componentes seleccionados por PCA: {num_components}")
            
            # Predecir los valores del conjunto de prueba
            y_pred = self.model_pipeline.predict(self.X_test)
            signature = infer_signature(self.X_test, y_pred)
            
            # Registrar accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy: {accuracy}")
            
            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=config_params["reports"]["target_names"] , columns=config_params["reports"]["target_names"])
            print("Matriz de Confusión:")
            print(cm_df)
            # Guardar la matriz de confusión como una imagen
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap=config_params["reports"]["cmap"], cbar = config_params["reports"]["colorbar"] ) 
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            
            # Reporte de clasificación
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=config_params["reports"]["target_names"], zero_division=config_params["reports"]["clasification_report_zero_division"]))
            report = classification_report(self.y_test, y_pred, target_names=config_params["reports"]["target_names"], zero_division=config_params["reports"]["clasification_report_zero_division"], 
                                           output_dict=config_params["reports"]["clasification_report_output_dict"])
            mlflow.log_metric("Accuracy",report['accuracy'])
            mlflow.log_metric("Presicion",report['macro avg']['precision'])
            mlflow.log_metric("Recall",report['macro avg']['recall'])
            mlflow.log_metric("F1 score",report['macro avg']['f1-score'])
            #Cross validation
            scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=config_params["reports"]["cv"])
            avg_score = np.mean(scores)
            print("Average Accuracy with CV:", avg_score)
            mlflow.log_metric("average_accuracy_cv", avg_score)
            mlflow.sklearn.log_model(
                sk_model = self.model_pipeline,
                artifact_path=config_params["mlflow_config"]["model_artifact_path"],
                signature=signature,
                registered_model_name=config_params["mlflow_config"]["model_name"]
            )
            
# Función principal para ejecutar el pipeline
def main():
    mlflow.set_tracking_uri(uri=config_params["mlflow_config"]["uri"])
    # Create a new MLflow Experiment
    mlflow.set_experiment(config_params["mlflow_config"]["experiment_name"])
    with mlflow.start_run():
        filepath = config_params["data"]["file_path"] 
        model = CervicalCancerModel(filepath)
        (model.load_data()
            .preprocess_data()  # Aqui se eliminan outliers y se normaliza el dataset completo antes del split
            .train_model())  # Entrenar el modelo usando el pipeline que incluye escalado y PCA
        

if __name__ == '__main__':
    main()