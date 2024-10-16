import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from mlflow.models import infer_signature
import yaml

# Cargar archivo YAML con configuración
with open("MNA_MLOps_Eq34\src\params.yaml") as config_file:
    config_params = yaml.safe_load(config_file)

# Clase del Modelo con Grid Search
class CervicalCancerModel:
    def __init__(self, filepath, target=config_params["data"]["target_column"]):
        self.filepath = filepath
        self.target = target
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=config_params["preprocessing"]["PCA_threshold"])),  # Aplicar PCA
            ('classifier', LogisticRegression())
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config_params["data_split"]["test_size"], random_state=config_params["data_split"]["random_state"],
            shuffle=config_params["data_split"]["shuffle"]
        )
        return self

    def train_model_with_grid_search(self):
        # Configurar Grid Search con los parámetros del YAML
        param_grid = config_params["grid_search"]["param_grid"]
        grid_search = GridSearchCV(self.model_pipeline, param_grid, cv=config_params["grid_search"]["cv"], scoring=config_params["grid_search"]["scoring"])

        # Iniciar un experimento en MLflow
        mlflow.log_param("grid_search_params", param_grid)
        mlflow.log_param("cv_folds", config_params["grid_search"]["cv"])

        # Entrenar el modelo con búsqueda en cuadrícula
        grid_search.fit(self.X_train, self.y_train)

        # Registrar los resultados de todas las combinaciones
        cv_results = grid_search.cv_results_
        for i in range(len(cv_results['params'])):
            run_name = f"GridSearch Run {i+1}"
            with mlflow.start_run(nested=True, run_name=run_name):
                # Registrar los parámetros de la combinación actual
                mlflow.log_params(cv_results['params'][i])
                # Registrar la métrica media de validación cruzada (accuracy)
                mlflow.log_metric(str(run_name) + " mean_test_score", cv_results['mean_test_score'][i])
                # Registrar otras métricas opcionales si es necesario
                mlflow.log_metric(str(run_name) + " std_test_score", cv_results['std_test_score'][i])

                # Predecir con el modelo actual y registrar métricas adicionales
                y_pred = grid_search.best_estimator_.predict(self.X_test)
                report = classification_report(self.y_test, y_pred, target_names=config_params["reports"]["target_names"],
                                               zero_division=config_params["reports"]["clasification_report_zero_division"],
                                               output_dict=config_params["reports"]["clasification_report_output_dict"])
                mlflow.log_metric(" Accuracy", report['accuracy'])
                mlflow.log_metric(str(run_name) + " Precision", report['macro avg']['precision'])
                mlflow.log_metric(str(run_name) + " Recall", report['macro avg']['recall'])
                mlflow.log_metric(str(run_name) + " F1 score", report['macro avg']['f1-score'])

        # Obtener el mejor modelo y sus parámetros
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Registrar parámetros y métricas del mejor modelo
        mlflow.log_params(best_params)
        mlflow.log_metric("grid_search_best_cv_score", best_score)

        # Predecir con el mejor modelo
        y_pred = best_model.predict(self.X_test)
        signature = infer_signature(self.X_test, y_pred)

        # Registrar accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}")

        print("Classification Report:")
        report = classification_report(self.y_test, y_pred, target_names=config_params["reports"]["target_names"],
                                       zero_division=config_params["reports"]["clasification_report_zero_division"],
                                       output_dict=config_params["reports"]["clasification_report_output_dict"])
        mlflow.log_metric("Classifaction_report_Accuracy",report['accuracy'])
        mlflow.log_metric("Classifaction_report_Presicion",report['macro avg']['precision'])
        mlflow.log_metric("Classifaction_report_Recall",report['macro avg']['recall'])
        mlflow.log_metric("Classifaction_report_F1-score",report['macro avg']['f1-score'])
        print(yaml.dump(report))

        # Guardar el mejor modelo con MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=config_params["mlflow_config"]["model_artifact_path"],
            signature=signature,
            registered_model_name=config_params["mlflow_config"]["model_name"]
        )

# Función principal para ejecutar el pipeline
def main():
    mlflow.set_tracking_uri(uri=config_params["mlflow_config"]["uri"])
    mlflow.set_experiment(config_params["mlflow_config"]["experiment_name"])
    with mlflow.start_run(nested = True):
        filepath = config_params["data"]["file_path"]
        model = CervicalCancerModel(filepath)
        (model.load_data()
             .preprocess_data()
             .train_model_with_grid_search())

if __name__ == '__main__':
    main()