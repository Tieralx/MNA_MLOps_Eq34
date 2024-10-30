import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import yaml


# Cargar archivo YAML con configuración



def preprocess_data(data_path):
        # Preprocesar los datos: eliminar outliers y normalizar
        target=config_params["data"]["target_column"]
        data = pd.read_csv(data_path)
        numeric_columns = data.select_dtypes(include='number').columns
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound))
        data = data[~outliers.any(axis=1)]

        skew = data.skew()
        log_transform_columns = [index for index, value in skew.items() if value <= -1]
        sqrt_transform_columns = [index for index, value in skew.items() if -1 < value < -0.5]

        data[log_transform_columns] = data[log_transform_columns].apply(np.log1p)
        data[sqrt_transform_columns] = data[sqrt_transform_columns].apply(np.sqrt)
        #data.to_csv(r"MNA_MLOps_Eq34\data\processed\cervical_cancer_processed.csv", index=False)

 
        # Dividir los datos en características (X) y objetivo (y)
        X = data.drop(target, axis=1)
        y = data[target]
        
        # Dividir el dataset en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config_params["data_split"]["test_size"], random_state=config_params["data_split"]["random_state"],
            shuffle=config_params["data_split"]["shuffle"], stratify=y
        )
        return X_train, X_test, y_train, y_test
        
if __name__ == '__main__':
    data_path = sys.argv[1]
    yaml_file = sys.argv[2]
    output_train_features = sys.argv[3]
    output_test_features = sys.argv[4]
    output_train_target = sys.argv[5]
    output_test_target = sys.argv[6]
    
    
    with open(yaml_file) as config_file:
        config_params = yaml.safe_load(config_file)
    
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)
    
    