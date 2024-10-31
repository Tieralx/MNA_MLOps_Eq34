import pandas as pd
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import os
import json
from dvclive import Live
import mlflow
import yaml
from datetime import date




def evaluate_model(model_path, X_test_path, y_test_path, output_path, model_type):
    #os.makedirs(os.path.dirname(f"plots/{model_type}_cm.json"), exist_ok=True)
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
    
    write_evaluation_report(output_path, report, cm, model_type)
    predictions_pd = pd.concat([y_test,pd.DataFrame(predictions)], names=['y_true', 'y_pred'], axis = 1)
    mlflow.log_input(mlflow.data.from_pandas(X_test), context="X_test")
    mlflow.log_input(mlflow.data.from_pandas(y_test), context="y_test")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(predictions)), context="predictions")
    
    
    return predictions_pd

def write_evaluation_report(file_path, report, confusion_matrix, model_type):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(f"metrics/{model_type}_metrics.json"), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix))
    metrics = {
        f"Evaluate {model_type}" : {
            'accuracy' : report['accuracy'],
            'presicion' : report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        }
    }
    metrics_file_nm = f"metrics/{model_type}_metrics.json"
    with open(metrics_file_nm, 'w') as metrics_file:
        metrics_file.write(json.dumps(metrics, indent=2) + '\n')
    mlflow.log_metric("accuracy", report['accuracy'])
    mlflow.log_metric("precision", report['macro avg']['precision'])
    mlflow.log_metric("recall", report['macro avg']['recall'])
    mlflow.log_metric("F1-score", report['macro avg']['f1-score'])
    with Live() as live:
        live.log_metric("accuracy", report['accuracy'])
        live.log_metric("precision", report['macro avg']['precision'])
        live.log_metric("recall", report['macro avg']['recall'])
        live.log_metric("F1-score", report['macro avg']['f1-score'])
        
  
     
     
    

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    output_path = sys.argv[4]
    pred_path = sys.argv[5]
    yaml_file = sys.argv[6]
    model_type = sys.argv[7]
    with open(yaml_file, 'r') as yaml_file:
        config_params = yaml.safe_load(yaml_file)
    mlflow.set_tracking_uri(uri=config_params["mlflow_config"]["uri"])
    mlflow.set_experiment(config_params["mlflow_config"]["experiment_name"])
    run_name = f"{model_type}_run_{date.today()}"
    run_id = mlflow.search_runs(filter_string=f"run_name='{run_name}'")['run_id']
    with mlflow.start_run(run_id=run_id[0]):
     pred = evaluate_model(model_path, X_test_path, y_test_path, output_path, model_type)
    pd.DataFrame(pred).to_csv(pred_path, index=False)