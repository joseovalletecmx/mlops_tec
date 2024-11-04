import pandas as pd
import sys
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from typing import Text
import yaml
import argparse


def evaluate_model(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    model = joblib.load(config['data_model']['model_path'])
    X_test = pd.read_csv(config['data_split']['x_test_path'])
    y_test = pd.read_csv(config['data_split']['y_test_path'])
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1)
    matrix = confusion_matrix(y_test, predictions)

    report_path = config['data_model']['model_report_path']
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

if __name__ == '__main__':
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Evaluate model and export report')
    args_parser.add_argument('--config', dest ='config', required=True)
    args = args_parser.parse_args()    
    # Call the function with command line arguments
    evaluate_model(config_path= args.config)