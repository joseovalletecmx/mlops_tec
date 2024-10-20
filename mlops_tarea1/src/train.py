import yaml
import sys
from typing import Text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
import joblib
import mlflow

def train_model(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

# Load datasets
    X_train = pd.read_csv(config['data_split']['x_train_path'])
    y_train = pd.read_csv(config['data_split']['y_train_path'])
    X_test  = pd.read_csv(config['data_split']['x_test_path'])
    y_test  = pd.read_csv(config['data_split']['y_test_path'])

    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()

# Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, np.array(y_train).flatten())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Print accuracy and confusion matrix
    print("Accuracy of the model:", round(accuracy,4))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize the confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion, annot = True)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(f'users/test_pipeline_v3')

    model_path = config['data_model']['model_path']
    joblib.dump(model, model_path)

    mlflow.start_run()
    model.fit(X_train,y_train)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_params('n_estimators',100)
    mlflow.log_params('random_state',42)
    mlflow.log_metric('acc', accuracy)
    mlflow.end_run()



if __name__ == '__main__':
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Train random forest and export model')
    args_parser.add_argument('--config', dest ='config', required=True)
    args = args_parser.parse_args()    
    # Call the function with command line arguments
    train_model(config_path= args.config)
