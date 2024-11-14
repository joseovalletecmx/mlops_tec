import yaml
import sys
from typing import Text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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
    model = LogisticRegression()
    model.fit(X_train, np.array(y_train).flatten())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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
    mlflow.set_experiment(f'users/appendicitis_model')

    model_path = config['data_model']['model_path']
    joblib.dump(model, model_path)

    mlflow.start_run()
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision_score', precision)
    mlflow.log_metric('recall_score', recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.end_run()

if __name__ == '__main__':
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Train random forest and export model')
    args_parser.add_argument('--config', dest ='config', required=True)
    args = args_parser.parse_args()    
    # Call the function with command line arguments
    train_model(config_path= args.config)
