import yaml
import sys
from typing import Text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import argparse
import joblib
import mlflow
import shap

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


    print(X_train.info())

# Initialize classifier
    model = LogisticRegression()
    model.fit(X_train, np.array(y_train).flatten())
    y_pred = model.predict(X_test)

    # calculate evaluation metris
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print accuracy and confusion matrix
    print("Accuracy of the model:", round(accuracy,4))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


    # SHAP analysis
    k = len(X_train)
    background_sampled = shap.sample(X_train, k)

    # Calculate SHAP values using the KernelExplainer (logistic regression is linear)
    explainer = shap.Explainer(model, background_sampled)
    shap_values = explainer(X_test)

    # Visualize SHAP results
    shap.plots.bar(shap_values, show = False) # this one works too
    plt.savefig(config['explainer']['explainer_path'])
    plt.close()

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(f'users/appendicitis_model')

    model_path = config['data_model']['model_path']
    joblib.dump(model, model_path)
    
    mlflow.start_run()
    mlflow.sklearn.log_model(model, "model", registered_model_name="sk-learn-logistic_regression-reg-model")
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision_score', precision)
    mlflow.log_metric('recall_score', recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_artifact(config['explainer']['explainer_path'], "explainability_artifacts")
    mlflow.end_run()

if __name__ == '__main__':
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Train and export model')
    args_parser.add_argument('--config', dest ='config', required=True)
    args = args_parser.parse_args()    
    # Call the function with command line arguments
    train_model(config_path= args.config)
