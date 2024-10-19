import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__": 
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment = mlflow.set_experiment("pediatric_appendicitis_experiment")

    print("mlflow tracking uri:", mlflow.tracking.get_tracking_uri())
    print("experiment:", experiment)
    warnings.filterwarnings("ignore")

    # Cargar el dataset
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)
    df_X = regensburg_pediatric_appendicitis.data.features
    df_y = regensburg_pediatric_appendicitis.data.targets

    # Imputar valores numéricos con la mediana
    df_numeric = df_X.select_dtypes(include=['float64', 'int64'])
    df_X[df_numeric.columns] = df_X[df_numeric.columns].fillna(df_X[df_numeric.columns].median())

    # Imputar valores categóricos con la moda
    categorical_cols = df_X.select_dtypes(exclude=np.number).columns
    df_X[categorical_cols] = df_X[categorical_cols].fillna(df_X[categorical_cols].mode().iloc[0])

    # Escalamiento de las variables
    df_encoded = pd.get_dummies(df_X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # Dividir en entrenamiento (80%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df_y, test_size=0.3, random_state=42)

    # Normalizar las variables numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_X.select_dtypes(include=[np.number]))
    X_scaled_df = pd.DataFrame(X_scaled, columns=df_X.select_dtypes(include=[np.number]).columns)

    # Usar One-Hot Encoding para variables categóricas
    X_encoded = pd.get_dummies(df_X, drop_first=True)
    X_encoded.head()

    # Reducción de dimensionalidad
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Preparación de los datos
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    y_train_clean = y_train.dropna(subset=['Severity'])
    y_test_clean = y_test.dropna(subset=['Severity'])

    X_train_df.reset_index(drop=True, inplace=True)
    X_test_df.reset_index(drop=True, inplace=True)
    y_train_clean.reset_index(drop=True, inplace=True)
    y_test_clean.reset_index(drop=True, inplace=True)

    X_train_clean = X_train_df.loc[y_train_clean.index]
    X_test_clean = X_test_df.loc[y_test_clean.index]

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_clean, y_train_clean['Severity'])
        
        # Make predictions and evaluate the model
        y_pred = model.predict(X_test_clean)

        # Ensure the target and predictions are in the same format
        y_test_clean_severity = y_test_clean['Severity'].values
        y_pred = y_pred.flatten()
        accuracy = accuracy_score(y_test_clean_severity, y_pred)
        
        # Log parameters and metrics to MLFlow
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("max_iter", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        model_info = mlflow.sklearn.log_model(model, "model")

        sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
        
        print(f"Model accuracy: {accuracy}")