import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
from typing import Text
import argparse


with open('/Users/joseovalle/Desktop/mlops_jovalle/mlops_tec/mlops_tarea1/params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
    
def get_features(features_df, feature_type):
    df_featuretype = features_df[(features_df['type'] == feature_type) & (features_df['role'] == 'Feature')]
    features = df_featuretype['name'].unique()
    features = list(features)    
    return features

def continuous_imputation(dataframe,featuretype) -> None:
    for feature in featuretype:
        mean_value = dataframe[feature].mean()    
        dataframe.loc[:, feature] = dataframe[feature].fillna(mean_value)

def integer_imputation(dataframe,featuretype) -> None: 
    for feature in featuretype:
        mode_value = dataframe[feature].mode()[0]    
        dataframe.loc[:, feature] = dataframe[feature].fillna(mode_value)      

def categorical_imputation(dataframe,featuretype) -> None:
    for feature in featuretype:
        dataframe.loc[:, feature] = dataframe[feature].fillna('Not present') 

def binary_imputation(dataframe,featuretype) -> None:
    for feature in featuretype:
        dataframe.loc[:, feature] = dataframe[feature].fillna('Unkown')

def impute_y(dataframe):
    dataframe = dataframe['Diagnosis']
    dataframe = pd.DataFrame(dataframe)
    dataframe.loc[dataframe['Diagnosis']=='appendicitis'] = 1
    dataframe.loc[dataframe['Diagnosis']=='no appendicitis'] = 0
    mode_value = dataframe['Diagnosis'].mode()[0]
    dataframe =  dataframe.fillna(mode_value)   
    return dataframe

def normalize(dataframe, features):
    scaler = MinMaxScaler()
    continuous_df_features = pd.DataFrame(scaler.fit_transform(dataframe[features]), columns = dataframe[features].columns)
    return continuous_df_features

def encode(dataframe, categorical_features, binary_features):
    categorical_df_features = pd.concat([dataframe[categorical_features],dataframe[binary_features]], axis = 1)
    categorical_df_features = pd.get_dummies(categorical_df_features, drop_first = True)
    return categorical_df_features

def select_features(dataframe):
    selected_features = ['Age','BMI','Height','WBC_Count']
    print(selected_features, ' were selected for dimensionality reduction')
    dataframe = dataframe[selected_features]
    return dataframe

def integrate_features_data(X_dataframe,y_dataframe,continuous_df_features,categorical_df_features,integer_features):
    # Integración de variables continuas y categóricas
    X_preprocessed = pd.concat([continuous_df_features,categorical_df_features], axis = 1)
    # Integración de variables integer
    X_preprocessed = pd.concat([X_preprocessed,X_dataframe[integer_features]], axis = 1)
    # Dataframe resultante listo para entrengar el modelo
    data_preprocessed_df = pd.concat([X_preprocessed,y_dataframe], axis = 1)

    return {
         'X_preprocessed': X_preprocessed
        ,'y_preprocessed': y_dataframe
        ,'preprocessed_data': data_preprocessed_df
    } 

def save_preproccesed_data(dataframe) -> None:
    path = config['data_preprocess']['preprocessed_path']
    dataframe.to_csv(path, index= False)
    print("Data stored at:", path)

def split_train_test(dict) -> None:
    # split data
    X_train, X_test, y_train, y_test = train_test_split( dict['X_preprocessed']
                                                        ,dict['y_preprocessed'].astype(int)
                                                        ,test_size= config['data_split']['test_size']
                                                        ,random_state= config['data_split']['random_state']
                                                        ,stratify= dict['y_preprocessed'].astype(int)
                                                        )
    
    # save datasets 
    path_x_train = config['data_split']['x_train_path']
    path_x_test = config['data_split']['x_test_path']
    path_y_train = config['data_split']['y_train_path']
    path_y_test = config['data_split']['y_test_path']           
    
    X_train.to_csv(path_x_train, index= False)
    X_test.to_csv(path_x_test, index= False)
    y_train.to_csv(path_y_train, index= False)
    y_test.to_csv(path_y_test, index= False)           

    print("Data stored at:", path_x_train)  
    print("Data stored at:", path_x_test)  
    print("Data stored at:", path_y_test)  
    print("Data stored at:", path_y_train)    



if __name__ == "__main__":
    # Set up argument parsing
    
    # args_parser = argparse.ArgumentParser(description='Load raw data and save to CSV files.')
    # args_parser.add_argument('--config', dest ='config', required=True)
    # args = args_parser.parse_args()   
    
    # Execute preprocessing pipeline
    # Call the function with command line arguments
    # load_rawdata(config_path= args.config)

    X = pd.read_csv(config['data_load']['features_path'] )
    y = pd.read_csv(config['data_load']['targets_path'] )
    features = pd.read_csv(config['data_load']['variables_path'])
    continuous_features = get_features(features,'Continuous')
    categorical_features = get_features(features,'Categorical')
    binary_features = get_features(features,'Binary')
    integer_features = get_features(features,'Integer')

    continuous_imputation(X,continuous_features)
    categorical_imputation(X,categorical_features)
    binary_imputation(X,binary_features)
    integer_imputation(X,integer_features)
    y = impute_y(y)

    continuous_df_features = normalize(X, continuous_features)
    categorical_df_features = encode(X, categorical_features,binary_features)
    select_features(continuous_df_features)
    preprocessed_data = integrate_features_data(X, y, continuous_df_features,categorical_df_features,integer_features)
    save_preproccesed_data(preprocessed_data['preprocessed_data'])
    split_train_test(preprocessed_data)

