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

class DataPreprocessor:
    def __init__(self,config_path: Text) -> None:
        self.config_path = config_path
        self.config = self.load_config()
        self._raw_df = None  # Private attribute to store raw DataFrame
        self._features = None  # Private attribute for features
        self._targets = None  # Private attribute for targets
        self._variables = None  # Private attribute for variables
        self._continuous_features = None # Private attribute for continuous features
        self._categorical_features = None # Private attribute for categorical features 
        self._integer_features = None # Private attribute for integer features 
        self._binary_features = None # Private attribute for binary features
        self._target = None # Private attribute for binary features 
        self._continuous_df_features = None # Private attribute for normalized continuous features
        self._categorical_df_features = None # Private attribute for encoded categorical features
        self._selected_features = None   # Private attribute for engineered features
        self._X_preprocessed = None # Private attribute for preprocessed features 
        self._y_preprocessed = None # Private attribute for preprocessed target 
        self._preprocessed_data = None # Private attribute for preprocessed data
        self._X_train = None # Private attribute for training dataset
        self._X_test = None # # Private attribute for test dataset
        self._y_train = None # # Private attribute for training dataset
        self._y_test = None # Private attribute for test dataset
    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    @property
    def raw_df(self):
        return self._raw_df

    @property
    def variables(self):
        return self._variables
    
    @property
    def continuous_features(self):
        return self._continuous_features
    
    @property
    def categorical_features(self):
        return self._categorical_features
    
    @property
    def integer_features(self):
        return self._integer_features
    
    @property
    def binary_features(self):
        return self._binary_features
     
    @property
    def continuous_df_features(self):
        return self._continuous_df_features 

    @property
    def categorical_df_features(self):
        return self._categorical_df_features 
    
    @property
    def X_preprocessed(self):
        return self._X_preprocessed

    @property
    def y_preprocessed(self):
        return self._y_preprocessed

    @property
    def preprocessed_data(self):
        return self._X_preprocessed_data
    
    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def selected_features(self):
        return self._selected_features

    def load_config(self):
        with open(self.config_path) as conf_file:
            config = yaml.safe_load(conf_file)
        return config

    def load_data(self):
        self._features = pd.read_csv(self.config['data_load']['features_path']) 
        self._targets = pd.read_csv(self.config['data_load']['targets_path']) 
        self._variables = pd.read_csv(self.config['data_load']['variables_path']) 
        self._raw_df = pd.read_csv(self.config['data_load']['raw_path'])

    def get_continuous_features(self):
        df_featuretype = self.variables[(self.variables['type'] == 'Continuous') & (self.variables['role'] == 'Feature')]
        features = df_featuretype['name'].unique()
        self._continuous_features = list(features)     

    def get_categorical_features(self):
        df_featuretype = self.variables[(self.variables['type'] == 'Categorical') & (self.variables['role'] == 'Feature')]
        features = df_featuretype['name'].unique()
        self._categorical_features = list(features)

    def get_integer_features(self):
        df_featuretype = self.variables[(self.variables['type'] == 'Integer') & (self.variables['role'] == 'Feature')]
        features = df_featuretype['name'].unique()
        self._integer_features = list(features)

    def get_binary_features(self):
        df_featuretype = self.variables[(self.variables['type'] == 'Binary') & (self.variables['role'] == 'Feature')]
        features = df_featuretype['name'].unique()
        self._binary_features = list(features)

    def impute_continuous_features(self):
        for feature in self._continuous_features:
            mean_value = self._features[feature].mean()    
            self._features.loc[:, feature] = self._features[feature].fillna(mean_value)

    def impute_integer_features(self): 
        for feature in self._integer_features:
            mode_value = self._features[feature].mode()[0]    
            self._features.loc[:, feature] = self._features[feature].fillna(mode_value)

    def impute_categorical_features(self):
        for feature in self._categorical_features:
            self._features.loc[:, feature] = self._features[feature].fillna('Not present')

    def impute_binary_features(self):
        for feature in self._binary_features:
            self._features.loc[:, feature] = self._features[feature].fillna('Unkown')
            
    def impute_target(self):
        self._target = self._targets['Diagnosis']
        self._target = pd.DataFrame(self._target)
        self._target.loc[self._target['Diagnosis']=='appendicitis'] = 1
        self._target.loc[self._target['Diagnosis']=='no appendicitis'] = 0
        mode_value = self._target['Diagnosis'].mode()[0]
        self._target =  self._target.fillna(mode_value)

    def normalize(self):
        scaler = MinMaxScaler()
        self._continuous_df_features = pd.DataFrame(scaler.fit_transform(self._features[self._continuous_features]), columns = self._features[self._continuous_features].columns)

    def encode(self):
        self._categorical_df_features = pd.concat([self._features[self._categorical_features],self._features[self._binary_features]], axis = 1)
        self._categorical_df_features= pd.get_dummies(self._categorical_df_features, drop_first = True)

    def engineer_features(self):
        self._selected_features = self.config['feature_selection']['selected_features']
        self._continuous_df_features = self._continuous_df_features[self.selected_features]

    def integrate_data(self):
        self._X_preprocessed = pd.concat([self._continuous_df_features,self._categorical_df_features], axis = 1)
        self._X_preprocessed = pd.concat([self._X_preprocessed,self._features[self._integer_features]], axis = 1)
        self._y_preprocessed = self._target
        self._preprocessed_data = pd.concat([self._X_preprocessed,self._target], axis = 1)
        path = self.config['data_preprocess']['preprocessed_path']
        self._preprocessed_data.to_csv(path)

    def train_test_split(self):        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split( self._X_preprocessed
                                                        ,self._y_preprocessed.astype(int)
                                                        ,train_size = .8
                                                        ,test_size=0.2
                                                        ,random_state=42
                                                        ,stratify= self._y_preprocessed.astype(int)
                                                        )
    def save_data_split(self):  
        path_x_train = self.config['data_split']['x_train_path']
        path_x_test = self.config['data_split']['x_test_path']
        path_y_train = self.config['data_split']['y_train_path']
        path_y_test = self.config['data_split']['y_test_path']           
        
        self._X_train.to_csv(path_x_train, index= False)
        self._X_test.to_csv(path_x_test, index= False)
        self._y_train.to_csv(path_y_train, index= False)
        self._y_test.to_csv(path_y_test, index= False)
        
if __name__ == "__main__":
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Preprocess data')
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    # Create DataLoader object and run the process
    model = DataPreprocessor(config_path=args.config)
    model.load_data()
    model.get_continuous_features()
    model.get_categorical_features()
    model.get_integer_features()
    model.get_binary_features()
    model.impute_continuous_features()
    model.impute_integer_features()
    model.impute_categorical_features()
    model.impute_binary_features()
    model.impute_target()
    model.normalize()
    model.encode()
    model.engineer_features()
    model.integrate_data()
    model.train_test_split()
    model.save_data_split()
