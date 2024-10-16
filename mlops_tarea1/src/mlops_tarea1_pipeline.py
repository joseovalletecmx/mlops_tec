
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 
import joblib 
import pickle
import yaml


with open('/Users/joseovalle/Desktop/mlops_jovalle/mlops_tec/mlops_tarea1/params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

def load_rawdata():
    # fetch dataset 
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
    # create dataframes (features, targets, variables, raw_df)
    features = regensburg_pediatric_appendicitis.data.features 
    targets = regensburg_pediatric_appendicitis.data.targets 
    variables = regensburg_pediatric_appendicitis.variables
    raw_df = pd.concat([features,targets], axis = 1)
    # export paths
    features_path = config['data_load']['features_path']  
    targets_path = config['data_load']['targets_path'] 
    variables_path = config['data_load']['variables_path'] 
    raw_df_path = config['data_load']['raw_path'] 
    # export to csv
    features.to_csv(features_path, index = False)
    targets.to_csv(targets_path, index = False)
    variables.to_csv(variables_path, index = False)
    raw_df.to_csv(raw_df_path, index = False)
    print("Data imported succesfully") 

def get_features(features_df, feature_type):
    df_featuretype = features_df[(features_df['type'] == feature_type) & (features_df['role'] == 'Feature')]
    features = df_featuretype['name'].unique()
    features = list(features)    

    print(feature_type,"features list created succesfully")
    return features

def continuous_imputation(dataframe,featuretype):
    for feature in featuretype:
        # calculate mean value for the featue within the featureype
        mean_value = dataframe[feature].mean()    
        # Fill NaN values with the median of the column
        dataframe.loc[:, feature] = dataframe[feature].fillna(mean_value)
        # print("Mean imputation for {} executed succesfully".format(feature))
    return None

def integer_imputation(dataframe,featuretype): 
    for feature in featuretype:
        # calculate mean value for the featue within the featureype
        mode_value = dataframe[feature].mode()[0]    
        # Fill NaN values with the median of the column
        dataframe.loc[:, feature] = dataframe[feature].fillna(mode_value)
        #print("Mode imputation for {} executed succesfully".format(feature))       
    return None

def categorical_imputation(dataframe,featuretype):
    for feature in featuretype:
        dataframe.loc[:, feature] = dataframe[feature].fillna('Not present') 
    return None

def binary_imputation(dataframe,featuretype):
    for feature in featuretype:
        dataframe.loc[:, feature] = dataframe[feature].fillna('Unkown') 
    return None

def impute_y(dataframe):
    # select target feature
    dataframe = dataframe['Diagnosis']
    dataframe = pd.DataFrame(dataframe)
    
    # covert strinf descrriptions to integer featutres
    dataframe.loc[dataframe['Diagnosis']=='appendicitis'] = 1
    dataframe.loc[dataframe['Diagnosis']=='no appendicitis'] = 0

    # impute target feature
    mode_value = dataframe['Diagnosis'].mode()[0]
    dataframe =  dataframe.fillna(mode_value)   
    return dataframe

def run_eda(dataframe, features):
    print('')
    print(dataframe[features[:5]].describe())
    plt.figure(figsize=(10, 6))
    sns.pairplot(dataframe[features[:5]], height= 2)
    return None

def normalize(dataframe, features):
    scaler = MinMaxScaler()
    continuous_df_features = pd.DataFrame(scaler.fit_transform(dataframe[features]), columns = dataframe[features].columns)
    return continuous_df_features

def encode(dataframe, categorical_features, binary_features):
    categorical_df_features = pd.concat([dataframe[categorical_features],dataframe[binary_features]], axis = 1)
    categorical_df_features = pd.get_dummies(categorical_df_features, drop_first = True)
    return categorical_df_features

def run_pca(dataframe):
    pca = PCA(n_components = 3)
    pca_result = pca.fit_transform(dataframe)

    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    # Explained variance ratio (how much variance each principal component explains)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Explained variance ratio for each component
    explained_variance = pca.explained_variance_ratio_

    # Create the loadings matrix
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Convert to a DataFrame for better readability
    loadings_df = pd.DataFrame(loadings, index=dataframe.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
    loadings_df

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, center=0)
    plt.title('PCA Loadings Matrix')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.show()

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

def save_preproccesed_data(dataframe):
    path = config['data_preprocess']['preprocessed_path']
    dataframe.to_csv(path, index= False)
    print("Data stored at:", path)
    return None

def split_train_test(dict):
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

    return None


def train_random_forest(X_train, X_test, y_train,y_test):

# Initialize the Random Forest Classifier

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, np.array(y_train).flatten())
    y_pred = rf_classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Print accuracy and confusion matrix
    print("Accuracy of the model:", round(accuracy,4))
    print("")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot = True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def get_param_grid(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5,8],
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, np.array(y_train).flatten())

    # Best parameters
    return  grid_search.best_params_

def train_parametrized_model(X_train, y_train, X_test, y_test, parameters):

    rf_classifier_hp = RandomForestClassifier(n_estimators = parameters['n_estimators']
                                            ,random_state=42
                                            ,max_depth = parameters['max_depth'] 
                                            ,min_samples_split = parameters['min_samples_split']                                           
                                            )
    rf_classifier_hp.fit(X_train, np.array(y_train).flatten())
    y_pred_hp = rf_classifier_hp.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred_hp)
    confusion = confusion_matrix(y_test, y_pred_hp)

    # Print accuracy and confusion matrix
    print("Accuracy of the model:", round(accuracy,4))
    print("")

    print("Classification Report:")
    print(classification_report(y_test, y_pred_hp))

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot = True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return rf_classifier_hp



def export_model(model):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    path = config['data_model']['model_path']
    joblib.dump(model,path)

    print("Data stored at:", path)
    return None


def write_evaluation_report(report, confusion_matrix):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    path = config['data_model']['model_report_path']
    with open(path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix))

def evaluate_model(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1)
    cm = confusion_matrix(y_test, predictions)
    write_evaluation_report(report, cm)

load_rawdata()

X = pd.read_csv(config['data_load']['features_path'] )
y = pd.read_csv(config['data_load']['targets_path'] )
features = pd.read_csv(config['data_load']['variables_path'])

continuous_features = get_features(features,'Continuous')
categorical_features = get_features(features,'Categorical')
binary_features = get_features(features,'Binary')
integer_features = get_features(features,'Integer')

print("Running imputation for continuous features")
continuous_imputation(X,continuous_features)

print("Running imputation for categorical features")
categorical_imputation(X,categorical_features)

print("Running imputation for binary features")
binary_imputation(X,binary_features)

print("Running imputation for integer features")
integer_imputation(X,integer_features)

# Value imputation for y feature
print("Running imputation for target feature")
y = impute_y(y)

print('Normalizing continuous features')
continuous_df_features = normalize(X, continuous_features)
continuous_df_features.head()

print('encoding categorical features')
categorical_df_features = encode(X, categorical_features,binary_features)
categorical_df_features.head()

print('running EDA')
run_eda(X,continuous_features)

print('running PCA')
continuous_df_features = run_pca(continuous_df_features)

preprocessed_data = integrate_features_data(X, y, continuous_df_features,categorical_df_features,integer_features)
save_preproccesed_data(preprocessed_data['preprocessed_data'])


split_train_test(preprocessed_data)

X_train = pd.read_csv(config['data_split']['x_train_path'])
X_test  = pd.read_csv(config['data_split']['x_test_path'])
y_train = pd.read_csv(config['data_split']['y_train_path'])
y_test  = pd.read_csv(config['data_split']['y_test_path'])


train_random_forest(X_train, X_test, y_train,y_test)

parameters = get_param_grid(X_train, y_train)
print(parameters)

improved_model = train_parametrized_model(X_train, y_train, X_test, y_test, parameters)

export_model(improved_model)


model_path = config['data_model']['model_path']
X_test_path =  config['data_split']['x_test_path']
y_test_path =  config['data_split']['y_test_path']
evaluate_model(model_path, X_test_path, y_test_path)

