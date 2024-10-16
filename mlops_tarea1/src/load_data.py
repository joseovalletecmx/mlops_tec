
from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Text
import argparse
import yaml

def load_rawdata(config_path: Text) -> None:

    with open('/Users/joseovalle/Desktop/mlops_jovalle/mlops_tec/mlops_tarea1/params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)
    
    # fetch dataset 
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
    
    # create dataframes (features, targets, variables, raw_df)
    features = regensburg_pediatric_appendicitis.data.features 
    targets = regensburg_pediatric_appendicitis.data.targets 
    variables = regensburg_pediatric_appendicitis.variables
    raw_df = pd.concat([features, targets], axis=1)

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
    return None

if __name__ == "__main__":
    # Set up argument parsing
    args_parser = argparse.ArgumentParser(description='Load raw data and save to CSV files.')
    args_parser.add_argument('--config', dest ='config', required=True)
    args = args_parser.parse_args()    
    # Call the function with command line arguments
    load_rawdata(config_path= args.config)
