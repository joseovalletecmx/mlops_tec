
from ucimlrepo import fetch_ucirepo
import pandas as pd
import argparse

def load_rawdata(features_path, targets_path, variables_path, raw_df_path):
    # fetch dataset 
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
    
    # create dataframes (features, targets, variables, raw_df)
    features = regensburg_pediatric_appendicitis.data.features 
    targets = regensburg_pediatric_appendicitis.data.targets 
    variables = regensburg_pediatric_appendicitis.variables
    raw_df = pd.concat([features, targets], axis=1)

    # export to csv
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    variables.to_csv(variables_path, index=False)
    raw_df.to_csv(raw_df_path, index=False)
    
    print("Data imported succesfully") 
    return None

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Load raw data and save to CSV files.')
    parser.add_argument('features_path', type=str, help='Path to save features CSV')
    parser.add_argument('targets_path', type=str, help='Path to save targets CSV')
    parser.add_argument('variables_path', type=str, help='Path to save variables CSV')
    parser.add_argument('raw_df_path', type=str, help='Path to save raw DataFrame CSV')
    
    args = parser.parse_args()
    
    # Call the function with command line arguments
    load_rawdata(args.features_path, args.targets_path, args.variables_path, args.raw_df_path)

  