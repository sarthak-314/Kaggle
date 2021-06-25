"""
⚒ Improvements & Ideas ⚒
-------------------------
- read omega conf documentation
- clean function descriptions and also the code
- add function to create meta_df in feature engineering

SAMPLE - SPLIT - STANDARDIZE
Builds interium dataframes for the dataset


PARAMETERS
----------
    input_folder: input folder for the raw dataframes
    output_folder: output folder for the final dataframes

MAIN FUNCTIONS 
---------------
read_raw_train(input_folder)
    read train dataframe from input folder

read_raw_test(input_folder)
    read test dataframe from input folder

read_raw_sample_sub(input_folder)
    read sample_sub from input folder     

add_intermediate_columns(train)
    add non feature columns like group and stratify to train 
    
sample_train(train, debug_percentage): 
    sample train according to sampling strategy

split_func(df): 
    splits dataframe into split_dfs and returns them 

main(input_folder, output_folder, build_options): 
    main functions to make dataset
"""

from time import time
import pandas as pd 
import numpy as np  
import glob
import os 

import src
import src.data.utils
import src.comp.utils

from src.data.comp.config import (
    # Competition specific config
    DATASET_NAME, LABELS, LABEL_COLS, SPLIT, RENAME_MAP, 
    # Mostly constants
    HOLDOUT_PERCENTAGE, NUM_FOLDS, RANDOM_STATE, 
    # Paths for the dataset
    RAW_DATASET_PATH, INTERIM_DATASET_PATH, PROCESSED_DATASET_PATH, 
)

# READ RAW DATAFRAMES FROM THE DATASET

def merge_input_dataframes(train_img, train_study):
    image_level_rename_map = { 'StudyInstanceUID': 'study_id', 'id': 'img_id' }
    train_img.id = train_img.id.str.replace('_image', '')
    train_img = train_img.rename(columns=image_level_rename_map)
    study_level_rename_map = {'id':'study_id'}
    train_study.id = train_study.id.str.replace('_study', '')
    train_study = train_study.rename(columns=study_level_rename_map)
    train = train_img.merge(train_study, on='study_id')
    return train

def read_raw_train(input_folder=RAW_DATASET_PATH):
    train_study = pd.read_csv(input_folder / 'train_study_level.csv')
    train_img = pd.read_csv(input_folder / 'train_image_level.csv')
    train = merge_input_dataframes(train_img, train_study)
    return train

def get_path_components(path): 
    normalized_path = os.path.normpath(path)
    path_components = normalized_path.split(os.sep)
    return path_components

def read_raw_test(input_folder=RAW_DATASET_PATH):
    filepaths = glob.glob(str(input_folder / 'test/**/*dcm'), recursive=True)
    test = pd.DataFrame({ 'img_path': filepaths })
    test['img_id'] = test.img_path.map(lambda x: get_path_components(x)[-1].replace('.dcm', ''))
    test['study_id'] = test.img_path.map(lambda x: get_path_components(x)[-3].replace('.dcm', ''))
    return test 

def read_raw_sample_sub(input_folder=RAW_DATASET_PATH):
    sample_sub = pd.read_csv(input_folder / 'sample_submission.csv')
    return sample_sub

def standardize_train(train): 
    # One hot encode and add labels
    train['one_hot'] = train[LABEL_COLS].apply(lambda row: row.values, axis='columns')
    train['label'] = train.one_hot.apply(lambda array: np.argmax(array))

    # Add stratify column and group column
    train['stratify'] = train['one_hot'].apply(str)
    train['group'] = train['study_id'].apply(str)
    
    return train

def sample_train(train, num_values_to_sample, random_state=RANDOM_STATE):
    return train.sample(num_values_to_sample, random_state=random_state)

def main(input_folder=RAW_DATASET_PATH, output_folder=INTERIM_DATASET_PATH): 
    """
    Main function to build the initial dataset and folds

    Args:
        input_folder (Path): files read from here
        output_folder (Path): dataframes saved here
    """
    start_time = time()
    
    # Read input dataframes
    train, test, sample_sub = read_raw_train(input_folder), read_raw_test(input_folder), read_raw_sample_sub(input_folder)
    
    # Standardize train
    train = standardize_train(train)
    
    # Split train into holdout and save all
    train, holdout = src.data.utils.get_holdout(train, HOLDOUT_PERCENTAGE)
    for df, df_name in zip([train, test, holdout, sample_sub], ['train_raw', 'test', 'holdout', 'sample_sub']): 
        df_path = output_folder / df_name
        os.makedirs(output_folder, exist_ok=True)
        src.data.utils.save_df(df, df_path)    
    
    get_fold_dfs_func = lambda df: src.data.utils.get_fold_dfs(df=df, split_by='group', num_folds=NUM_FOLDS)
    train_fold_dfs = get_fold_dfs_func(train)
    for fold in range(NUM_FOLDS): 
        for debug_percentage in ['one', 'five', 'twenty', 'full']: 
            train = sample_train(train, src.data.utils.get_num_values_to_sample(train, debug_percentage))
            train, valid = src.data.utils.get_fold_from_fold_dfs(fold_dfs=train_fold_dfs, fold=fold)
            valid_75, valid_25 = src.data.utils.get_fold_from_fold_dfs(get_fold_dfs_func(valid), fold)
            
            # Save in folder
            inner_folder = output_folder / f'fold_{fold}' / debug_percentage 
            os.makedirs(inner_folder, exist_ok=True)
            for df, df_name in zip([train, valid, valid_75, valid_25], ['train', 'valid', 'valid_75', 'valid_25']): 
                df_path = inner_folder / df_name
                src.data.utils.save_df(df=df, path=df_path)
        print(f'dataframes created for fold {fold}')           
    
    print(f'making {NUM_FOLDS} folds took {time()-start_time} seconds')
    
if __name__ == '__main__':
    main()
    
    