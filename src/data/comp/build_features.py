"""
Build feature columns for the dataframes (train, test)
TRICK: To apply only one function, change the feature engineering pipeline function and change input folder to processed

PARAMETERS
----------
    input_folder: input folder for the interim dataframes / processed
    output_folder: output folder for the procesed dataframes
    feature_engineering_pipeline: all the functions in this are applied sequentially to the input

FUNCTIONS
---------
common_feature_engineering_pipeline(df)
    common feature engineering functions for test and train
    
TODO
----
- Save both as pickle and csv
- improve the pipeline
"""
from distutils.dir_util import copy_tree
from pathlib import Path
from time import time
import pandas as pd
import glob
import os 

# fastai dicom stuff
# from fastai.medical.imaging import get_dicom_files

from src.data.utils import feature_col
from src.data.comp.config import (
    # Competition specific config
    DATASET_NAME, LABELS, LABEL_COLS, SPLIT, RENAME_MAP, 
    # Mostly constants
    HOLDOUT_PERCENTAGE, NUM_FOLDS, RANDOM_STATE, 
    # Paths for the dataset
    RAW_DATASET_PATH, INTERIM_DATASET_PATH, PROCESSED_DATASET_PATH, 
)
import src.data.comp.preprocess

# Main functions for feature engineering
def add_file_path(df, raw_dataset_path=RAW_DATASET_PATH): 
    @feature_col
    def file_path(filepaths, img_id, study_id): 
        for filepath in filepaths: 
            if img_id in filepath and study_id in filepath: 
                return filepath
        return None
    glob_re = str(raw_dataset_path/ '**/*dcm')
    filepaths = glob.glob(glob_re, recursive=True)
    df = file_path(df, filepaths=filepaths)
    return df    

def add_dicom_metadata(df, df_type, raw_dataset_path): 
    """
    add metadata about images like dim0, dim1
    for dicom files, add all the metadata that is availible
    """
    # all_dicom_files = get_dicom_files(raw_dataset_path)
    return df

# FINAL FEATURE ENGINEERING PIPELINES
def feature_engineering_pipeline(df, df_type='train', raw_dataset_path=RAW_DATASET_PATH): 
    """
    Final feature engineering pipeline for dataframes

    Args:
        df: pipeline applied to this dataframe
        df_type: split of dataframe. Either train or test
        raw_dataset_path: read / generate filename from here
        interim_dataset_path: already splitted dataset
    """
    # Test specific stuff
    if df_type == 'test': 
        df = src.data.comp.preprocess.read_raw_test(input_folder=raw_dataset_path)
    else: # Train specific stuff
        pass
    df = add_file_path(df, raw_dataset_path)
    df = add_dicom_metadata(df, df_type, raw_dataset_path)
    return df
    


def main(input_folder=INTERIM_DATASET_PATH, output_folder=PROCESSED_DATASET_PATH, raw_dataset_path=RAW_DATASET_PATH, feature_engineering_pipeline=feature_engineering_pipeline): 
    start_time = time()
    # Copy everything from input_folder to output_folder
    if input_folder != output_folder: 
        copy_tree(str(input_folder), str(output_folder))
        print(f'reading dataframes from {input_folder} and writing to {output_folder}')
    else: 
        print('reading from and writing to the same folder')    
    
    # Read all the dataframe related files
    all_csv_files = glob.glob(str(output_folder/'**/*.csv'), recursive=True)
    all_pkl_files = glob.glob(str(output_folder/'**/*.pkl'), recursive=True)
    all_files = all_csv_files + all_pkl_files
    
    for df_file in all_files: 
        df = pd.read_csv(df_file) if df_file.endswith('.csv') else pd.read_pickle(df_file)
        df_type = 'test' if 'test' in df_file else 'train'
        if 'sample_sub' in df_file: continue
        df = feature_engineering_pipeline(df=df, df_type=df_type, raw_dataset_path=raw_dataset_path)
        df.to_csv(df_file) if df_file.endswith('.csv') else df.to_pickle(df_file)
    print(f'{time()-start_time} seconds to build the features') #23 secs
    
if __name__ == '__main__': 
    # Build from scratch
    main()
        
    # To skip expensive operations
    # MODIFY_FOLDER = Path('./dataframes')
    # add_function = lambda df, df_type, raw_dataset_path: df # add this function to all dataframes
    # main(input_folder=Path('./dataframes'), output_folder=Path('./dataframes'), feature_engineering_pipeline=lambda df: df)        
