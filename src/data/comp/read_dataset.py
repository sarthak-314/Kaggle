"""
The main point of contact for other modules and function 
As long as everything in this works perfectly, the internals do not matter.
Focus on getting to this point as quickly as possible and iterate fast

API
---
read_dataframes
    read all the processed dataframes
    
"""
import pandas as pd
import os 

from src.data.comp.config import (
    # Competition specific config
    DATASET_NAME, LABELS, LABEL_COLS, SPLIT, RENAME_MAP, 
    # Mostly constants
    HOLDOUT_PERCENTAGE, NUM_FOLDS, RANDOM_STATE, 
    # Paths for the dataset
    RAW_DATASET_PATH, INTERIM_DATASET_PATH, PROCESSED_DATASET_PATH, 
)
import src.data.comp.preprocess_raw_dataset
import src.data.comp.build_features

def main(input_folder=PROCESSED_DATASET_PATH, fold=0, debug_percentage='full'): 
    """
    Main function to read dataframes from processed dataframes
    In competitions, to iterate fast, upload the processed dataframes and use this function to read the dataframes 

    Args:
        input_folder: Base path for processed dataframes. In kaggle this will be /kaggle/input/dataset-name
        fold, debug_percentage
    """
    res = {}
    # read the files from the outer folder
    res['train_raw'] = pd.read_pickle(input_folder / 'train_raw.pkl')
    res['test'] = pd.read_pickle(input_folder / 'test.pkl')
    res['holdout'] = pd.read_pickle(input_folder / 'holdout.pkl')
    
    # read the files from the inner folder 
    inner_folder = input_folder / f'fold_{fold}' / debug_percentage
    os.makedirs(inner_folder, exist_ok=True)
    res['train'] = pd.read_pickle(inner_folder / 'train.pkl')
    res['valid'] = pd.read_pickle(inner_folder / 'valid.pkl')
    res['valid_75'] = pd.read_pickle(inner_folder / 'valid_75.pkl')
    res['valid_25'] = pd.read_pickle(inner_folder / 'valid_25.pkl')
    
    # make some testing files
    res['tr'] = res['train'].head(10)
    res['te'] = res['test'].head(5)
    res['val'] = res['valid'].head(5)
    
    return res

def build_test(raw_dataset_path=RAW_DATASET_PATH): 
    # Read raw test and then add features to it
    test = src.data.comp.preprocess_raw_dataset.read_raw_test(raw_dataset_path)
    test = src.data.comp.build_features.feature_engineering_pipeline(df=test, df_type='test')
    return test

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


if __name__ == '__main__':
    print(main().keys())
    print('working!')