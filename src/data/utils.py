from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split
from pathlib import Path
from time import time
import pandas as pd
import math
import os 

# Default values
RANDOM_STATE = 42
N_SPLITS = 4 

# Data Folder Path (environment dependent)
DATA_FOLDER_PATH = Path('C:\\Users\\sarth\\Desktop\\kaggle-v2\\data')
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ: 
    DATA_FOLDER_PATH = Path('/kaggle/input/data')


# Common Utility Functions
def save_df(df, path): 
    df.to_csv(str(path) + '.csv', index=False)
    df.to_pickle(str(path) + '.pkl')

def save_dataframes(df_to_path_dict):
    for df, path in df_to_path_dict.items(): 
        save_df(df, path) 


def read_raw_input_dataframe(csv_file, raw_dataset_folder_path):
    """
    read the raw dataframe from csv file in the dataset
    looks inside raw_dataset_folder_path / dataset_name to find the csv_file
    """ 
    csv_file_path = raw_dataset_folder_path / csv_file
    if not csv_file_path.exists(): 
        print(f'csv file not found at {csv_file_path}')
    df = pd.read_csv(csv_file_path)
    return df

def rename(df, rename_mapping):
    df = df.sample(frac=1, random_state=RANDOM_STATE)
    rename_mapping_ = {k:v for k, v in rename_mapping.items() if k in df.columns}
    df_with_renamed_columns = df.rename(columns=rename_mapping_)
    return df_with_renamed_columns 

def add_fold_column(df, fold_class=GroupKFold, num_folds=N_SPLITS): 
    fold_fn = fold_class(num_folds)
    if fold_class == StratifiedKFold:
        split_with = df.stratify
    elif fold_class == GroupKFold: 
        split_with = df.group
    df['fold'] = -1
    for fold_, (_, val_idx) in enumerate(fold_fn.split(df, df, split_with)): 
        df.fold.iloc[val_idx] = fold_
    return df

def get_fold_dfs(df, split_by, num_folds): 
    fold_class = {'group': GroupKFold, 'stratify': StratifiedKFold}[split_by]
    fold_fn = fold_class(num_folds)
    fold_dfs = []
    for fold_, (_, val_idx) in enumerate(fold_fn.split(df, df, df[split_by])): 
        fold_df = df.iloc[val_idx]
        fold_dfs.append(fold_df)
    return fold_dfs

def get_fold_from_fold_dfs(fold_dfs, fold): 
    train = pd.concat(fold_dfs[:fold]+fold_dfs[fold+1:])
    valid = fold_dfs[fold]
    return train, valid

def get_num_values_to_sample(train, debug_percentage): 
    if debug_percentage == 'full': 
        return len(train)
    debug_percentage_to_num = { 'full': 100, 'twenty': 20, 'five': 5, 'one': 1 }
    debug_percentage = debug_percentage_to_num[debug_percentage]
    num_values_to_take = math.ceil(len(train) * debug_percentage / 100)
    return num_values_to_take

def get_holdout(train, holdout_percentage): 
    train_99, holdout = train_test_split(train, test_size=holdout_percentage/100, random_state=RANDOM_STATE, shuffle=True)
    return train_99, holdout

def undo_rename(df, rename_mapping): 
    reverse_mapping = {v:k for k, v in rename_mapping.items() if k in df.columns}
    return df.rename(columns=reverse_mapping)

def sample_top_classes(train, num_values_to_take): 
    for i, cumsum in enumerate(pd.Index(train.stratify.value_counts().values.cumsum())): 
        if cumsum >= num_values_to_take: 
            num_classes_to_take = i+1
            break
    num_classes_to_take = max(num_classes_to_take, 3)
    taking_classes = train.stratify.value_counts().nlargest(num_classes_to_take).index.tolist()
    print(f'Taking {num_classes_to_take} unique stratify vals out of total {train.stratify.nunique()} stratify values')
    train = train[train.stratify.isin(taking_classes)]
    return train

def build_fold_column(df, fold_class=GroupKFold, num_folds=N_SPLITS): 
    fold_fn = fold_class(num_folds)
    if fold_class == StratifiedKFold: 
        split_with = df.stratify
    elif fold_class == GroupKFold: 
        split_with = df.group
    df['fold'] = -1
    for fold_, (tr_idx, val_idx) in enumerate(fold_fn.split(df, df, split_with)): 
        df.fold.iloc[val_idx] = fold_
    return df['fold'].values

def feature_col(func):
    def wrapper(df):
        def func_wrapper(row): 
            kwargs = {}
            for col in func.__code__.co_varnames: 
                kwargs[col] = row[col]
            return func(**kwargs)
        df[func.__name__] = df.apply(func_wrapper, axis='columns')
        return df
    return wrapper 


def sample_top_groups(train, num_values_to_take): 
    # Take the most common groups with all classes
    for i, cumsum in enumerate(pd.Index(train.group.value_counts().values.cumsum())): 
        if cumsum >= num_values_to_take: 
            num_groups_to_take = i+1
            break
    num_groups_to_take = max(num_groups_to_take, 3)
    taking_groups = train.group.value_counts().nlargest(num_groups_to_take).index.tolist()
    print(f'Taking most common {num_groups_to_take} unique groups vals out of total {train.group.nunique()} groups')
    train = train[train.group.isin(taking_groups)]
    print(f'Took {len(train)} values from train')
    return train


def sample_train(train, debug_percentage): 
    num_values_to_take = get_num_values_to_sample(train, debug_percentage)
    train = sample_top_groups(train, num_values_to_take)
    return train

def split_train(train, fold):
    train['fold'] = build_fold_column(train, fold_class=GroupKFold, num_folds=N_SPLITS)
    train, valid = train[train.fold != fold], train[train.fold == fold]
    return train, valid

def split_valid(valid): 
    valid['fold'] = build_fold_column(valid, fold_class=GroupKFold, num_folds=N_SPLITS)
    valid_75, valid_25 = valid[valid.fold != 0], valid[valid.fold == 0]
    return valid_75, valid_25


            
            
# II. Utility functions for build_features.py             

def feature_col(func): 
    def wrapper(df, **kwargs):
        start_time = time() 
        def func_wrapper(row): 
            func_input_kwargs = kwargs
            for input_var in func.__code__.co_varnames: 
                is_input_var_column = input_var in row
                if is_input_var_column: 
                    col = input_var
                    func_input_kwargs[col] = row[col]
            return func(**func_input_kwargs)
        column_name = func.__name__
        df[column_name] = df.apply(func_wrapper, axis='columns')
        # print(f'{func.__name__} took {time()-start_time} seconds to execute for df of length {len(df)}')
        return df
    return wrapper    
            
