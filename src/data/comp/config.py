from src.data.utils import DATA_FOLDER_PATH 

# Competition specific config
DATASET_NAME = 'siim-covid19-detection'
LABELS = ['negative', 'typical']
LABEL_COLS = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']
SPLIT = 'group' # make folds by splitting on groups instead of stratifying
RENAME_MAP = {}  # rename some columns for standardization


# Mostly constants
HOLDOUT_PERCENTAGE = 1 # percentage of data that no model has seen  
NUM_FOLDS = 4 # number of folds
RANDOM_STATE = 42 # random state parameter in sklearn


# Paths for the dataset
RAW_DATASET_PATH = DATA_FOLDER_PATH / 'raw' / DATASET_NAME
INTERIM_DATASET_PATH = DATA_FOLDER_PATH / 'interim' / DATASET_NAME 
PROCESSED_DATASET_PATH = DATA_FOLDER_PATH / 'processed' / DATASET_NAME