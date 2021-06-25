"""
Startup script for Jupyter notebooks. It's a good way to load all the libraries, extensions and variables fast
"""
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path 
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import warnings
import random
import shutil
import pickle
import torch
import wandb
import json
import math
import glob
import cv2
import sys
import gc
import os

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output 
from IPython import get_ipython

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
ipython = get_ipython()
try: 
    ipython.magic('matplotlib inline')
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
except: 
    print('could not load ipython magic extensions')

# Add PATHS, pip_install
COMP_NAME = 'siim-covid19-detection'
KAGGLE_PATHS = {
    'working': Path('/kaggle/working'), 
    'input': Path('/kaggle/input'), 
    'tmp': Path('/kaggle/tmp'), 
    'comp': Path(f'/kaggle/input/{COMP_NAME}'), 
    'kaggle': Path('/kaggle/working/Kaggle'),
    'dataframes': Path('/kaggle/working/Kaggle/dataframes'), 
}

train = pd.read_csv(KAGGLE_PATHS['dataframes'] / 'fold_0' / 'full' / 'train.csv')
valid = pd.read_csv(KAGGLE_PATHS['dataframes'] / 'fold_0' / 'full' / 'valid.csv')