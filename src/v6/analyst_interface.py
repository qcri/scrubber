import os
from utils import myconfig
import v6.data_io, v6.feature_selection, v6.label_debugger
import pandas as pd
import numpy as np
import time

## get data from LabelDebugger config files
def read_config(config_file):
    params = myconfig.read_config(config_file)
    # other config params
    params['fs_alg'] = 'xgboost'
    params['max_list_len'] = 20
    params['detectors'] = 'fpfn'
    params['num_cores'] = 4
    params['num_folds'] = 5
    params['min_con_dim'] = 1
    params['counting_only'] = False
    params['top_k'] = 10
    return params

def read_table(basedir, tab_name):
    return pd.read_csv(os.path.join(basedir, tab_name))
