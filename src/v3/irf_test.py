'''
Created on Mar 22, 2017

@author: hzhang0418
'''
import time
import numpy as np

from sklearn.model_selection import KFold

import v3.preprocessing as prep
import v3.incremental_rf as rf
import v3.fpfn_irf as fpfn

import utils.myconfig

def test():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    start = time.time()
    
    tool = fpfn.FPFN_IRF(list(range(len(labels))), features, labels, 5)
    ranked = tool.detect_and_rank()
        
    end = time.time()
    print("Time (secs): ", (end-start))
    print("First iteration: ", len(ranked))

    user_fb = {}
    for index in ranked[:20]:
        if labels[index]==0:
            user_fb[index] = 1
        else: 
            user_fb[index] = 0
    
    start = time.time()
    
    tool.use_feedback(user_fb)
    ranked = tool.detect_and_rank()
    
    end = time.time()
    print("Time (secs): ", (end-start))
    
    print("Second iteration: ", len(ranked))
    