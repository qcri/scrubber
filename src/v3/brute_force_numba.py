'''
Created on Mar 3, 2017

@author: hzhang0418
'''

import numpy as np
import time

from numba import jit

import utils.myconfig
import v3.preprocessing as prep

@jit
def compute_violations(A, B, n):
    """
    Compute the violations
    """
    index2incons = {}
    
    for i in range(n):
        # Pick a vector from matches
        a = A[i]
        # Find difference with all the vectors in non matches
        c = a - B
        # Convert the diff. to 0 or 1
        c[c <= 0] = 1
        c[c > 0] = 0
        
        # Sum the violations for each non-match feature vector
        x = np.sum(c, axis=1)
        
        # Report if the number of violations is greater than 15
        #print x

def example():
    A = np.random.randint(100, size=(500000, 17)) # Match feature vectors 500k x 17
    B = np.random.randint(100, size=(500000, 17))
    
    start = time.time()
    compute_violations(A, B, 100)
    end = time.time()
    print("Time (secs): ", (end-start))
    
    
def test():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    (nrows, ncols) = features.shape
    
    match_indices = []
    nonmatch_indices = []
    
    index = 0
    for _ in range(len(labels)):
        if labels[index] == 1:
            match_indices.append(index)
        else:
            nonmatch_indices.append(index)
        index += 1
    
    A = features[ match_indices ]
    B = features[ nonmatch_indices ]
    
    start = time.time()
    index2incons = compute_violations(A, B, len(A))
    end = time.time()
    print("Time (secs): ", (end-start))
    
    