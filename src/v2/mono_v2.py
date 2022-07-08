'''
Created on Oct 24, 2016

@author: hzhang0418
'''
import numpy as np

import pandas as pd
from scipy.cluster._hierarchy import inconsistent
from sympy.physics.secondquant import ViolationOfPauliPrinciple

debug = False

def prepare(table_H, table_features):
    
    (nrow, ncol) = table_H.shape
    print(nrow, ncol)
    feature_names = table_features["feature_name"].tolist()
    if debug: print(feature_names)

    feature_vector = np.empty(shape = (nrow, len(feature_names)), dtype = np.float32) # each row is for one pair
    match_pairs = [] # list of match pairs with their row index
    nonmatch_pairs = [] # list of nonmatch pairs with their row index
    labels = {} # map pairs to their labels
    
    row_index = 0
    for _, row in table_H.iterrows():
        id1= row['ltable.id'];
        id2 = row['rtable.id'];
        label = row['label']
        pair = (id1, id2)
        
        if label == 1:
            match_pairs.append( (pair, row_index) )
        if label == 0:
            nonmatch_pairs.append( (pair, row_index) )
            
        for i, name in enumerate(feature_names):
            val = row[name]
            val = float("{0:.2f}".format(val)) # only 2 decimal points?
            feature_vector[row_index, i] = val
            
        labels[pair] = label
    
    return feature_vector, match_pairs, nonmatch_pairs, labels 


def compare(feature_vector, nfeatures, match_index, nonmatch_index, min_dim):
    
    inconsistent = True
    num_of_cons_dim = 0 # num of dimesions that matched pair is better than nonmatched pair
    
    for i in range(nfeatures):
        if feature_vector[match_index, i] > feature_vector[nonmatch_index, i]:
            num_of_cons_dim += 1
            if num_of_cons_dim >= min_dim: # consider that they are consistent
                inconsistent = False
                break
        
    return inconsistent

def brute_force(feature_vector, match_pairs, nonmatch_pairs, min_dim=1):
    
    (npairs, nfeatures) = feature_vector.shape
    print(npairs, nfeatures)
    
    num_of_violations = np.zeros(npairs, dtype = np.int32)
    violations = {}
    if debug:
        for p in match_pairs:
            violations[p[0]] = []
        for p in nonmatch_pairs:
            violations[p[0]] = []
            
    k = 10000
    print("Start pairwise with k=", k)
    
    for p1 in match_pairs[:k]:
        match_index = p1[1]
        for p2 in nonmatch_pairs[:k]:
            nonmatch_index = p2[1]
            inconsistent = compare(feature_vector, nfeatures, match_index, nonmatch_index, min_dim)
            if inconsistent == True:
                num_of_violations[match_index] += 1
                num_of_violations[nonmatch_index] += 1
                
                if debug:
                    violations[p1[0]].append(p2[0])
                    violations[p2[0]].append(p1[0])
    
    return num_of_violations, violations


def spartial_blocking(feature_vector, match_pairs, nonmatch_pairs, min_dim=1):

    num_of_violations = {}
    
    
    return num_of_violations

def mvc(num_of_violations):
    
    return