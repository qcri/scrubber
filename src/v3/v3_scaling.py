'''
Created on Mar 1, 2017

@author: hzhang0418
'''
import os
import gc

import time
import random

import numpy as np

import v3.preprocessing as prep

import v3.fpfn as fpfn

import v3.brute_force as bf
import v3.sort_probing as sp

import utils.myconfig


def test():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/product.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    del feature_tuples[:]
    gc.collect()
    
    print("FPFN with original dataset:")
    test_fpfn(features, labels)
    
    num = 1000
    tmp = random_flip(np.copy(labels), num)
    test_fpfn(features, tmp)
    
    num = 0
    print("Mono with original dataset:")
    test_mono(features, labels)
    
    num = 50
    tmp = random_flip(np.copy(labels), num)
    test_mono(features, tmp)
    
    num = 100
    tmp = random_flip(np.copy(labels), num)
    test_mono(features, tmp)
    
    num = 200
    tmp = random_flip(np.copy(labels), num)
    test_mono(features, tmp)
    
    num = 400
    tmp = random_flip(np.copy(labels), num)
    test_mono(features, tmp)
    
    num = 1000
    tmp = random_flip(np.copy(labels), num)
    test_mono(features, tmp)

def random_flip(labels, num):
    print("Random flip labels: ", num)
    
    sampled = random.sample(range(len(labels)), num)
    
    for s in sampled:
        if labels[s] == 1:
            labels[s] = 0
        else:
            labels[s] = 1
    
    return labels

def test_mono(features, labels):
    start = time.time()
    # Mono
    #m = bf.BruteForce(features, labels, 1)
    m = sp.SortProbing(features, labels, 1)
    indices = m.detect(10)
    end = time.time()
    print("Time (secs): ", (end-start))
    print(indices)
    
    
def test_mono_inc_update():
    pass

def test_fpfn(features, labels):
    start = time.time()
    #FPFN
    f = fpfn.FPFN(features, labels, 5)    
    indices = f.detect(10)
    end = time.time()
    print("Time (secs): ", (end-start))
    print(indices)
    