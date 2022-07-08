'''
Created on Nov 21, 2016

@author: hzhang0418
'''

import dataprocessing as dpc
import bruteforce as bfc
import sortprobing as spc

import os
import time

import monotonicity.procedures as proc
import utils.myconfig


def test():
    # init jvm
    proc.init_jvm()
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/product.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_20k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_50k.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    # dataset
    dataset_name = params['dataset_name']
    
    # base dir
    basedir = params['basedir']
    
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    hpath = os.path.join(basedir, params['hpath'])
    
    min_cons_dim = 1
    
    #test_bf(hpath, min_cons_dim)
    
    test_sp(hpath)
    
    
def test_bf(hpath, min_cons_dim=1):
    '''
    Brute force
    '''
    print("Brute Force:")
    
    start = time.time()
    print("Start:", start)
    header, tuples = dpc.read_feature_vector(hpath)
    print("Finish reading feature vector file at time: ", time.time()-start)
    
    feature_names = []
    for f in header:
        if f=='_id' or f=='ltable.id' or f=='rtable.id' or f=='label':
            continue
        feature_names.append(f)
    
    index2pair, feature_vectors, nfeatures, match_indices, nonmatch_indices, index2label = dpc.convert_for_python(header, tuples, feature_names)
    
    index2incons = bfc.brute_force(header, tuples, feature_names, min_cons_dim)
    
    print("Finish compute inconsistent indices at time:",  time.time()-start)
    
    tmp = [ len(t) for t in index2incons.values() ]
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
def test_sp(hpath):
    '''
    Sort Probing
    '''
    print("Sort Probing:")
    
    start = time.time()
    print("Start:", start)
    header, tuples = dpc.read_feature_vector(hpath)
    print("Finish reading feature vector file at time: ", time.time()-start)
    
    feature_names = []
    for f in header:
        if f=='_id' or f=='ltable.id' or f=='rtable.id' or f=='label':
            continue
        feature_names.append(f)
    
    index2incons = spc.sort_probing(header, tuples, feature_names)
    
    print("Finish compute inconsistent indices at time:",  time.time()-start)
    
    tmp = [ len(t) for t in index2incons.values() ]
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))