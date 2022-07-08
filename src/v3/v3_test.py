'''
Created on Feb 16, 2017

@author: hzhang0418
'''
import os

import time
import numpy as np

import py_entitymatching as em
import v3.preprocessing as prep
import v3.fpfn as fpfn
import v3.dataset as ds
import v3.brute_force as bf
import v3.sort_probing as sp
import v3.spatial_blocking as sb
import v3.tool as tool
import v3.feature_selection as fs

import parallel.extractfeatures as ef

import utils.myconfig

def test():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/product.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_20k.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    
    #config_file=r'E:\\Projects\\eclipse\\DataCleaning\\product.config'
    #config_file=r'E:\\Projects\\eclipse\\DataCleaning\\citations_20k.config'
    #config_file=r'E:\\Projects\\eclipse\\DataCleaning\\citations_100k.config'
    #config_file=r'E:\\Projects\\eclipse\\DataCleaning\\citations_large.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    alg = 'mono'
    
    if alg=='mono':
        start = time.time()
        
        # Mono
        #m = bf.BruteForce(features, labels, 1, True)
        m = sp.SortProbing(features, labels, 1, True)
        #m = sb.SpatialBlocking(features, labels, 1, 2, True)
        
        indices = m.detect(5)
        end = time.time()
        print("Time (secs): ", (end-start))
        
        print(indices)
        
        #'''
        corrected_match_indices = []
        corrected_nonmatch_indices = []
        
        for index in indices:
            if labels[index] == 0:
                corrected_match_indices.append(index)
            else:
                corrected_nonmatch_indices.append(index)
                
        print(corrected_match_indices)
        print(corrected_nonmatch_indices)
        
        user_fb = {}
        for index in corrected_match_indices:
            user_fb[index] = 1
        for index in corrected_nonmatch_indices:
            user_fb[index] = 0
        
        m.use_feedback(user_fb)
        
        indices = m.detect(30)
        print(indices)
        #'''
    
    else:
        start = time.time()
        
        #FPFN
        f = fpfn.FPFN(features, labels, 5)
        
        indices = f.detect(10)
        
        print(indices)
        
        end = time.time()
        
        print("Time (secs): ", (end-start))
        
        
def test_tool():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    params['max_iter'] = 20
    params['top_k'] = 20
    params['approach'] = 'fpfn'
    #params['approach'] = 'mono'
    #params['approach'] = 'hybrid'
    
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    pair2golden = pp.read_golden()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    tl = tool.Tool(params, features, labels)
    tl.create_debugger()
    
    max_iter = params['max_iter']
    
    num_checked = 0
    num_errors = 0
    
    for k in range(max_iter):
        print("Iteration: ", k)
        indices = tl.detect()
        
        print(indices)
        if len(indices)==0:
            break
        
        # check labels to create feedback
        user_fb = {}
        for index in indices:
            golden = pair2golden[ index2pair[index] ]
            if labels[index] != golden:
                user_fb[index] = golden
        
        print("Feedback:", user_fb.keys())
        tl.get_feedback(user_fb)
        
        num_checked += len(indices)
        num_errors += len(user_fb)
    
    print("Number of pairs checked: ", num_checked)
    print("Number of errors found: ", num_errors)
    
def test_feature_selection():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    
    params['max_iter'] = 20
    params['top_k'] = 50
    params['approach'] = 'fpfn'
    #params['approach'] = 'mono'
    #params['approach'] = 'hybrid'
    
    # preprocessing
    pp = prep.Preprocessing(params)
    
    feature_header, feature_tuples = pp.read_features()
    pair2golden = pp.read_golden()
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    feature_names = [ name for name in feature_header if name not in tmp ]
    
    index2pair, features, labels = pp.convert_from_tuples(feature_header, feature_tuples, feature_names)
    
    #selected = fs.select_features(features, labels, 5, 3)
    #print(selected)
    
    start = time.time()
        
    # Mono
    #m = bf.BruteForce(features, labels, 1)
    #m = sp.SortProbing(features, labels, 1)
    m = sb.SpatialBlocking(features, labels, 1, 2)
    
    indices = m.detect(5)
    end = time.time()
    print("Time (secs): ", (end-start))
    
    print(indices)
    
def test_feature_computation():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/tmp.config'
    
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
    
    
    table_A, table_B, table_features = ds.retrieve(dataset_name, apath, bpath, gpath)
    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')

    ds.compute_feature_vector(table_G, table_features, hpath)
    
    
def test_parallel():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/tmp.config'
    
    start = time.clock()
    
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
    
    table_A, table_B, table_features = ds.retrieve(dataset_name, apath, bpath, gpath)
    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    print(len(table_G))
    
    end = time.clock()
    
    print("Time to read: ", (end-start))
    
    num_jobs=1
    
    start = time.time()
    table_H = ef.extract_feature_vecs(table_G, attrs_before=None, feature_table=table_features, attrs_after=["label"], show_progress=False, n_jobs=num_jobs)
    end = time.time()
    
    print("Time with njobs=", num_jobs , ":", (end-start))
    print(len(table_H))
    
    num_jobs=2
    
    start = time.time()
    table_H = ef.extract_feature_vecs(table_G, attrs_before=None, feature_table=table_features, attrs_after=["label"], show_progress=False, n_jobs=num_jobs)
    end = time.time()
    
    print("Time with njobs=", num_jobs , ":", (end-start))
    print(len(table_H))
    
    num_jobs=3
    
    start = time.time()
    table_H = ef.extract_feature_vecs(table_G, attrs_before=None, feature_table=table_features, attrs_after=["label"], show_progress=False, n_jobs=num_jobs)
    end = time.time()
    
    print("Time with njobs=", num_jobs , ":", (end-start))
    print(len(table_H))
    
    #table_H.fillna(0, inplace=True)
    #table_H.to_csv(hpath)
    
    