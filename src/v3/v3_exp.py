'''
Created on Mar 1, 2017

@author: hzhang0418
'''

import os
import sys
sys.stdout = open('stdout.txt', 'a')

import py_entitymatching as em
import v3.dataset as ds

#import magellan as mg
#import monotonicity.dataset as ds_old

import v3.noise as noise
import v3.preprocessing as prep
import v3.tool as tool

import utils.myconfig
import utils.mylogger

debug = False

def compare_label_with_golden(config_file, noise_method="random", percentage = 0.05, seed=0, top_k = 20, max_iter = 20, logger=None):
    log_info = []
    
    # read config
    params = utils.myconfig.read_config(config_file)
    # dataset
    dataset_name = params['dataset_name']
    
    log_info.append(dataset_name)
    log_info.extend( [noise_method, percentage, seed, top_k, max_iter] )
    
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    hpath = os.path.join(basedir, params['hpath'])
    tpath = os.path.join(basedir, 'golden.csv')
    
    
    table_A, table_B, table_features = ds.retrieve(dataset_name, apath, bpath, gpath)
    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    if os.path.isfile(hpath) == False:
        ds.compute_feature_vector(table_G, table_features, hpath)
        
    table_H = em.read_csv_metadata(hpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    table_T = em.read_csv_metadata(tpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    
    '''
    # for old magellan
    table_A, table_B, table_features = ds_old.retrieve(dataset_name, apath, bpath, gpath)
    table_G = mg.read_csv(gpath, key='_id', ltable = table_A, rtable = table_B, foreign_key_ltable='ltable.id', foreign_key_rtable='rtable.id')
    if os.path.isfile(hpath) == False:
        ds.compute_feature_vector(table_G, table_features, hpath)
        
    table_H = mg.read_csv(hpath, key='_id', ltable = table_A, rtable = table_B, foreign_key_ltable='ltable.id', foreign_key_rtable='rtable.id')
    
    table_T = mg.read_csv(tpath, key='_id', ltable = table_A, rtable = table_B, foreign_key_ltable='ltable.id', foreign_key_rtable='rtable.id')
    '''
    
    num_noise = int(len(table_H)* percentage)
    #print("Number of true noise: ", num_noise)
    
    if noise_method == 'random':
        match_ratio = 0.5
        index_of_noises = noise.select_noise(table_G, table_T, num_noise, match_ratio, seed)
    else:
        index_of_noises = noise.select_noise_using_AL(table_G, table_H, table_T, num_noise, seed)
    
    noise.insert_noise(table_G, table_T, index_of_noises)
    noise.insert_noise(table_H, table_T, index_of_noises)
    log_info.extend(noise.check_noise(table_G, table_T))
    
    use_mvc = True
    use_mvc = False
    log_info.append(use_mvc)
    
    
    print("Mono approach:")
    table_H_tmp = table_H.copy()
    info, mono_errors = iterate(params, table_A, table_B, table_G, table_H_tmp, table_features, table_T, "mono", top_k, max_iter, use_mvc)
    log_info.extend(info)
    
    
    print("FPFN approach:")
    table_H_tmp = table_H.copy()
    info, fpfn_errors = iterate(params, table_A, table_B, table_G, table_H_tmp, table_features, table_T, "fpfn", top_k, max_iter)
    log_info.extend(info)
    
    
    print("Hybrid approach:")
    table_H_tmp = table_H.copy()
    info, hybrid_errors = iterate(params, table_A, table_B, table_G, table_H_tmp, table_features, table_T, "hybrid", top_k, max_iter, use_mvc)
    log_info.extend(info)
    
    
    if logger is not None:
        logger.log_tuple(log_info)
        logger.flush()
        
    # error comparison
    #error_analysis(mono_errors, fpfn_errors, hybrid_errors)
    
    print(log_info)
    print('\n\n\n')
    
    
def error_analysis(mono_errors, fpfn_errors, hybrid_errors):
    if type(mono_errors[0][0]) is str or type(fpfn_errors[0][0]) is str:
        A = set([ (t[0], t[1]) for t in mono_errors ])
        B = set([ (t[0], t[1]) for t in fpfn_errors ])
        C = set([ (t[0], t[1]) for t in hybrid_errors ])
    else:
        A = set([ (int(t[0]), int(t[1])) for t in mono_errors ])
        B = set([ (int(t[0]), int(t[1])) for t in fpfn_errors ])
        C = set([ (int(t[0]), int(t[1])) for t in hybrid_errors ])
    
    mono_and_fpfn = A.intersection(B)
    mono_not_fpfn = A.difference(B)
    fpfn_not_mono = B.difference(A)
    hybrid_only = C.difference(A.union(B))
    not_hybrid = A.union(B).difference(C)
    
    print("Errors detected by both mono and fpfn: ", len(mono_and_fpfn))
    print(mono_and_fpfn)
    print("Errors detected by mono but not fpfn: ", len(mono_not_fpfn))
    print(mono_not_fpfn)
    print("Errors detected by fpfn but not mono: ", len(fpfn_not_mono))
    print(fpfn_not_mono)
    print("Errors detected by hybrid only: ", len(hybrid_only))
    print(hybrid_only)
    print("Errors detected by the other two but not hybrid:", len(not_hybrid))
    print(not_hybrid)

def iterate(params, table_A, table_B, table_G, table_H, table_features, table_T, approach="fpfn", top_k=20, max_iter=20, use_mvc=True):
    info = []
    
    user_labels = {}
    
    for index, row in table_H.iterrows():
        pair = (row['ltable.id'], row['rtable.id'])
        label = row['label']
        user_labels[pair] = label, index
        
    golden_labels = {}
    for index, row in table_T.iterrows():
        pair = (row['ltable.id'], row['rtable.id'])
        label = row['golden']
        golden_labels[pair] = label, index
    
    # convert from dataframe    
    pp = prep.Preprocessing(params)
    
    feature_table = table_H
    feature_names = list(table_H)
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    for t in tmp:
        feature_names.remove(t)
        
    #print(feature_names)
    
    index2pair, features, labels = pp.convert_from_dataframe(feature_table, feature_names)
    
    params['approach'] = approach
    params['top_k'] = top_k
    params['max_iter'] = max_iter
    params['use_mvc'] = use_mvc
    
    tl = tool.Tool(params, features, labels)
    tl.create_debugger()
    
    checked_pairs = set()
    found_errors = []
    
    num_iter_without_error = 0
    
    for i in range(max_iter):
        ranked = tl.detect()
        
        #print(ranked)
        
        num_errors = 0
        error_pairs = []
        user_fb = {}
        for index in ranked:    
            p = index2pair[index]
            checked_pairs.add(p)
            
            # compare and correct wrong labels
            golden = golden_labels[p][0]
            if user_labels[p][0] != golden:
                num_errors += 1
                error_pairs.append(p)
                user_fb[index] = golden     
        
        print("Number of errors found: ", num_errors)
        found_errors.extend(error_pairs)
        
        '''
        if len(error_pairs)==0:
            print("No new errors at iteration: ", i)
            break
        '''
        
        if len(error_pairs)==0:
            num_iter_without_error += 1
            if num_iter_without_error>=3:
                print("No new errors in last three iterations: ", i)
                break
        else:
            num_iter_without_error = 0
        
        #print("Feedback:", user_fb.keys())
        tl.get_feedback(user_fb)
    
    print("Total number of checked pairs: ", len(checked_pairs))
    print("Total number of found errors:", len(found_errors))
    
    info.extend([ approach, i, len(checked_pairs), len(found_errors) ])
    
    return info, found_errors
    
def test():
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurant.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/product.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/anime.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/baby.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cosmetics.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/electronics.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/music.config'
    
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    
    run_real_dataset(config_file)
    
    run_simulation(config_file)
    
    sys.stdout.flush()
    
def run_real_dataset(config_file):
    log_file=r'/scratch/hzhang0418/projects/datasets/mono/20170301.log'
    logger = utils.mylogger.MyLogger(log_file)
    logger.open()
    logger.log_datetime()
    
    seeds = [0]
    percs = [0]
    
    top_k = 20
    max_iter = 100
    
    for percentage in percs:
        noise_method = 'active'
        for seed in seeds:
            compare_label_with_golden(config_file, noise_method, percentage, seed, top_k, max_iter, logger)
    
        '''
        noise_method = 'random'
        for seed in seeds:
            compare_label_with_golden(config_file, noise_method, percentage, seed, top_k, max_iter, logger)
        '''
    
    logger.close()
    
def run_simulation(config_file):
    
    log_file=r'/scratch/hzhang0418/projects/datasets/mono/20170301.log'
    logger = utils.mylogger.MyLogger(log_file)
    logger.open()
    logger.log_datetime()
    
    #seeds = [0]
    #percs = [0]
    
    percs = [0.05]
    
    percs = [0.02, 0.03]
    seeds = [0, 4876, 182428, 301077, 789243]
    
    top_k = 20
    max_iter = 100
    
    for percentage in percs:
        noise_method = 'active'
        for seed in seeds:
            compare_label_with_golden(config_file, noise_method, percentage, seed, top_k, max_iter, logger)
    
        noise_method = 'random'
        for seed in seeds:
            compare_label_with_golden(config_file, noise_method, percentage, seed, top_k, max_iter, logger)
    
    logger.close()
    
    
def simple_test():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    log_file=r'/scratch/hzhang0418/projects/datasets/mono/20170301.log'
    logger = utils.mylogger.MyLogger(log_file)
    logger.open()
    logger.log_datetime()
    
    top_k = 20
    max_iter = 30

    noise_method = 'random'
    compare_label_with_golden(config_file, noise_method, 0, 0, top_k, max_iter, logger)
         
    #noise_method = 'random'
    #compare_label_with_golden(config_file, noise_method, 0.05, 0, top_k, max_iter, logger)
    
    logger.close()
    
    sys.stdout.flush()
    