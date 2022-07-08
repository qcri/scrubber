'''
Created on Oct 24, 2016

@author: hzhang0418
'''
import os
import time
import numpy as np

import magellan as mg
import monotonicity.procedures as proc
import monotonicity.dataset as ds
import monotonicity.experiment as exp

import monotonicity.spatial_blocking as sb

import v2.spatial_blocking_v2 as sbv2
import v2.spatial_blocking_v3 as sbv3
import v2.spatial_blocking_v4 as sbv4
import v2.sort_probing as sp
import v2.sort_probing_v2 as spv2
import v2.sort_probing_v3 as spv3

import v2.brute_force as bf

import utils.myconfig

def v1(dataset_name, apath, bpath, gpath, hpath, min_dim=1):
    '''
    spatial blocking v1
    '''
    
    print("Spatial Blocking V1:")
    
    start = time.time()
    print("Start:", start)
    # read tables
    table_A, table_B, table_features = ds.retrieve(dataset_name, apath, bpath, gpath)
    
    print("Start to read feature vector at time: ", time.time()-start)
    table_H = mg.read_csv(hpath, key='_id', ltable = table_A, rtable = table_B)
    print("Finish reading at time: ", time.time()-start)
    
    feature_names = table_features["feature_name"].tolist()
    print(len(feature_names))
    
    headers = list(table_H.columns.values)
    print( set(feature_names) - set(headers) )
    print( set(headers) - set(feature_names) )
    
    tmp = []
    for f in feature_names:
        if f in headers:
            tmp.append(f)
    print(len(tmp))
    feature_names = tmp
    
    print("Start to convert feature vector at time: ", time.time()-start)
    # now prepare data to search for monotonicity
    (positives, negatives, features, labels) = exp.convertDataFrameFeatures(table_H, feature_names)
    print("Finish converting feature vector at time: ", time.time()-start)
    
    block = sb.SpatialBlock(positives, negatives, features, 4, min_dim)
    incons = block.get_inconsistent()
    print("Finish compute list of inconsistent pairs at time: ", time.time()-start)
    
    incons = exp.sortIncons(incons)
    
    tmp = []
    for i in range(10):
        if i>=len(incons):
            break
        tmp.append(len(incons[i][1]))
    print (tmp)
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))

def v2(hpath, min_dim=1):
    '''
    spatial blocking v2
    '''
    
    print("Spatial Blocking V2:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    feature_vector, nfeatures, match_pairs, nonmatch_pairs, labels = sbv2.read_feature_vector(hpath)
    #num_violations, _ = v2.mono_v2.brute_force(feature_vector, match_pairs, nonmatch_pairs, min_dim)
    #num_violations.sort()
    #print(num_violations[-10:])
    print("Finish reading at time: ", time.time()-start)
    
    #k = 10000
    
    '''
    block = sbv2.SpatialBlockV2(feature_vector, nfeatures, match_pairs, nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent()
    print("Finish compute #inconsistent pairs at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    block = sbv2.SpatialBlockV2(feature_vector, nfeatures, match_pairs, nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent_pairs()
    print("Finish compute list of inconsistent pairs at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    '''
    
    block = sbv2.SpatialBlockV2(feature_vector, nfeatures, match_pairs, nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent_origin()
    print("Finish compute #inconsistent pairs with original method at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))

def v3(hpath, min_dim=1):
    '''
    spatial blocking v3
    '''
    
    print("Spatial Blocking V3:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels = sbv3.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)

    #k = 10000
    
    '''
    block = sbv3.SpatialBlockV3(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent()
    print("Finish compute #inconsistent pairs at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    block = sbv3.SpatialBlockV3(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent_indices()
    print("Finish compute list of inconsistent pairs at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    '''
    
    block = sbv3.SpatialBlockV3(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_dim)
    incons_count_map, incons_pair_map = block.get_inconsistent_origin()
    print("Finish compute #inconsistent pairs with original method at time: ", time.time()-start)
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
def test_bf(hpath, min_cons_dim=1):
    '''
    Brute force
    '''
    
    print("Brute Force:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, match_indices, nonmatch_indices, index2labels = sbv3.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)
    
    index2incons = bf.brute_force(feature_vector, nfeatures, match_indices, nonmatch_indices, min_cons_dim)
    print("Finish compute inconsistent indices at time:",  time.time()-start)
    
    tmp = [ len(t) for t in index2incons.values() ]
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
    
def v4(hpath, min_cons_dim):
    '''
    spatial blocking v3
    '''
    
    print("Spatial Blocking V3:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels = bf.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)
    
    '''
    block = sbv4.SpatialBlockV4(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_cons_dim)
    incons_count_map = block.count_inconsistent()
    tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    print("Finish compute #inconsistent pairs at time: ", time.time()-start)
    '''
    
    '''
    block = sbv4.SpatialBlockV4(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_cons_dim)
    incons_index_map = block.get_inconsistent_indices()
    tmp = [ len(t) for t in incons_index_map.values() ]
    tmp.sort(reverse = True)
    print(tmp[:10])
    
    print("Finish compute the list of inconsistent pairs at time: ", time.time()-start)
    '''
    
    block = sbv4.SpatialBlockV4(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, 4, min_cons_dim)
    incons_count_map = block.count_inconsistent_for_dense_dataset()
    tmp = tmp = list(incons_count_map.values())
    tmp.sort(reverse = True)
    print(tmp[:20])
    
    print("Finish compute the list of inconsistent pairs at time: ", time.time()-start)
    
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
def sp_test(hpath):
    '''
    sort probing
    '''
    
    print("Sort Probing:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels = bf.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)
    
    
    agent = sp.SortProbing(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs)
    
    incons_index_map = agent.get_inconsistency_indices()
    tmp = [ len(t) for t in incons_index_map.values() ]
    tmp.sort(reverse = True)
    print(tmp[:20])
    
    print("Finish compute the list of inconsistent pairs at time: ", time.time()-start)
    
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
def spv2_test(hpath):
    '''
    sort probing bitarray
    '''
    
    print("Sort Probing with Bitarray:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels = bf.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)
    
    
    agent = spv2.SortProbingV2(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs)
    
    incons_index_map = agent.get_inconsistency_indices()
    tmp = [ len(t) for t in incons_index_map.values() ]
    tmp.sort(reverse = True)
    print(tmp[:20])
    
    print("Finish compute the list of inconsistent pairs at time: ", time.time()-start)
    
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))
    
def spv3_test(hpath):
    '''
    sort probing bitarray v3
    '''
    
    print("Sort Probing with Bitarray V3:")
    
    start = time.time()
    print("Start:", start)
    #feature_vector, match_pairs, nonmatch_pairs, labels = v2.mono_v2.prepare(table_H, table_features)
    index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels = bf.read_feature_vector(hpath)
    print("Finish reading at time: ", time.time()-start)
    
    
    agent = spv3.SortProbingV3(feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs)
    
    incons_index_map = agent.get_inconsistency_indices()
    tmp = [ len(t) for t in incons_index_map.values() ]
    tmp.sort(reverse = True)
    print(tmp[:20])
    
    print("Finish compute the list of inconsistent pairs at time: ", time.time()-start)
    
    
    end = time.time()
    print("End:", end)
    
    print("Time (secs): ", (end-start))

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
    
    #v1(dataset_name, apath, bpath, gpath, hpath, min_dim)
    
    #v2(hpath, min_dim)
    
    #v3(hpath, min_dim)
    
    #test_bf(hpath, min_cons_dim)
    
    #v4(hpath, min_cons_dim)
    
    #sp_test(hpath)
    
    #spv2_test(hpath)
    
    spv3_test(hpath)

    