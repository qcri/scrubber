'''
Created on Nov 17, 2018

@author: hzhang0418
'''
import os
import gc
import time
import random

import pandas as pd

import v3.brute_force as bf
import v3.spatial_blocking as sb
import v3.sort_probing as sp
import v3.preprocessing as prep
import py_entitymatching as em

import mp_parallel.mp_bf as mp_bf
import mp_parallel.mp_sb as mp_sb
import mp_parallel.mp_sp as mp_sp

import utils.myconfig

def mono(config_file):
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
    
    # read correct labels
    # dataset
    dataset_name = params['dataset_name']
    print(dataset_name, len(feature_names))
    
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    #hpath = os.path.join(basedir, params['hpath'])
    
    if dataset_name.startswith('citations_') or dataset_name == 'clothing' or dataset_name == 'home' or dataset_name == 'electronics' \
        or dataset_name == 'tools' or dataset_name == 'trunc_clothing' or dataset_name == 'trunc_home' \
        or dataset_name == 'trunc_electronics' or dataset_name == 'trunc_tools':
        tpath=gpath
    else:
        tpath = os.path.join(basedir, 'golden.csv')
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')    
    #table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    table_T = em.read_csv_metadata(tpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    pair2correct = {}
    if dataset_name.startswith('citations_') or dataset_name == 'clothing' or dataset_name == 'home' or dataset_name == 'electronics' \
        or dataset_name == 'tools' or dataset_name == 'trunc_clothing' or dataset_name == 'trunc_home' \
        or dataset_name == 'trunc_electronics' or dataset_name == 'trunc_tools': 
        for _, row in table_T.iterrows():
            pair = ( str(row['ltable.id']), str(row['rtable.id']) )
            pair2correct[pair] = row['label']
    else:
        for _, row in table_T.iterrows():
            pair = ( str(row['ltable.id']), str(row['rtable.id']) )
            pair2correct[pair] = row['golden']
            
    #num = 1000
    #labels = random_flip(labels, num)
    
    alg = 'sp' #'mp_sp'
    fs_alg = 'model' # none, rf, model
    indices = time_mono(features, labels, alg, fs_alg, nthreads=2)
        
    nfound = 0
    for index in indices:
        pair = index2pair[index]
        if labels[index]!=pair2correct[pair]:
            nfound += 1
    print(len(indices), nfound, len(indices)-nfound) 
    
    # combine those suspicious pairs into a dataframe and save to file
    all_pairs_with_label = []
    for index in indices[:200]:
        p = index2pair[index]
        label = labels[index]
        left = table_A.loc[ table_A['id'] == int(p[0])]
        right = table_B.loc[ table_B['id'] == int(p[1])]
        tmp = {}
        for col in left:
            tmp['ltable.'+col] = left.iloc[0][col]
        for col in right:
            tmp['rtable.'+col] = right.iloc[0][col]
        tmp['label'] = label
        all_pairs_with_label.append(tmp)
    df = pd.DataFrame(all_pairs_with_label)

    output_file = 'citations_mono_03072019.csv'
    df.to_csv(output_file, index=False)
    
    '''
    alg = 'sp'
    indices = time_mono(features, labels, alg, fs_alg)
        
    nfound = 0
    for index in indices:
        pair = index2pair[index]
        if labels[index]!=pair2correct[pair]:
            nfound += 1
    print(len(indices), nfound, len(indices)-nfound) 
    '''
    
    '''
    alg = 'mp_sp'
    for nthreads in [1, 2, 4]:
        indices = time_mono(features, labels, alg, nthreads)
        
        nfound = 0
        for index in indices:
            pair = index2pair[index]
            if labels[index]!=pair2correct[pair]:
                nfound += 1
        print(len(indices), nfound, len(indices)-nfound)   
    '''
        
        
def random_flip(labels, num):
    print("Random flip labels: ", num)
    random.seed(0)
    
    sampled = random.sample(range(len(labels)), num)
    
    for s in sampled:
        if labels[s] == 1:
            labels[s] = 0
        else:
            labels[s] = 1
    
    return labels  

def time_mono(features, labels, alg='sp', fs_alg='none', nthreads=1):
    print(alg, fs_alg, nthreads)
    start = time.time()
    if alg=='bf':
        m = bf.BruteForce(features, labels, 1, False, fs_alg)
    elif alg=='sb':
        m = sb.SpatialBlocking(features, labels, 1, 3, False, fs_alg)
    elif alg=='sp':
        m = sp.SortProbing(features, labels, 1, False, fs_alg)
    elif alg=='mp_bf':
        m = mp_bf.MP_BF(features, labels, 1, False, fs_alg, nthreads)
    elif alg=='mp_sb':
        m = mp_sb.MP_SB(features, labels, 1, 3, False, fs_alg, nthreads)
    elif alg=='mp_sp':
        m = mp_sp.MP_SP(features, labels, 1, False, fs_alg, nthreads)
    else:
        print("Unsupported algorithm: ", alg)
        return
    
    indices = m.detect(len(labels))
    end = time.time()
    print(alg, nthreads, (end-start), len(indices))
    
    return indices

def test():
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_20k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_50k.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_home.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono2019/citations_new.config'
    
    mono(config_file)    