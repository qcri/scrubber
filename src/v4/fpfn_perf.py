'''
Created on Nov 20, 2018

@author: hzhang0418
'''
import os
import random
import time
import gc

import pandas as pd

import py_entitymatching as em

import v3.fpfn as fpfn
import v3.fpfn_svm as svm
import v3.fpfn_irf as ipn
import v4.loofpfn as loof
import v4.k_loofpfn as kloof
import v3.preprocessing as prep

import utils.myconfig

def comp_fpfn(config_file):
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
        tpath = gpath
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
            
    #num = 50
    #labels = random_flip(labels, num)
    
    '''
    alg = 'svm'
    for nfolds in [2, 5, 10, 20, 40, len(labels)]:
        indices = time_fpfn(features, labels, alg, nfolds)
        
        nfound = 0
        for index in indices:
            pair = index2pair[index]
            if labels[index]!=pair2correct[pair]:
                nfound += 1
        print(alg, 'nfolds='+str(nfolds), len(indices), nfound, len(indices)-nfound) 
    '''
    
    '''
    nfolds = len(labels)
    nmax = 10
    
    for alg in ['irf', 'loo', 'kloo']: #['fpfn', 'irf', 'loo', 'kloo']:
        indices = time_fpfn(features, labels, alg, nfolds, nmax)
        
        nfound = 0
        for index in indices:
            pair = index2pair[index]
            if labels[index]!=pair2correct[pair]:
                nfound += 1
        print(len(indices), nfound, len(indices)-nfound)   
    '''
    
    rank_alg = 'ml' # 'ml' or 'mono'
    fpfn_alg = 'irf'
    fs_alg = 'model'
    indices = time_fpfn(features, labels, alg=fpfn_alg, rank_alg=rank_alg, fs_alg=fs_alg)
    '''
    num = len(indices)
    nfound = 0
    for index in indices[: num]:
        pair = index2pair[index]
        if labels[index]!=pair2correct[pair]:
            nfound += 1
    print(num, nfound, len(indices)-nfound) 
    '''
    
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

    output_file = 'citations_irf_03072019.csv'
    df.to_csv(output_file, index=False)

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


def time_fpfn(features, labels, alg='fpfn', nfolds=5, nmax=10, rank_alg='ml', fs_alg='none'):
    start = time.time()
    if alg=='fpfn':
        m = fpfn.FPFN(features, labels, nfolds, rank_alg, fs_alg)
    elif alg=='svm':
        m = svm.FPFN(features, labels, nfolds, rank_alg, fs_alg)
    elif alg=='irf':
        m = ipn.FPFN_IRF(features, labels, nfolds, rank_alg, fs_alg)
    elif alg=='loo':
        m = loof.LooFPFN_IRF(features, labels)
    elif alg=='kloo':
        m = kloof.KLooFPFN_IRF(features, labels, nmax)
    else:
        print("Unsupported algorithm: ", alg)
        return
    
    indices = m.detect(len(labels))
    end = time.time()
    print(alg, (end-start), len(indices))
    
    return indices

def test():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/home.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_home.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono2019/citations_new.config'
    
    comp_fpfn(config_file)
    
    