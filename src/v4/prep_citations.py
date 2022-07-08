'''
Created on Feb 26, 2019

@author: hzhang0418
'''

import os
import random as rnd
import re

import v3.feature_generation as fg

#import py_entitymatching as em
import utils.myconfig
import utils.mycsv

def run():
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    config_file=r'/export/da/mkunjir/LabelDebugger/datasets/citeseer.config'
    # read config
    params = utils.myconfig.read_config(config_file)
    # dataset
    dataset_name = params['dataset_name']
    print(dataset_name)
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    
    table_A = utils.mycsv.read_csv_as_list(apath)
    table_B = utils.mycsv.read_csv_as_list(bpath)
    table_G = utils.mycsv.read_csv_as_list(gpath)
    
    matches = [t for t in table_G if t['label']=='1']
    candidates = find_candidate_pairs(table_A, table_B, title_max_diff=8)
    
    pairs2label = { (int(m['ltable.id']), int(m['rtable.id'])):m['label'] for m in matches}
    
    labeled_data = list(matches)
        
    count = 0
    for cand in candidates:
        if cand in pairs2label:
            count += 1
        else:
            labeled_data.append( {'_id':0, 'label':0, 'ltable.id':cand[0], 'rtable.id':cand[1]} )
    print("Number of candidates in given matches: ", count)

    new_gpath = os.path.join(basedir, 'new_labeled_data.csv')
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    rnd.shuffle(labeled_data)
    
    for index, data in enumerate(labeled_data):
        data['_id'] = index
        
    utils.mycsv.write_list_to_csv(labeled_data, fieldnames, new_gpath)
    
def generate_features():
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono2019/citations_new.config'
    config_file=r'/export/da/mkunjir/LabelDebugger/datasets/citeseer.config'
    # read config
    params = utils.myconfig.read_config(config_file)
    fg.generate_features(params)
    
'''

'''

def read_tables():
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    config_file=r'/export/da/mkunjir/LabelDebugger/datasets/citeseer.config'
    # read config
    params = utils.myconfig.read_config(config_file)
    # dataset
    dataset_name = params['dataset_name']
    print(dataset_name)
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    
    #table_A = em.read_csv_metadata(apath, key='id')
    #table_B = em.read_csv_metadata(bpath, key='id')
    #table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    table_A = utils.mycsv.read_csv_as_list(apath)
    table_B = utils.mycsv.read_csv_as_list(bpath)
    table_G = utils.mycsv.read_csv_as_list(gpath)
    
    matches = [t for t in table_G if t['label']=='1']
    
    return table_A, table_B, matches

def cluster_by_year(table, year_min_len=2):
    year2index = {}
    for index, row in enumerate(table):
        year = row['year']
        if len(year)<year_min_len:
            continue
        if year in year2index:
            year2index[year].append(index)
        else:
            year2index[year] = [index]
    return year2index

def estimate_size_after_year_blocking(table_A, table_B):
    year2index_A = cluster_by_year(table_A)
    year2index_B = cluster_by_year(table_B)
    
    size = 0
    for year in year2index_A.keys():
        if year in year2index_B:
            size += len(year2index_A[year])*len(year2index_B[year])
            
    print("Size after year blocking: ", size)

def cluster_by_title_length(table, indices, title_min_len=30):
    len2index = {}
    for index in indices:
        nlen = len(table[index]['title'])
        if nlen<title_min_len:
            continue
        if nlen in len2index:
            len2index[nlen].append(index)
        else:
            len2index[nlen] = [index]
    return len2index
        
def estimate_size_after_year_and_title_length_blocking(table_A, table_B, title_min_len=30, title_max_diff=3):
    year2index_A = cluster_by_year(table_A)
    year2index_B = cluster_by_year(table_B)
    
    size = 0
    for year in year2index_A.keys():
        if year in year2index_B:
            indices_A = year2index_A[year]
            indices_B = year2index_B[year]
            
            len2index_A = cluster_by_title_length(table_A, indices_A, title_min_len)
            len2index_B = cluster_by_title_length(table_B, indices_B, title_min_len)
            
            for nlen_A in len2index_A.keys():
                for nlen_B in len2index_B.keys():
                    if abs(nlen_A-nlen_B)<=title_max_diff:
                        size += len(len2index_A[nlen_A])*len(len2index_B[nlen_B])
                         
    print("Size after year and title length blocking: ", size)

def ngrams(sentence, n):
    nlen = len(sentence)
    tokens = set([ sentence[i: i+n] for i in range(nlen-n+1)])
    return tokens

def cluster_by_title_grams(table, indices, n):
    gram2index = {}
    for index in indices:
        title = table[index]['title']
        tokens = ngrams(title, n)
        for t in tokens:
            if t in gram2index:
                gram2index[t].append(index)
            else:
                gram2index[t] = [index]            
    return gram2index

def tokenize(s, min_len=6):
    t = re.sub('[^a-zA-Z]+', ' ', s)
    tmp = t.split()
    return set([t.lower() for t in tmp if len(t)>=min_len])

def cluster_by_title_tokens(table, indices):
    token2index = {}
    for index in indices:
        title = table[index]['title']
        tokens = tokenize(title)
        for t in tokens:
            if t in token2index:
                token2index[t].append(index)
            else:
                token2index[t] = [index]            
    return token2index
    
def estimate_size_after_token_blocking(table_A, table_B, title_min_len=30, title_max_diff=3):
    year2index_A = cluster_by_year(table_A)
    year2index_B = cluster_by_year(table_B)
    
    size = 0
    for year in year2index_A.keys():
        if year in year2index_B:
            indices_A = year2index_A[year]
            indices_B = year2index_B[year]
            
            len2index_A = cluster_by_title_length(table_A, indices_A, title_min_len)
            len2index_B = cluster_by_title_length(table_B, indices_B, title_min_len)
            
            tmp = {}
            for nlen_A in len2index_A.keys():
                index2tokens = { index:list(tokenize(table_A[index]['title'])) for index in len2index_A[nlen_A]} 
                
                for nlen_B in len2index_B.keys():
                    if abs(nlen_A-nlen_B)<=title_max_diff:
                        if nlen_B in tmp:
                            token2index_B = tmp[nlen_B]
                        else:
                            token2index_B = cluster_by_title_tokens(table_B, len2index_B[nlen_B])
                            tmp[nlen_B] = token2index_B
                        
                        for index_A in len2index_A[nlen_A]:
                            tokens = index2tokens[index_A]
                            two_common = []
                            for k in range(len(tokens)-2):
                                first = tokens[k]
                                if first not in token2index_B:
                                    continue
                                first_indices = set(token2index_B[first])
                                for i in range(k+1, len(tokens)-1):
                                    second = tokens[i]
                                    if second not in token2index_B:
                                        continue
                                    second_indices = set(token2index_B[second])
                                    for j in range(i+1, len(tokens)):
                                        third = tokens[j]
                                        if third not in token2index_B:
                                            continue
                                        third_indices = token2index_B[third]
                                        intersection = [index for index in third_indices if index in first_indices and index in second_indices]
                                        two_common.extend(intersection)
                            size += len(set(two_common))          
    print("Size after year and title token blocking: ", size)
            
def find_candidate_pairs(table_A, table_B, title_min_len=30, title_max_diff=3):
    
    year2index_A = cluster_by_year(table_A)
    year2index_B = cluster_by_year(table_B)
    
    size = 0
    candidates = []
    for year in year2index_A.keys():
        if year in year2index_B:
            indices_A = year2index_A[year]
            indices_B = year2index_B[year]
            
            len2index_A = cluster_by_title_length(table_A, indices_A, title_min_len)
            len2index_B = cluster_by_title_length(table_B, indices_B, title_min_len)
            
            tmp = {}
            for nlen_A in len2index_A.keys():
                index2tokens = { index:list(tokenize(table_A[index]['title'])) for index in len2index_A[nlen_A]} 
                
                for nlen_B in len2index_B.keys():
                    if abs(nlen_A-nlen_B)<=title_max_diff:
                        if nlen_B in tmp:
                            token2index_B = tmp[nlen_B]
                        else:
                            token2index_B = cluster_by_title_tokens(table_B, len2index_B[nlen_B])
                            tmp[nlen_B] = token2index_B
                        
                        for index_A in len2index_A[nlen_A]:
                            tokens = index2tokens[index_A]
                            two_common = []
                            for k in range(len(tokens)-2):
                                first = tokens[k]
                                if first not in token2index_B:
                                    continue
                                first_indices = set(token2index_B[first])
                                for i in range(k+1, len(tokens)-1):
                                    second = tokens[i]
                                    if second not in token2index_B:
                                        continue
                                    second_indices = set(token2index_B[second])
                                    for j in range(i+1, len(tokens)):
                                        third = tokens[j]
                                        if third not in token2index_B:
                                            continue
                                        third_indices = token2index_B[third]
                                        intersection = [index for index in third_indices if index in first_indices and index in second_indices]
                                        two_common.extend(intersection)
                            size += len(set(two_common))
                            for t in set(two_common):
                                candidates.append( (int(table_A[index_A]['id']), int(table_B[t]['id'])))  
    return candidates

def compare(matches, candidates):
    pairs2label = { (int(m['ltable.id']), int(m['rtable.id'])):m['label'] for m in matches}
    count = 0
    for cand in candidates:
        if cand in pairs2label:
            count += 1
    print("Number of candidates in given matches: ", count)
    
    return pairs2label


