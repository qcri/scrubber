'''
Created on Apr 17, 2019

@author: hzhang0418
'''

import os
import random
import re

import pandas as pd

from unidecode import unidecode

import utils.myconfig
import utils.mycsv

def read_tables():
    config_file=r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs_1m.config'
    # read config
    params = utils.myconfig.read_config(config_file)
    # dataset
    dataset_name = params['dataset_name']
    print(dataset_name)
    # base dir
    basedir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs'
    # path for table A, B, G, H
    apath = os.path.join(basedir, 'msd.csv')
    mpath = os.path.join(basedir, 'matches_msd_msd.csv')
    
    table_A = pd.read_csv(apath)
    table_G = pd.read_csv(mpath)
    
    return table_A, table_G

def check_title(table_A, table_G):
    id_col = table_A['id'].values
    att_col = table_A['title'].astype('str').values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2value[ t['id1'] ]
        right = A_id2value[ t['id2'] ]
        
        if left.lower()==right.lower():
            count += 1
    
    print("Percentage with same title is: ", 1.0*count/len(table_G))
    
def check_exact_same(table_A, table_G, attribute):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2value[ t['id1'] ]
        right = A_id2value[ t['id2'] ]
        
        if left==right:
            count += 1
    
    print("Percentage with same value for given attribute ", attribute, " is: ", 1.0*count/len(table_G))
    
def tokenize(s, min_len=6):
    t = re.sub('[^a-zA-Z]+', ' ', s)
    tmp = t.split()
    return set([t.lower() for t in tmp if len(t)>=min_len])
    
def check_num_common_word(table_A, table_G, attribute, min_num=1, min_len=3):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].astype('str').values
    
    A_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2words[ t['id1'] ]
        right = A_id2words[ t['id2'] ]
        
        if len(left.intersection(right))>=min_num:
            count += 1
    
    print("Percentage with min common word for given attribute ", attribute, " is: ", 1.0*count/len(table_G))
    
def tokenize_attribute(table_A, attribute, min_len=3):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].astype('str').values
    
    A_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    return A_id2words
    
def cluster_by_attribute(table_A, attribute):
    id_col = table_A['id'].values
    att_col = table_A[attribute].values
    
    A_value2ids = {}
    for k,v in zip(att_col, id_col):
        if k in A_value2ids:
            A_value2ids[k].append(v)
        else:
            A_value2ids[k] = [v]
            
    return A_value2ids

def hash_blocking(table_A, attribute):
    A_value2ids = cluster_by_attribute(table_A, attribute)
    
    num_cand = 0
    for k,v in A_value2ids.items():
        num_cand += len(v)**2
        
    print("Number of candidates: ", num_cand)
    
def blocking_year_and_min_common_words(table_A, attribute, min_len=3):
    A_value2ids = cluster_by_attribute(table_A, 'year')
    A_id2words = tokenize_attribute(table_A, attribute, min_len)
    
    num_cand = 0
    for k,v in A_value2ids.items():
        # word to ids
        w2ids = {}
        for i in v:
            words = A_id2words[i]
            for w in words:
                if w in w2ids:
                    w2ids[w].append(i)
                else:
                    w2ids[w] = [i]
        
        for ids in w2ids.values():
            num_cand += len(ids)**2
            
        
    print("Number of candidates: ", num_cand)
    
def check_value_max_diff(table_A, table_G, attribute, max_diff):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].astype('float').values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2value[ t['id1'] ]
        right = A_id2value[ t['id2'] ]
        
        if abs(left-right)<=max_diff:
            count += 1
    
    print("Percentage with same value for given attribute ", attribute, " is: ", 1.0*count/len(table_G))
    
def blocking_year_and_duration_max_diff(table_A, max_diff=10):
    A_value2ids = cluster_by_attribute(table_A, 'year')
    
    id_col = table_A['id'].values
    att_col = table_A['duration'].astype('float').values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    num_cand = 0
    for k,v in A_value2ids.items():
        for v1 in v:
            d1 = A_id2value[v1]
            for v2 in v:
                d2 = A_id2value[v2]
                
                if abs(d1-d2)<=max_diff:
                    num_cand += 1
                
        
    print("Number of candidates: ", num_cand)
    
def check_len_max_diff(table_A, table_G, attribute, max_diff):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].astype('str').values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2value[ t['id1'] ]
        right = A_id2value[ t['id2'] ]
        
        if abs(len(left)-len(right))<=max_diff:
            count += 1
    
    print("Percentage with same value for given attribute ", attribute, " is: ", 1.0*count/len(table_G))
    
def blocking_year_and_title_len_max_diff(table_A, max_diff=3):
    A_value2ids = cluster_by_attribute(table_A, 'year')
    
    id_col = table_A['id'].values
    att_col = table_A['title'].astype('str').values
    
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    num_cand = 0
    for k,v in A_value2ids.items():
        # title len to ids
        tl2ids = {}
        for i in v:
            tl = len(A_id2value[i])
            if tl in tl2ids:
                tl2ids[tl].append(i)
            else:
                tl2ids[tl] = [i]
                
        for ids in tl2ids.values():
            num_cand += len(ids)**2

    print("Number of candidates: ", num_cand)
    
def test():
    table_A, table_G = read_tables()
    #check_exact_same(table_A, table_G, 'year')
    #check_num_common_word(table_A, table_G, 'title', min_len=7)
    #check_value_max_diff(table_A, table_G, 'duration', 10)
    #check_len_max_diff(table_A, table_G, 'title', 0)
    
    check_title(table_A, table_G)
    
    #hash_blocking(table_A, 'year')
    #blocking_year_and_min_common_words(table_A, 'title', min_len=7)
    #blocking_year_and_duration_max_diff(table_A, max_diff=10)
    #blocking_year_and_title_len_max_diff(table_A, max_diff=3)
    

def gen_candidates(table_A, table_G):
    '''
    same year and title
    '''
    id_col = table_A['id'].values
    year_col = table_A['year'].values
    title_col = table_A['title'].astype('str').values
    
    ytl2ids = {}
    for y,t,i in zip(year_col, title_col, id_col):
        p = (y,t.lower())
        if p in ytl2ids:
            ytl2ids[p].append(i)
        else:
            ytl2ids[p] = [i]
    
    candidates = set()
    
    for v in ytl2ids.values():
        for v1 in v:
            for v2 in v:
                candidates.add( (v1, v2) )
                
    print(len(candidates))
                
    # add those from golden
    id1_col = table_G['id1'].values
    id2_col = table_G['id2'].values
    
    for v1,v2 in zip(id1_col, id2_col):
        candidates.add((v1,v2))
    
    print(len(candidates))
    
    return candidates
    
def gen_labeled_data(table_A, table_G, candidates):
    id1_col = table_G['id1'].values
    id2_col = table_G['id2'].values
    
    matches = set()
    for v1,v2 in zip(id1_col, id2_col):
        matches.add((v1,v2))
        
    pairs = set(matches)
    
    random.seed(0)
    
    A_last_index = len(table_A) - 1
    B_last_index = len(table_A) - 1
    
    num = len(matches)
    count = 0
    
    id_col = table_A['id'].values
    
    while True:
        index_A = random.randint(0, A_last_index)
        index_B = random.randint(0, B_last_index)
            
        p = (id_col[index_A], id_col[index_B])
        if p in candidates or p in pairs:
            continue
        
        pairs.add(p)
        count += 1
        
        if count>=num:
            break
    
    pairs_list = []
    for p in pairs:
        label = 1 if p in matches else 0
        pairs_list.append({'ltable.id':p[0], 'rtable.id':p[1], 'label':label})
        
    print(len(pairs_list))
    
    random.shuffle(pairs_list)
    
    for k, p in enumerate(pairs_list):
        p['_id'] = k
    
    labeled_pairs_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs/labeled_data.csv'
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    utils.mycsv.write_list_to_csv(pairs_list, fieldnames, labeled_pairs_file)
    
import v4.dataset_dl
        
def gen_features():
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs_1m.config'
    
    v4.dataset_dl.prepare_features(config_file) 