'''
Created on Apr 18, 2019

@author: hzhang0418
'''

import os
import random
import re

import pandas as pd

import utils.myconfig
import utils.mycsv

def read_tables():
    #config_file=r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k.config'
    config_file=r'/export/da/mkunjir/LabelDebugger/config/citeseer.config'
    # read config
    params = utils.myconfig.read_config(config_file)
    # dataset
    dataset_name = params['dataset_name']
    print(dataset_name)
    # base dir
    basedir = params['basedir'] #r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k'
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    mpath = os.path.join(basedir, 'matches_citeseer_dblp.csv')
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    table_G = pd.read_csv(mpath)
    
    return table_A, table_B, table_G

def check_exact_same(table_A, table_B, table_G, attribute):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].values
    A_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    id_col = table_B['id'].values
    att_col = table_B[attribute].values
    B_id2value = { k:v for k,v in zip(id_col, att_col)  }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2value[ t['id1'] ]
        right = B_id2value[ t['id2'] ]
        
        if left==right:
            count += 1
    
    print("Percentage with same value for given attribute ", attribute, " is: ", 1.0*count/len(table_G))

def tokenize(s, min_len=6):
    t = re.sub('[^a-zA-Z]+', ' ', s)
    tmp = t.split()
    return set([t.lower() for t in tmp if len(t)>=min_len])

def check_num_common_word(table_A, table_B, table_G, attribute, min_num=1, min_len=3):
    
    id_col = table_A['id'].values
    att_col = table_A[attribute].astype('str').values
    A_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    id_col = table_B['id'].values
    att_col = table_B[attribute].astype('str').values
    B_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    count = 0
    for _, t in table_G.iterrows():
        left = A_id2words[ t['id1'] ]
        right = B_id2words[ t['id2'] ]
        
        if len(left.intersection(right))>=min_num:
            count += 1
    
    print("Percentage with min common word for given attribute ", attribute, " is: ", 1.0*count/len(table_G))
    
def blocking_on_title_min_two_common_words(table_A, table_B, min_len=3):
    id_col = table_A['id'].values
    att_col = table_A['title'].astype('str').values
    A_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    id_col = table_B['id'].values
    att_col = table_B['title'].astype('str').values
    B_id2words = { k:tokenize(v, min_len) for k,v in zip(id_col, att_col) }
    
    A_w2ids = {}
    for k,v in A_id2words.items():
        for w in v:
            if w in A_w2ids:
                A_w2ids[w].append(k)
            else:
                A_w2ids[w] = [k]
                
    B_w2ids = {}
    for k,v in B_id2words.items():
        for w in v:
            if w in B_w2ids:
                B_w2ids[w].append(k)
            else:
                B_w2ids[w] = [k]
                
    common_words = set(A_w2ids.keys()).intersection(set(B_w2ids.keys()))
    print("Num of common words: ", len(common_words))
    
    A_p2ids = {}
    for i, words in A_id2words.items():
        tmp = [w for w in words if w in common_words]
        num = len(tmp)
        for index1 in range(num):
            w1 = tmp[index1]
            for index2 in range(index1+1, num):
                w2 = tmp[index2]
                p = (w1, w2) if w1<w2 else (w2, w1)
                if p in A_p2ids:
                    A_p2ids[p].append(i)
                else:
                    A_p2ids[p] = [i]
                    
                
    
    B_p2ids = {}
    for i, words in B_id2words.items():
        tmp = [w for w in words if w in common_words]
        num = len(tmp)
        for index1 in range(num):
            w1 = tmp[index1]
            for index2 in range(index1+1, num):
                w2 = tmp[index2]
                p = (w1, w2) if w1<w2 else (w2, w1)
                if p in B_p2ids:
                    B_p2ids[p].append(i)
                else:
                    B_p2ids[p] = [i]
                    
    id_col = table_A['id'].values
    att_col = table_A['authors'].astype('str').values
    A_id2author_words = { k:tokenize(v, 5) for k,v in zip(id_col, att_col) }
    
    id_col = table_B['id'].values
    att_col = table_B['authors'].astype('str').values
    B_id2author_words = { k:tokenize(v, 5) for k,v in zip(id_col, att_col) }
    
    candidates = []
            
    num_cand = 0
    for p, ids in A_p2ids.items():
        if p in B_p2ids:
            ids_B = B_p2ids[p]
            for a in ids:
                for b in ids_B:
                    if len(A_id2author_words[a].intersection(B_id2author_words[b])):
                        candidates.append({'ltable.id':a, 'rtable.id':b} )
                        num_cand += 1
    print(num_cand)
    
    random.shuffle(candidates)
    
    for k,t in enumerate(candidates):
        t['_id'] = k
        
    #cand_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k/candidates.csv'
    cand_file=r'/export/da/mkunjir/LabelDebugger/datasets/citeseer-dblp/candidates.csv'
    fieldnames = ['_id', 'ltable.id', 'rtable.id']
    utils.mycsv.write_list_to_csv(candidates, fieldnames, cand_file)
            
def test():
    table_A, table_B, table_G = read_tables()
    #check_exact_same(table_A, table_B, table_G, 'year')
    #check_num_common_word(table_A, table_B, table_G, 'authors', min_num=1, min_len=5)
    #check_num_common_word(table_A, table_B, table_G, 'title', min_num=2, min_len=8)
    blocking_on_title_min_two_common_words(table_A, table_B, min_len=8)
    
def gen_labeled_data():
    #cand_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k/candidates.csv'
    cand_file=r'/export/da/mkunjir/LabelDebugger/datasets/citeseer-dblp/candidates.csv'
    table_cand = pd.read_csv(cand_file)
    id1_col = table_cand['ltable.id'].values
    id2_col = table_cand['rtable.id'].values
    candidates = set()
    for v1,v2 in zip(id1_col, id2_col):
        candidates.add((v1,v2))
    
    table_A, table_B, table_G = read_tables()
    
    id1_col = table_G['id1'].values
    id2_col = table_G['id2'].values
    matches = set()
    for v1,v2 in zip(id1_col, id2_col):
        matches.add((v1,v2))
        
    pairs = set(matches)
    
    random.seed(0)
    
    A_last_index = len(table_A) - 1
    B_last_index = len(table_B) - 1
    
    num = len(matches)
    count = 0
    
    id_col_A = table_A['id'].values
    id_col_B = table_B['id'].values
    
    while True:
        index_A = random.randint(0, A_last_index)
        index_B = random.randint(0, B_last_index)
            
        p = (id_col_A[index_A], id_col_B[index_B])
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
    
    #labeled_pairs_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k/labeled_data.csv'
    labeled_pairs_file = r'/export/da/mkunjir/LabelDebugger/datasets/citeseer-dblp/labeled_data.csv'
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    utils.mycsv.write_list_to_csv(pairs_list, fieldnames, labeled_pairs_file)
    
    
import v4.dataset_dl
        
def gen_features():
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k.config'
    config_file=r'/export/da/mkunjir/LabelDebugger/config/citeseer.config'
    
    v4.dataset_dl.prepare_features(config_file) 

    
#gen_labeled_data()
gen_features()
