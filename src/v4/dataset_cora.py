'''
Created on Apr 16, 2019

@author: hzhang0418

'''
import random
import re

import pandas as pd

import utils.mycsv

def test():
    file_with_correct_label = r'/scratch/hzhang0418/projects/datasets/mono/cora/mgcora_corrected.csv'
    df = pd.read_csv(file_with_correct_label)
    print(len(df))
    
    attribute = 'class'
    matches = set()
    corret_label2indices = cluster(df, attribute)
    for _, tmp in corret_label2indices.items():
        for i in tmp:
            for k in tmp:
                matches.add((i,k))
                
    attribute = 'correct_class'
    golden = set()
    corret_label2indices = cluster(df, attribute)
    for _, tmp in corret_label2indices.items():
        for i in tmp:
            for k in tmp:
                golden.add((i,k))
                
    print(len(matches), len(golden))
    print(len(matches.intersection(golden)))
    print(golden.difference(matches))

def cluster(df, attribute):
    corret_label2indices = {}
    for index, row in df.iterrows():
        label = row[attribute]
        if label in corret_label2indices:
            corret_label2indices[label].append(index)
        else:
            corret_label2indices[label] = [index]
    return corret_label2indices

def tokenize(s, min_len=6):
    t = re.sub('[^a-zA-Z]+', ' ', s)
    tmp = t.split()
    return set([t.lower() for t in tmp if len(t)>=min_len])

def tokenize_table(table, attribute, min_len=1):
    index2tokens = {}
    for index, row in table.iterrows():
        index2tokens[index] = tokenize(row[attribute], min_len)
    return index2tokens

def blocking(table_A, table_B, attribute, min_len):
    
    A_index2tokens = tokenize_table(table_A, attribute, min_len)
    B_index2tokens = tokenize_table(table_B, attribute, min_len)
    
    threshold = 0.1
    candidates = []
    
    for index_A, tokens_A in A_index2tokens.items():
        for index_B, tokens_B in B_index2tokens.items():
            # jaccard
            num_overlap = len(tokens_A.intersection(tokens_B))
            num_total = len(tokens_A) + len(tokens_B) - num_overlap
            if num_total==0:
                continue
            score = 1.0*num_overlap/num_total
            if score>=threshold:
                candidates.append( (index_A, index_B))
                
    print(threshold, len(candidates))
    
    return set(candidates)
    
def test2():
    file_with_correct_label = r'/scratch/hzhang0418/projects/datasets/mono/cora/mgcora_corrected.csv'
    df = pd.read_csv(file_with_correct_label)
    print(len(df))
    candidates = blocking(df, df, 'title', 4)
    
    attribute = 'class'
    corret_label2indices = cluster(df, attribute)
    num_matches = 0
    num_in_candidates = 0
    for _, tmp in corret_label2indices.items():
        num_matches += len(tmp)**2
        for i in tmp:
            for k in tmp:
                if (i,k) in candidates:
                    num_in_candidates += 1
    print(num_matches, num_in_candidates)
    
def create_data():
    file_with_correct_label = r'/scratch/hzhang0418/projects/datasets/mono/cora/mgcora_corrected.csv'
    df = pd.read_csv(file_with_correct_label)
    print(len(df))
    candidates = blocking(df, df, 'title', 4)
    
    attribute = 'class'
    corret_label2indices = cluster(df, attribute)
    matches = set()
    num_in_candidates = 0
    for _, tmp in corret_label2indices.items():
        for i in tmp:
            for k in tmp:
                matches.add((i,k))
                if (i,k) in candidates:
                    num_in_candidates += 1
    print(len(matches), num_in_candidates)
    
    pairs = set(matches)
    
    random.seed(0)
    
    A_last_index = len(df) - 1
    B_last_index = len(df) - 1
    
    num = len(matches)
    count = 0
    
    while True:
        index_A = random.randint(0, A_last_index)
        index_B = random.randint(0, B_last_index)
            
        p = (index_A, index_B)
        if p in candidates or p in pairs:
            continue
        
        pairs.add(p)
        count += 1
        
        if count>=num:
            break
    
    pairs_list = []
    for p in pairs:
        label = 1 if p in matches else 0
        pairs_list.append({'ltable.id':p[0]+1, 'rtable.id':p[1]+1, 'label':label})
        
    print(len(pairs_list))
    
    random.shuffle(pairs_list)
    
    for k, p in enumerate(pairs_list):
        p['_id'] = k
    
    labeled_pairs_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new/labeled_data.csv'
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    utils.mycsv.write_list_to_csv(pairs_list, fieldnames, labeled_pairs_file)


def create_golden():
    file_with_correct_label = r'/scratch/hzhang0418/projects/datasets/mono/cora/mgcora_corrected.csv'
    df = pd.read_csv(file_with_correct_label)
    print(len(df))
    
    attribute = 'correct_class'
    corret_label2indices = cluster(df, attribute)
    matches = set()
    for _, tmp in corret_label2indices.items():
        for i in tmp:
            for k in tmp:
                matches.add((i+1,k+1))
    print(len(matches))
    
    labeled_pairs_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new/labeled_data.csv'
    df2 = pd.read_csv(labeled_pairs_file)
    golden_pairs_list = []
    for _, row in df2.iterrows():
        p = (row['ltable.id'], row['rtable.id'])
        label = 0
        if p in matches:
            label = 1
        golden_pairs_list.append({'_id':row['_id'], 'ltable.id':p[0], 'rtable.id':p[1], 'golden':label })
            
    golden_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new/golden.csv'
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'golden']
    utils.mycsv.write_list_to_csv(golden_pairs_list, fieldnames, golden_file)
        
 
import v4.dataset_dl

def gen_features():
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new.config'
    
    v4.dataset_dl.prepare_features(config_file)   
    
        