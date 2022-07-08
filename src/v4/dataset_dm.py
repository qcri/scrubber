'''
Created on Apr 10, 2019

@author: hzhang0418
'''
import os
import random

import pandas as pd
import utils.mycsv

def walmart_amazon():

    basedir = r'/scratch/hzhang0418/structured_data_blocking/Walmart-Amazon'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/Walmart-Amazon'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total_try.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)
    
def abt_buy():

    basedir = r'/scratch/hzhang0418/structured_data_blocking/Abt-Buy'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/Abt-Buy'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)
    
def dblp_acm():

    basedir = r'/scratch/hzhang0418/structured_data_blocking/DBLP-ACM'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/DBLP-ACM'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)
    
def dblp_googlescholar():

    basedir = r'/scratch/hzhang0418/structured_data_blocking/DBLP-GoogleScholar'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/DBLP-GoogleScholar'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)
    
def amazon_google():

    basedir = r'/scratch/hzhang0418/structured_data_blocking/Amazon-Google'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/Amazon-Google'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)
    
def fodors_zagats():
    basedir = r'/scratch/hzhang0418/structured_data_blocking/Fodors-Zagats'
    destdir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/Fodors-Zagats'
    
    table_A_file = os.path.join(basedir, 'tableA.csv')
    table_B_file = os.path.join(basedir, 'tableB.csv')
    candidates_file = os.path.join(basedir, 'total.csv')
    matches_file = os.path.join(basedir, 'gold.csv')
    labeled_pairs_file = os.path.join(destdir, 'labeled_data.csv')
    
    process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file)

def process(table_A_file, table_B_file, candidates_file, matches_file, labeled_pairs_file):
    
    #table_A_as_list = utils.mycsv.read_csv_as_list(table_A_file)
    #table_B_as_list = utils.mycsv.read_csv_as_list(table_B_file)
    table_A_df = pd.read_csv(table_A_file)
    table_B_df = pd.read_csv(table_B_file)
    candidates_list = utils.mycsv.read_csv_as_list(candidates_file)
    matches_list = utils.mycsv.read_csv_as_list(matches_file)
    
    #table_A_ids = [t['id'] for t in table_A_as_list ]
    #table_B_ids = [t['id'] for t in table_B_as_list ]
    table_A_ids = [str(t) for t in table_A_df['id'].values ] 
    table_B_ids = [str(t) for t in table_B_df['id'].values ] 
    
    candidate_pairs = set([ (t['ltable_id'], t['rtable_id']) for t in candidates_list ])
    matches_pairs = set([ (t['tableA_id'], t['tableB_id']) for t in matches_list ])
    
    random.seed(0)
    
    count = 0
    num = len(matches_pairs)
    
    pairs = set(matches_pairs)
    
    A_last_index = len(table_A_ids) - 1
    B_last_index = len(table_B_ids) - 1
    
    while True:
        index_A = random.randint(0, A_last_index)
        index_B = random.randint(0, B_last_index)
        
        p = (table_A_ids[index_A], table_B_ids[index_B])
        if p in candidate_pairs or p in pairs:
            continue
        
        pairs.add(p)
        count += 1
        
        if count>=num:
            break
    
    pairs_list = []
    for p in pairs:
        label = 1 if p in matches_pairs else 0
        pairs_list.append({'ltable.id':p[0], 'rtable.id':p[1], 'label':label})
    
    random.shuffle(pairs_list)
    
    for k, p in enumerate(pairs_list):
        p['_id'] = k
    
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    
    utils.mycsv.write_list_to_csv(pairs_list, fieldnames, labeled_pairs_file)
    
    return pairs_list    

def test():
    #walmart_amazon()
    #abt_buy()
    #dblp_acm()
    #dblp_googlescholar()
    #amazon_google()
    fodors_zagats()
    
import v4.dataset_dl    

def test2():
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/walmart_amazon.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/abt_buy.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/dblp_acm.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/dblp_googlescholar.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/amazon_google.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/fodors_zagats.config'
    
    v4.dataset_dl.prepare_features(config_file)