'''
Created on Nov 17, 2018

@author: hzhang0418
'''

import os
import py_entitymatching as em

import utils.myconfig

def info(config_file):
    
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
    #hpath = os.path.join(basedir, params['hpath'])
    
    if dataset_name.startswith('citations_'):
        tpath=gpath
    else:
        tpath = os.path.join(basedir, 'golden.csv')
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    table_T = em.read_csv_metadata(tpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    if 'golden' not in table_T:
        table_T = table_T.rename(columns={'label':'golden'})
    
    # data set info
    print(dataset_name)
    
    print("Table A size:", len(table_A))
    print("Table B size:", len(table_B))
    
    print("Number of pairs:", len(table_G))
    
    # number of match and nonmatch pairs
    print("Number of matches:", table_G[ table_G['label']==1 ].count()['label'])
    print("Number of nonmatches:", table_G[ table_G['label']==0 ].count()['label'])
    
    # number of noises
    err_match = 0
    err_nonmatch = 0
    for index, row in table_T.iterrows():
        current = table_G.loc[ table_G['_id']==row['_id']].iloc[0]
        if row['golden']==0:
            if current['label']==1:
                err_match += 1
    
        else:
            if current['label']==0:
                err_nonmatch += 1
    
    print("Number of label errors:", err_match+err_nonmatch)
    print("Number of errors in matched pairs:", err_match)
    print("Number of errors in nonmatched pairs:", err_nonmatch)
    
def test():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    info(config_file)    

    
    