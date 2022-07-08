'''
Created on Jan 31, 2019

@author: hzhang0418
'''

'''
Process datasets from Han
'''

import os

import py_entitymatching as em

import v3.feature_generation as fg

import utils.mycsv
import utils.myconfig

def process(basedir):
    
    pass

def prepare_labeled_data(basedir):
    
    train_file = os.path.join(basedir, 'train.csv')
    valid_file = os.path.join(basedir, 'valid.csv')
    test_file = os.path.join(basedir, 'test.csv')
    
    labeled_data = utils.mycsv.read_csv_as_list(train_file)    
    labeled_data.extend( utils.mycsv.read_csv_as_list(valid_file) )
    labeled_data.extend( utils.mycsv.read_csv_as_list(test_file) )
    
    print(len(labeled_data))
    print(labeled_data[0])
    
    for i, data in enumerate(labeled_data, 1):
        data['_id'] = i
        data['label'] = data['gold']
        data['ltable.id'] = data['ltable_id']
        data['rtable.id'] = data['rtable_id']
    
    fieldnames = ['_id', 'ltable.id', 'rtable.id', 'label']
    
    output_file = 'labeled_data.csv'
    utils.mycsv.write_list_to_csv(labeled_data, fieldnames, os.path.join(basedir, output_file))
    
def prepare_features(config_file):
    params = utils.myconfig.read_config(config_file)
    fg.generate_features(params)
    
    
def test():
    basedir = r'/scratch/hzhang0418/Datasets from Han/Clothing'
    basedir = r'/scratch/hzhang0418/Datasets from Han/Home'
    basedir = r'/scratch/hzhang0418/Datasets from Han/Electronics'
    basedir = r'/scratch/hzhang0418/Datasets from Han/Tools'
    prepare_labeled_data(basedir)
    
    
def test2():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/clothing.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/home.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'
    
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_home.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_electronics.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    prepare_features(config_file)
    
    
def find_attributes_to_drop(table_A, table_B, perc = 0.1):
    
    attr_to_drop = set()
    
    size = len(table_A)
    for col in table_A:
        num_null = table_A[col].isnull().sum()
        if num_null>=size*perc:
            attr_to_drop.add(col)
            
    size = len(table_B)
    for col in table_B:
        num_null = table_B[col].isnull().sum()
        if num_null>=size*perc:
            attr_to_drop.add(col)        
    
    return list(attr_to_drop)
    
def test3():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/clothing.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/home.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')  
    

    attr_to_drop = find_attributes_to_drop(table_A, table_B, 0.05)
    print(attr_to_drop)
    
    table_A.drop(columns=attr_to_drop, inplace=True)
    table_B.drop(columns=attr_to_drop, inplace=True)

    apath_new = os.path.join(basedir, r'trunc_'+params['apath'])
    bpath_new = os.path.join(basedir, r'trunc_'+params['bpath'])
    
    table_A.to_csv(apath_new, index=False)
    table_B.to_csv(bpath_new, index=False)
    
def test4():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_home.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    # read config
    params = utils.myconfig.read_config(config_file)
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id') 
    
    table_A_missing_ids = set()
    for col in table_A:
        ids = set(table_A[table_A[col].isnull()]['id'].values)
        table_A_missing_ids.update(ids)
        
    table_B_missing_ids = set()
    for col in table_B:
        ids = set(table_B[table_B[col].isnull()]['id'].values)
        table_B_missing_ids.update(ids)
        
    print(len(table_A_missing_ids))
    print(len(table_B_missing_ids))
    
    tmp_A = table_A[~table_A['id'].isin(table_A_missing_ids)]
    tmp_B = table_B[~table_B['id'].isin(table_B_missing_ids)]
    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    tmp_G = table_G[ ~table_G['ltable.id'].isin(table_A_missing_ids)]
    tmp_G = tmp_G[~tmp_G['rtable.id'].isin(table_B_missing_ids)]
    print(len(table_G), len(tmp_G))
    
    tmp_A.to_csv(apath, index=False)
    tmp_B.to_csv(bpath, index=False)
    tmp_G.to_csv(gpath, index=False)
        
    
def merge(basedir):
    train_file = os.path.join(basedir, '_train_feat_vecs.csv')
    valid_file = os.path.join(basedir, '_valid_feat_vecs.csv')
    test_file = os.path.join(basedir, '_test_feat_vecs.csv')
    
    features = utils.mycsv.read_csv_as_list(train_file)    
    features.extend( utils.mycsv.read_csv_as_list(valid_file) )
    features.extend( utils.mycsv.read_csv_as_list(test_file) )
    
    for k, f in enumerate(features):
        f['_id'] = k
        f['ltable.id'] = f['ltable_id']
        f['rtable.id'] = f['rtable_id']
    
    fieldnames = list(features[0].keys())
    fieldnames.remove('ltable_id')
    fieldnames.remove('rtable_id')
    
    output_file = 'feature_vector_tfidf.csv'
    utils.mycsv.write_list_to_csv(features, fieldnames, os.path.join(basedir, output_file))
    
def test5():
    basedir = r'/scratch/hzhang0418/temp_feat/Clothing'
    basedir = r'/scratch/hzhang0418/temp_feat/Home'
    basedir = r'/scratch/hzhang0418/temp_feat/Electronics'
    #basedir = r'/scratch/hzhang0418/temp_feat/Tools'
    
    merge(basedir)

    
    
    