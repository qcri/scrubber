'''
Created on Feb 16, 2019

@author: hzhang0418
'''

import os
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import py_entitymatching as em

import utils.myconfig
import v3.preprocessing as prep

def select(config_file):
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
    
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf = clf.fit(features, labels)
    print(clf.feature_importances_)
    
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(features)
    print(X_new.shape) 
    
    
def correlation(config_file):
    # read config
    params = utils.myconfig.read_config(config_file)
    # base dir
    basedir = params['basedir']
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    hpath = os.path.join(basedir, params['hpath'])
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')    
    table_H = em.read_csv_metadata(hpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    
    to_be_removed = []
    
    for col in table_H:
        if col in tmp:
            continue
        
        print(col, table_H[col].corr(table_H['label']), table_H[col].min(), table_H[col].max())
    
def test():
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_100k.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_home.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'
    
    #select(config_file)
    correlation(config_file)