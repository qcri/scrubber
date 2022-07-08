'''
Created on Feb 20, 2017

@author: hzhang0418

Generating features using Magellan (py_entitymatching)

'''

import os
import py_entitymatching as em
import v3.dataset as ds

def generate_features(params):
    
    # dataset
    dataset_name = params['dataset_name']
    
    # base dir
    basedir = params['basedir']
    
    # path for table A, B, G, H
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    gpath = os.path.join(basedir, params['gpath'])
    hpath = os.path.join(basedir, params['hpath'])
    
    table_A, table_B, table_features = ds.retrieve(dataset_name, apath, bpath, gpath)
    
    table_G = em.read_csv_metadata(gpath, key='_id', ltable = table_A, rtable = table_B, fk_ltable='ltable.id', fk_rtable='rtable.id')
    
    ds.compute_feature_vector(table_G, table_features, hpath)
