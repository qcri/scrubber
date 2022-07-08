'''
Created on Feb 13, 2019

@author: hzhang0418
'''

import os

import py_entitymatching as em

import utils.myconfig

def process(config_file):
    # read config
    params = utils.myconfig.read_config(config_file)
    # base dir
    basedir = params['basedir']
    print(basedir)
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
        
        # normalize column
        cmax, cmin = table_H[col].max(), table_H[col].min()
        diff = cmax - cmin
        if diff<0.000001:
            to_be_removed.append(col)
            continue
        
        table_H[col] = (table_H[col] - cmin)/diff
        
        pos = table_H.loc[ table_H['label']==1]
        neg = table_H.loc[ table_H['label']==0]
        
        pavg, navg = pos[col].mean(), neg[col].mean()
        
        #print(col, cmax, cmin, pavg, navg)
    
        # check whether the column is in range
        if pavg<navg or pavg<0.2:
            to_be_removed.append(col)
            
    print('Cols to be removed: ', to_be_removed)    
    
    table_H.drop(columns=to_be_removed, inplace=True)
    
    output_path= os.path.join(basedir, 'feature_vector_truncated.csv')
    table_H.to_csv(output_path, index=False)

def test():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/clothing.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/home.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/electronics.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/tools.config'

    config_file = r'/export/da/mkunjir/LabelDebugger/config/tools.config'    
    process(config_file)
    
   
test() 
    
