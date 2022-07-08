'''
Created on Apr 27, 2019

@author: hzhang0418
'''

import os

import pandas as pd

import utils.myconfig

def run():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    params = utils.myconfig.read_config(config_file)
    
    basedir = params['basedir']
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    hpath = os.path.join(basedir, params['hpath'])
    
    df = pd.read_csv(hpath)
    
    filtered = df.loc[((df['label'] == 1) & (df['product_type_product_type_jac_qgm_3_qgm_3'] > 0.900) & \
                (df['product_name_product_name_jac_qgm_3_qgm_3'] < 0.70))]
    
    print(len(filtered))
    
    sampled = filtered.sample(n=200, random_state=0)
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    all_pairs_with_label = []
    for _, row in sampled.iterrows():
        label = row['label']
        left = table_A.loc[ table_A['id'] == row['ltable.id']]
        right = table_B.loc[ table_B['id'] == row['rtable.id']]
        tmp = {}
        for col in left:
            tmp['ltable.'+col] = left.iloc[0][col]
        for col in right:
            tmp['rtable.'+col] = right.iloc[0][col]
        tmp['label'] = label
        all_pairs_with_label.append(tmp)
    df = pd.DataFrame(all_pairs_with_label)

    output_file = 'trunc_clothing_0503_rule.csv'
    df.to_csv(output_file, index=False)
       
    
def rule_negative():
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    params = utils.myconfig.read_config(config_file)
    
    basedir = params['basedir']
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    hpath = os.path.join(basedir, params['hpath'])
    
    df = pd.read_csv(hpath)
    
    filtered = df.loc[((df['label'] == 0) & (df['product_type_product_type_jac_qgm_3_qgm_3'] < 0.300) & \
                (df['product_name_product_name_jac_qgm_3_qgm_3'] >= 0.9))]
    
    print(len(filtered))
    
    
    sampled = filtered.sample(n=100, random_state=0)
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    all_pairs_with_label = []
    for _, row in sampled.iterrows():
        label = row['label']
        left = table_A.loc[ table_A['id'] == row['ltable.id']]
        right = table_B.loc[ table_B['id'] == row['rtable.id']]
        tmp = {}
        for col in left:
            tmp['ltable.'+col] = left.iloc[0][col]
        for col in right:
            tmp['rtable.'+col] = right.iloc[0][col]
        tmp['label'] = label
        all_pairs_with_label.append(tmp)
    df = pd.DataFrame(all_pairs_with_label)

    output_file = 'trunc_clothing_0529_rule_negative.csv'
    df.to_csv(output_file, index=False)
    