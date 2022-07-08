'''
Created on May 11, 2019

@author: hzhang0418
'''

import os
import pandas as pd

def u300():
    basedir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/umetrics_300/'
    
    input_file = os.path.join(basedir, 'labeled_set_select_matcher_predict_matches.csv')
    output_file = os.path.join(basedir, 'labeled_data.csv')
    
    project_labeled_dataset(input_file, output_file)
    
def u400():
    basedir = r'/scratch/hzhang0418/projects/datasets/labeldebugger/umetrics_400/'
    
    input_file = os.path.join(basedir, 'labeled_set_estimation.csv')
    output_file = os.path.join(basedir, 'labeled_data.csv')
    
    project_labeled_dataset(input_file, output_file)
    
    file_one = os.path.join(basedir, 'umetrics_projected.csv')
    file_two = os.path.join(basedir, 'umetrics_projected_extra.csv')
    file_out = os.path.join(basedir, 'umetrics_projected_all.csv')
    
    combine(file_one, file_two, file_out)

def project_labeled_dataset(input_file, output_file):
    
    df_in = pd.read_csv(input_file)
    df_out = df_in[ ['_id', 'ltable.id', 'rtable.id', 'label']]
    df_out = df_out[ df_out['label']<3] # remove unsure
    df_out['label'].loc[ (df_out['label'] ==2) ] = 0
    
    df_out.to_csv(output_file, index=False)
    
def combine(file_one, file_two, file_out):
    
    df_1 = pd.read_csv(file_one)
    df_2 = pd.read_csv(file_two)
    
    df_2['id'] += 1336
    
    df = pd.concat([df_1, df_2], axis=0)
    
    df.to_csv(file_out, index=False)

import v4.dataset_dl

def gen_features():
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/umetrics_300.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/umetrics_400.config'
    v4.dataset_dl.prepare_features(config_file)   