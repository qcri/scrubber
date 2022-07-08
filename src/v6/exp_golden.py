'''
Created on Mar 25, 2019

@author: hzhang0418
'''
import os

import pandas as pd

import time 
import v6.data_io
import v6.label_debugger
import v6.feature_selection

import utils.myconfig

def run():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new.config'
    
    #mayuresh
    config_file=r'/export/da/mkunjir/LabelDebugger/config/bike.config'

    params = utils.myconfig.read_config(config_file)
    
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])
    gpath = os.path.join(basedir, params['gpath'])

    exclude_attrs = ['', '_id', 'ltable.id', 'rtable.id']
    
    features, labels, feature_labels, pair2index, index2pair = v6.data_io.read_feature_file(hpath, exclude_attrs)
    pair2golden = v6.data_io.read_golden_label_file(gpath)
    
    # label errors
    all_errors = []
    for index, p in index2pair.items():
        if labels[index]!=pair2golden[p]:
            all_errors.append(index)
            
    print(params['dataset_name'])
    
    # config params
    params['fs_alg'] = 'model'
    params['max_list_len'] = 20
    params['detectors'] = 'fpfn'
    params['confidence'] = False
    
    params['num_cores'] = 4
    params['num_folds'] = 5
    
    params['min_con_dim'] = 1
    params['counting_only'] = False 
    
    #index = 324
    #print( labels[index], pair2golden[index2pair[index]] )
    
    selected_features, selected_feature_indexes = v6.feature_selection.select_features(features, labels, params['fs_alg'])
    #print('All featues: ', feature_labels)
    print('Selected features: ', [feature_labels[i] for i in selected_feature_indexes])
    
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    
    all_detected_errors, match_detected_errors, iter_times = debug_labels(debugger, index2pair, pair2golden)
    
    print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    print("Of which false non-match errors: ", len(match_detected_errors))
    print("First iteration (secs): ", iter_times[0])
    print("For other iterations: ")
    print("Min (secs): ", min(iter_times[1:]))
    print("Max (secs): ", max(iter_times[1:]))
    print("Ave (secs): ", sum(iter_times[1:])/len(iter_times[1:]))
    
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    #show all errors
    table_A['id'] = table_A['id'].astype(str)
    table_B['id'] = table_B['id'].astype(str)
    all_error_pairs = []
    for index in all_detected_errors:
        p = index2pair[index]
        label = labels[index]
        left = table_A.loc[ table_A['id'] == str(p[0])]
        right = table_B.loc[ table_B['id'] == str(p[1])]
        tmp = {}
        for col in left:
            tmp['ltable.'+col] = left.iloc[0][col]
        for col in right:
            tmp['rtable.'+col] = right.iloc[0][col]
            tmp['label'] = label
        all_error_pairs.append(tmp)
        
    if len(all_error_pairs)>0:
        df = pd.DataFrame(all_error_pairs)
        output_file = params['dataset_name']+'_'+params['detectors']+'_detected_errors.csv'
        #df.to_csv(output_file, index=False)
        
    '''
    # show missed errors
    missed_error_pairs = []
    for index in all_errors:
        if index not in all_detected_errors:
            p = index2pair[index]
            label = labels[index]
            left = table_A.loc[ table_A['id'] == int(p[0])]
            right = table_B.loc[ table_B['id'] == int(p[1])]
            tmp = {}
            for col in left:
                tmp['ltable.'+col] = left.iloc[0][col]
            for col in right:
                tmp['rtable.'+col] = right.iloc[0][col]
                tmp['label'] = label
            missed_error_pairs.append(tmp)
        
    if len(missed_error_pairs)>0:
        df = pd.DataFrame(missed_error_pairs)
        output_file = params['dataset_name']+'missed_errors.csv'
        df.to_csv(output_file, index=False)
    '''

def debug_labels(debugger, index2pair, pair2golden):
   
    iter_times = [] 
    top_k = 20
    num_iter_without_errors = 0
    all_detected_errors = []
    match_detected_errors = []
    total_num_iters = 0
    start = time.clock()

    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)

        end = time.clock()
        iter_time = end-start
        iter_times.append(iter_time)

        #debugger.set_num_cores(4)

        
        # find their correct labels
        index2correct_label = { index:pair2golden[ index2pair[index] ]  for index in top_suspicious_indices}
        iter_count, num_errors, error_indices, match_error_indices, det_error_poses  = debugger.analyze(index2correct_label)
        print('Iteration: ', iter_count)
        print('Number of suspicious labels found: ', len(top_suspicious_indices))
        print("Number of errors found: ", num_errors)
        #print("Error indices: ", error_indices)
        print("Detector performance: ")
        for n, (count, pos) in enumerate(det_error_poses):
            print("Detector ", n, "found ", count, " errors")
            #print("Positions: ", pos)
            
        all_detected_errors.extend(error_indices)
        match_detected_errors.extend(match_error_indices)
            
        if num_errors==0:
            num_iter_without_errors += 1
        else:
            num_iter_without_errors = 0
            
        if num_iter_without_errors>=3:
            break
        
        start = time.clock()
        debugger.correct_labels(index2correct_label)
        ## mayuresh: explanations
        print('---Explainations for wrong matches and wrong non-matches follow---')
        debugger.explain_errors(False)
        debugger.explain_errors(True)
        
        total_num_iters += 1
        
        if total_num_iters>=40:
            break
        
    return all_detected_errors, match_detected_errors, iter_times
            
run()            
        
