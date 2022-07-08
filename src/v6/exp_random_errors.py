'''
Created on May 24, 2019

@author: hzhang0418
'''

import os
import sys
import random

import pandas as pd

import v6.data_io
import v6.label_debugger
import v6.feature_selection

import utils.myconfig

def run(detector='both', confusion=False, top_k=20):
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/fodors_zagats.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new.config'
    
    #mayuresh
    config_file=r'/export/da/mkunjir/LabelDebugger/config/cora.config'
    
    params = utils.myconfig.read_config(config_file)
    
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])
    gpath = os.path.join(basedir, 'golden.csv')

    exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
    
    features, labels, pair2index, index2pair = v6.data_io.read_feature_file(hpath, exclude_attrs)
    pair2golden = v6.data_io.read_golden_label_file(gpath) if os.path.exists(gpath) else {p:labels[i] for p, i in pair2index.items()} ## mayuresh: if no golden labels, then only new errors will be injected 

    params['fs_alg'] = 'xgboost'
    params['max_list_len'] = 40
    params['detectors'] = detector
    params['confusion'] = confusion

    params['num_cores'] = 4 
    params['num_folds'] = 5

    params['min_con_dim'] = 1
    params['counting_only'] = False 
    #debugger = v6.label_debugger.LabelDebugger(features, labels, params)
    
    # label errors
    all_errors = []
    false_nonmatch_errors = 0
    false_nonmatch_indices = []
    for index, p in index2pair.items():
        if labels[index]!=pair2golden[p]:
            #print('Error found at: ', index, ' the pair is: ', p)
            all_errors.append(index)
            if labels[index]==0:
               false_nonmatch_errors += 1
               false_nonmatch_indices.append(index)
    
    '''
    # TEMP: mayuresh: eliminate errors from labeled data    
    #mpath = os.path.join(basedir, 'labeled_data.csv')
    new_hpath = os.path.join(basedir, 'feature_vector_newerrors.csv')
    new_gpath = os.path.join(basedir, 'golden_newerrors.csv')
    fv_df = pd.read_csv(hpath)
    g_df = pd.read_csv(gpath)
    #fv_df.drop(fv_df.index[all_errors], inplace=True)
    #fv_df.to_csv(new_hpath, index=False)
    #exit()
    '''
        
    # randomly insert errors
    seed = 0 #random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)
    
    perc = 0.1 #rng.randint(5, 15)/100.0
    print("Error rate:", perc)
    
    num_err = int(len(labels)*perc - len(all_errors)) # accounting for existing errors
    #num_err = min(800, num_err) # 800 is the maximum errors we can check in the given budget

    while num_err < 0: ## correct some existing errors to make them equal to the required number
      i = 0#rng.randint(0, len(all_errors)-1)
      index = all_errors[i]
      labels[index] = 1 - labels[index]
      if labels[index]==1:
         false_nonmatch_errors -= 1
         false_nonmatch_indices.remove(index)
      all_errors.remove(index)
      num_err += 1

    error_indices = set(all_errors)
    new_error_indices = set()
    for _ in range(num_err*10):
        index = rng.randint(0, len(labels)-1)
        if index in error_indices or index in new_error_indices:
            continue
        new_error_indices.add(index)
        if labels[index]==1:
            false_nonmatch_errors += 1
            false_nonmatch_indices.append(index)
        labels[index] = 0 if labels[index]==1 else 1
        if len(new_error_indices)>=num_err:
            break

    '''
    ## mayuresh: update feature vector with new errors and new golden labels
    print('New errors inserted: ', len(new_error_indices))
    #g_df = g_df.iloc[0:0] # dropping current golden labels
    for cnt, index in enumerate(new_error_indices):
      pair = index2pair[index]
      fv_df.at[index, 'label'] = labels[index]
      #print('Changed fv location: ', index)
      g_df.at[index, 'golden'] = 1-labels[index] # = g_df.append({'_id':cnt, 'ltable.id':pair[0], 'rtable.id':pair[1], 'golden':1-labels[index]}, ignore_index=True)
    g_df.drop(fv_df.index[all_errors], inplace=True) #delete old errors
    g_df.to_csv(new_gpath, index=False)
    fv_df.drop(fv_df.index[all_errors], inplace=True) #delete old errors
    fv_df.to_csv(new_hpath, index=False)
    exit()
    '''

    print("Total number of errors: ", len(error_indices) + len(new_error_indices))
    print("Of which total false nonmatches: ", false_nonmatch_errors)
    all_errors = list(error_indices)
    all_errors.extend(list(new_error_indices))
    false_match_errors = len(all_errors) - false_nonmatch_errors

    print(params['dataset_name'])
    
    #index = 324
    #print( labels[index], pair2golden[index2pair[index]] )
   
    ## mayuresh: read selected features if available in file or else store them 
    #del params['spath']
    if 'spath' in params:
       selected_features_path = os.path.join(basedir, params['spath'])
       selected_features = pd.read_csv(selected_features_path).to_numpy()
    else:
       selected_features = v6.feature_selection.select_features(features, labels, params['fs_alg'])
       selected_features_path = os.path.join(basedir, 'selected_features.csv')
       pd.DataFrame(selected_features).to_csv(selected_features_path, index=False)
    print('Selected features of dim: ', selected_features.shape, ' from the original features: ', features.shape)


    params['fs_alg'] = 'none' # disabling feature selection now that we are done selecting   
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    
    all_detected_errors, detected_false_nonmatches = debug_labels(debugger, index2pair, pair2golden, top_k)
    
    print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    print("Of which the number of false nonmatches: ", len(detected_false_nonmatches))
    print("False match detection recall: ", 100.0 * (len(all_detected_errors) - len(detected_false_nonmatches)) / false_match_errors)
    print("False non-match detection recall: ", 100.0 * len(detected_false_nonmatches) / false_nonmatch_errors)
    print("False match detection prec: ", 100.0 * (len(all_detected_errors) - len(detected_false_nonmatches)) / (len(debugger.verified_indices) * false_match_errors / len(all_errors)))
    #print("False nonmatch detection prec: ", 100.0 * (len(detected_false_nonmatches)) / (len(debugger.verified_indices) * false_nonmatch_errors / len(all_errors)))

    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    '''
    #show all errors
    table_A['id'] = table_A['id'].astype(str)
    table_B['id'] = table_B['id'].astype(str)
    all_error_pairs = []
    for index in all_errors:
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
        output_file = params['dataset_name']+'_all_errors.csv'
        df.to_csv(output_file, index=False)
        
    
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

def debug_labels(debugger, index2pair, pair2golden, top_k=20):
    
    num_iter_without_errors = 0
    all_detected_errors = set() 
    all_detected_match_errors = set() # labels mislabeled as matches
    total_num_iters = 0
    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)
        
        # find their correct labels
        index2correct_label = { index:pair2golden[ index2pair[index] ]  for index in top_suspicious_indices}
        iter_count, num_errors, error_indices, error_indices_matches, det_error_poses  = debugger.analyze(index2correct_label)
        #print('Iteration: ', iter_count)
        #print("Number of errors found: ", num_errors)
        #print("Error indices: ", error_indices)
        #print("Detector performance: ")
        #for n, (count, pos) in enumerate(det_error_poses):
        #    print("Detector ", n, "found ", count, " errors")
        #    print("Positions: ", pos)
            
        all_detected_errors.update(error_indices)
        all_detected_match_errors.update(error_indices_matches)
            
        if num_errors==0:
            num_iter_without_errors += 1
        else:
            num_iter_without_errors = 0
            
        if num_iter_without_errors>=3:
            break
        
        debugger.correct_labels(index2correct_label)
        
        total_num_iters += 1
        
        if total_num_iters>=5000:
            break
        
    return all_detected_errors, all_detected_match_errors


run('fpfn', False, 20)
#run('mono', False, 20)
#run('both', False, 20)
#run('both', False, 10)
#run('both', False, 15)
#run('cleanlab', False)
run('cleanlab', True)
