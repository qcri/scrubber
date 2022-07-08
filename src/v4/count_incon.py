'''
Created on Feb 18, 2019

@author: hzhang0418
'''

def count(features, labels, suspicious_indices, golden_indices, min_con_dim=1):
    print('Count inconsistencies...')
    
    golden_pos_indices = [ index for index in golden_indices if labels[index]==1]
    golden_neg_indices = [ index for index in golden_indices if labels[index]==0]
    
    index2count = {}
    
    for s_index in suspicious_indices:
        label = labels[s_index]
        fs = features[s_index]
        num = 0
        if label==1:
            # compare with golden neg
            for g_index in golden_neg_indices:
                if compare_features(fs, features[g_index], min_con_dim)==True:
                    num += 1
        else:
            # compare with golden pos
            for g_index in golden_pos_indices:
                if compare_features(features[g_index], fs, min_con_dim)==True:
                    num += 1
        if num>0:
            index2count[s_index] = num
        
    return index2count

        
def compare_features(match_features, nonmatch_features, min_con_dim):
    '''
    Check whether the given match_features and nonmatch_features are consistent
    '''
    is_incon = True
    num_cons_dim = 0
    i = 0
    for nf in nonmatch_features:
        if nf < match_features[i]:
            num_cons_dim += 1
            if num_cons_dim >= min_con_dim:
                is_incon = False
                break
        i += 1
        
    return is_incon         