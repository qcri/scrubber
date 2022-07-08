'''
Created on Nov 4, 2016

@author: hzhang0418
'''

import random

import magellan as mg

import active.myAL

debug = False

def check_noise(table_G, table_T):
    index_of_match = []
    index_of_nonmatch = []
    
    for index, row in table_T.iterrows():
        current = table_G.loc[ table_G['_id']==row['_id']].iloc[0]
        if row['golden']==0:
            if current['label']==1:
                index_of_nonmatch.append(index)

        else:
            if current['label']==0:
                index_of_match.append(index)
    
    print("Statistics of noises: ")
    print(len(index_of_match), len(index_of_nonmatch))
    if debug: print(index_of_match)
    if debug: print(index_of_nonmatch)
    
    return [len(index_of_match), len(index_of_nonmatch)]

def select_noise(table_G, table_T, num_noise, match_ratio, seed = 0):
    if num_noise==0:
        return []
    
    index_of_noises = []
    
    num_match_noise = int(num_noise*match_ratio)
    num_nonmatch_noise = int(num_noise - num_match_noise)
    
    # go through ground truth to get index of match and nonmatch pairs
    existed_match_noise = []
    existed_non_match_noise = []
    index_of_match = []
    index_of_nonmatch = []
    
    for index, row in table_T.iterrows():
        current = table_G.loc[ table_G['_id'] == row['_id'] ].iloc[0]
        if row['golden']==0:
            if current['label']==0:
                index_of_nonmatch.append(index)
            else:
                existed_non_match_noise.append(index)
        else:
            if current['label']==1:
                index_of_match.append(index)
            else:
                existed_match_noise.append(index)
             
    #print(num_match_noise, num_nonmatch_noise)
    #print(len(existed_match_noise), len(existed_non_match_noise))
    #print(len(index_of_match), len(index_of_nonmatch))
           
    if debug:
        print(existed_match_noise)
        print(existed_non_match_noise)
        print(index_of_match)
        print(index_of_nonmatch)
    
    # sample index for match and nonmatch noises
    random.seed(seed)
    if num_match_noise-len(existed_match_noise)<len(index_of_match):
        tmp = random.sample(index_of_match, num_match_noise-len(existed_match_noise) )
        tmp.sort()
        index_of_noises.extend(tmp)
    else:
        print("Error:")
    
    if debug: print(index_of_noises)
    
    if num_nonmatch_noise-len(existed_non_match_noise)<=len(index_of_nonmatch):
        tmp = random.sample(index_of_nonmatch, num_nonmatch_noise-len(existed_non_match_noise))
        tmp.sort()
        index_of_noises.extend(tmp)
    else:
        print("Error:")
        
    if debug: print(index_of_noises)
    
    return index_of_noises

def select_noise_using_AL(table_G, table_H, table_T, num_noise, seed = 0):
    if num_noise==0:
        return []
    
    index_of_noises = []
    
    # go through ground truth to get index of match and nonmatch pairs
    existed_match_noise = []
    existed_non_match_noise = []
    index_of_match = []
    index_of_nonmatch = []
    
    for index, row in table_T.iterrows():
        current = table_G.loc[ table_G['_id'] == row['_id'] ].iloc[0]
        if row['golden']==0:
            if current['label']==0:
                index_of_nonmatch.append(index)
            else:
                existed_non_match_noise.append(index)
        else:
            if current['label']==1:
                index_of_match.append(index)
            else:
                existed_match_noise.append(index)
                
    existed_match_noise.sort()
    existed_non_match_noise.sort()
    #index_of_match.sort()
    #index_of_nonmatch.sort()
    #random.shuffle(index_of_match)
    #random.shuffle(index_of_nonmatch)
             
    #print(num_match_noise, num_nonmatch_noise)
    #print(len(existed_match_noise), len(existed_non_match_noise))
    #print(len(index_of_match), len(index_of_nonmatch))
           
    if debug:
        print(existed_match_noise)
        print(existed_non_match_noise)
        print(index_of_match)
        print(index_of_nonmatch)
        
    # params for AL
    
    num_samples = 50 # per category
    batch_size = 2
    
    # for cora large
    #num_samples = 500 # per category
    #batch_size = 20
    max_iters = num_noise
    
    # prepare candset
    candset = table_H.copy()
    candset = candset.drop('label', axis=1)
    row_dropped = []
    row_dropped.extend(existed_match_noise)
    row_dropped.extend(existed_non_match_noise)
    candset = candset.drop(row_dropped)
    
    # prepare gold and seed pairs
    gold_pairs = {} 
    for index in index_of_match:
        row = table_T.iloc[index]
        pair = str(row['ltable.id'])+','+str(row['rtable.id'])
        gold_pairs[pair] = 1
      
    random.seed( seed )
    seed_pairs = {}
    tmp = random.sample(index_of_match, min(num_samples, len(index_of_match)) )
    for index in tmp:
        row = table_T.iloc[index]
        pair = str(row['ltable.id'])+','+str(row['rtable.id'])
        seed_pairs[pair] = 1
        
    tmp = random.sample(index_of_nonmatch, min(num_samples, len(index_of_nonmatch)) )
    for index in tmp:
        row = table_T.iloc[index]
        pair = str(row['ltable.id'])+','+str(row['rtable.id'])
        seed_pairs[pair] = 0
        
    matcher = mg.RFMatcher(name='RandomForest', random_state = 0).clf
    learner = active.myAL.MyAL(matcher, batch_size, max_iters, gold_pairs, seed_pairs)
    noise_pairs = learner.learn(candset, '_id', 'ltable.id', 'rtable.id')
    
    if debug: print( noise_pairs)
    # get index of noise pairs
    for index, row in table_T.iterrows():
        pair = str(row['ltable.id']),str(row['rtable.id'])
        if pair in noise_pairs:
            index_of_noises.append(index)
    
    return index_of_noises[:num_noise-len(existed_match_noise)-len(existed_non_match_noise)]
    
def insert_noise(table_H, table_T, index_of_noises):
    pairs = []
    
    for index in index_of_noises:
        pairs.append( (table_H.iloc[index]['ltable.id'], table_H.iloc[index]['rtable.id']) )
        current_label = table_H.iloc[index][-1]
        new_label = 1
        if current_label == 1:
            new_label = 0
        table_H.set_value(index, 'label', new_label)
    
    return pairs