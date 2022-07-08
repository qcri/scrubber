'''
Created on Oct 27, 2016

@author: hzhang0418
'''

def read_feature_vector(feature_vector_file):
    
    feature_map = {}
    
    f = open(feature_vector_file)
    lines = f.read().splitlines()
    f.close()
    
    k = 0
    while lines[k][0]=='#':
        k = k+1
        continue
    header = lines[k]
    tmp = header.split(',')
    feature_names = []
    for i, attribute in enumerate(tmp):
        if attribute != '_id' and attribute != 'ltable.id' and attribute != 'rtable.id' and attribute != 'label':
            feature_names.append(attribute)
        feature_map[attribute] = i
    
    
    nfeatures = len(feature_names)
    
    tuples = [ line.split(',') for line in lines[k+1:] ]
    
    (nrow, ncol) = len(tuples), len(tmp)
    print(nrow, ncol)

    index2pair = {} # map index to the corresponding pair
    feature_vector = {} # map index to the feature vector of the corresponding pair
    index_of_match_pairs = [] # list of indices of match pairs
    index_of_nonmatch_pairs = [] # list of indices of nonmatch pairs
    index2labels = {} # map index to the label of the corresponding pair
    
    for row_index, t in enumerate(tuples):
        id1= t[ feature_map['ltable.id'] ];
        id2 = t[ feature_map['rtable.id'] ];
        label = t[ feature_map['label'] ]
        pair = (id1, id2)
        
        index2pair[row_index] = pair
        
        if int(label) == 1:
            index_of_match_pairs.append( row_index )
        elif int(label) == 0:
            index_of_nonmatch_pairs.append( row_index )
            
        feature_vector[row_index] = [ float( "{0:.2f}".format( float( t[ feature_map[name] ] ) ) ) for name in feature_names ]
            
        index2labels[row_index] = label
        
    print("Num of match:", len(index_of_match_pairs))
    print("Num of nonmatch:", len(index_of_nonmatch_pairs))
    print("Total:", len(index2labels))
    
    return index2pair, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, index2labels 


def compare_features(match_features, nonmatch_features, min_cons_dim):
        '''
        Check whether the given match_features and nonmatch_features are consistent
        '''
        inconsistent = True
        num_cons_dim = 0
        i = 0
        for nf in nonmatch_features:
            if nf < match_features[i]:
                num_cons_dim += 1
                if num_cons_dim >= min_cons_dim:
                    inconsistent = False
                    break
            i += 1
            
        return inconsistent

def brute_force(feature_vector, nfeatures, match_indices, nonmatch_indices, min_cons_dim):
    
    index2incons = {}
    for index in feature_vector.keys():
        index2incons[index] = []
    
    for mi in match_indices:
        match_features = feature_vector[mi]
        for ni in nonmatch_indices:
            inconsistent = compare_features(match_features, feature_vector[ni], min_cons_dim)
            if inconsistent == True:
                    index2incons[mi].append(ni)
                    index2incons[ni].append(mi)
    
    return index2incons