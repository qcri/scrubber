'''
Created on Oct 27, 2016

@author: hzhang0418

in this version, list is used for feature vector 

'''

import itertools

debug = False
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
    
    if debug: print(feature_names)
    
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

class SpatialBlockV3():
    '''
    Changes in this version:
    1. only indices are needed
    '''
    
    def __init__(self, feature_vector, nfeatures, index_of_match_pairs, index_of_nonmatch_pairs, npartitions, min_cons_dim):
        '''
        feature_vector: 2-d numpy array, (pair_index, feature_index)
        nfeatures: number of features
        match_indices: list of tuples (match_pair, pair_index) 
        nonmatch_indices: list of tuples (nonmatch_pair, pair_index) 
        npartitions: number of partitions for each dimesion
        min_cons_dim: minimum number of dimensions for consistency between a match pair and a nonmatch pair 
        '''
        
        self.feature_vector = feature_vector
        self.nfeatures = nfeatures
        self.match_indices = index_of_match_pairs
        self.nonmatch_indices = index_of_nonmatch_pairs
        self.npartitions = npartitions
        self.min_cons_dim = min_cons_dim
        
        self.delta = 0.0000001
        
        self.square_len = 1/float(npartitions) # length of the side of each square
        
        self.match_squares_map = {} # map each match square to list of match indices
        self.nonmatch_squares_map = {} # map each nonmatch square to list of nonmatch indices
        self.incons_square_map = {} # map each match square to list of nonmatch squares that are inconsistent with it
        self.susp_square_map = {} # map each match square to list of monmatch squares that are suspiciously inconsistent with it
        
        
    def get_square(self, features):
        '''
        compute the start of the square in each dimension
        '''
        square = []
        for value in features:
            #compute the start of the square 
            start_of_square = ((value - self.delta)/float(self.square_len)) * self.square_len
            square.append(start_of_square)
        return tuple(square)
        
    def map_to_squares(self, indices, squares_map):
        '''
        distribute indices into squares
        '''
        for index in indices:
            square = self.get_square(self.feature_vector[index])
            if square in squares_map:
                squares_map[square].append(index)
            else: 
                tmp = [index]
                squares_map[square] = tmp
    
    
    def compute_incons_and_susp_squares(self):
        '''
        for each match square, find the list of nonmatch squares that are inconsistent or suspiciously inconsistent with it
        
        Let k=min_cons_dim, then  a nonmatch square is:
        1. consistent if at least k-dim worse than the match square
        2. inconsistent if at least (n-k+1)-dim better than the match square
        3. suspiciously inconsistent if not in the above two cases 
        '''
        match_squares = self.match_squares_map.keys()
        nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for ms in match_squares:
            self.incons_square_map[ms] = []
            self.susp_square_map[ms] = []
            
            for ns in nonmatch_squares:
                # compute the number of dimesions 
                nworse = 0
                nbetter = 0
                for i in range(self.nfeatures):
                    if ns[i]<ms[i]:
                        nworse += 1
                    elif ns[i]>ms[i]:
                        nbetter += 1
                
                if nbetter >= self.nfeatures - self.min_cons_dim + 1:
                    self.incons_square_map[ms].append(ns)
                elif nworse < self.min_cons_dim:
                    self.susp_square_map[ms].append(ns)
                # otherwise they are consistent
            
    
    def compare(self, match_index, nonmatch_index):
        '''
        Check whether the given match_index and nonmatch_index are consistent
        '''
        
        match_features = self.feature_vector[match_index]
        nonmatch_features = self.feature_vector[nonmatch_index]
        
        inconsistent = True
        num_cons_dim = 0
        for i in range(self.nfeatures):
            if ( nonmatch_features[i] < match_features[i]):
                num_cons_dim += 1
            if num_cons_dim >= self.min_cons_dim:
                inconsistent = False
                break
            
        return inconsistent
    
    def compare_features(self, match_features, nonmatch_features):
        '''
        Check whether the given match_features and nonmatch_features are consistent
        '''
        
        inconsistent = True
        num_cons_dim = 0
        for nf, mf in itertools.izip(nonmatch_features, match_features):
            if nf < mf:
                num_cons_dim += 1
            if num_cons_dim >= self.min_cons_dim:
                inconsistent = False
                break
            
        return inconsistent
    
    def get_inconsistent(self):
        '''
        Compute the number of inconsistent indices with each index
        '''
        # distribute match indices to match squares
        self.map_to_squares(self.match_indices, self.match_squares_map)
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        # compute inconsistent and suspiciously inconsistent squares for each match square
        self.compute_incons_and_susp_squares() 
        
        incons_count_map = {} # map index to the number of inconsistency
        incons_index_map = {} # map index to the list of inconsistent indices (in another class), NOT USED!!!
        
        for p in self.match_indices:
            incons_count_map[p] = 0

        for p in self.nonmatch_indices:
            incons_count_map[p] = 0
        
        # handle inconsistent squares
        for ms, ns_list in self.incons_square_map.items() :
            mi_list = self.match_squares_map[ms]
            for ns in ns_list:
                ni_list = self.nonmatch_squares_map[ns]
                # each index in ms is inconsistent with each index in ns
                for mi in mi_list:
                    incons_count_map[mi] += len(ni_list)
                    
                for ni in ni_list:
                    incons_count_map[ni] += len(mi_list)
               
        # handle suspiciously inconsistent squares
        for ms, ns_list in self.susp_square_map.items() :
            mi_list = self.match_squares_map[ms]
            for ns in ns_list:
                ni_list = self.nonmatch_squares_map[ns]
                # check each pair of mi and ni
                for match_index in mi_list:
                    for nonmatch_index  in ni_list:
                        inconsistent = self.compare(match_index, nonmatch_index)
                        if inconsistent == True:
                            incons_count_map[match_index] += 1
                            incons_count_map[nonmatch_index] += 1 
             
        return incons_count_map, incons_index_map
        
        
    def get_inconsistent_indices(self):
        '''
        Compute the number of inconsistent indices with each index, and the list of inconsistent indices with each index
        '''
        # distribute match indices to match squares
        self.map_to_squares(self.match_indices, self.match_squares_map)
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        # compute inconsistent and suspiciously inconsistent squares for each match square
        self.compute_incons_and_susp_squares() 
        
        incons_count_map = {} # map index to the number of inconsistency
        incons_index_map = {} # map index to the list of inconsistent indices (in another class), NOT USED!!!
        
        for p in self.match_indices:
            incons_count_map[p] = 0
            incons_index_map[p] = []

        for p in self.nonmatch_indices:
            incons_count_map[p] = 0
            incons_index_map[p] = []
        
        # handle inconsistent squares
        for ms, ns_list in self.incons_square_map.items() :
            mi_list = self.match_squares_map[ms]
            for ns in ns_list:
                ni_list = self.nonmatch_squares_map[ns]
                # each index in ms is inconsistent with each index in ns
                for mi in mi_list:
                    incons_count_map[mi] += len(ni_list)
                    incons_index_map[mi].extend( ni_list )
                    
                for ni in ni_list:
                    incons_count_map[ni] += len(mi_list)
                    incons_index_map[ni].extend(mi_list)
               
        # handle suspiciously inconsistent squares
        for ms, ns_list in self.susp_square_map.items() :
            mi_list = self.match_squares_map[ms]
            for ns in ns_list:
                ni_list = self.nonmatch_squares_map[ns]
                # check each index of mi and ni
                for match_index in mi_list:
                    for nonmatch_index  in ni_list:
                        inconsistent = self.compare(match_index, nonmatch_index)
                        if inconsistent == True:
                            incons_count_map[match_index] += 1
                            incons_index_map[match_index].append(nonmatch_index)
                            
                            incons_count_map[nonmatch_index] += 1 
                            incons_index_map[nonmatch_index].append(match_index)
             
        return incons_count_map, incons_index_map
    
    '''
    
    '''
    def get_inconsistent_squares(self, match_square, nonmatch_squares):
        
        incons_squares = []
        
        for nonmatch_square in nonmatch_squares:
            count = 0
            for nsf, msf in itertools.izip(nonmatch_square, match_square):
                if nsf < msf:
                    count += 1
                if count >= self.min_cons_dim:
                    break
            if count < self.min_cons_dim:
                incons_squares.append(nonmatch_square)
                
        return incons_squares
    
    def get_inconsistent_origin(self):
        '''
        Compute the number of inconsistent indices with each index, and the list of inconsistent indices with each index
        '''
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        incons_count_map = {} # map index to the number of inconsistency
        incons_index_map = {} # map index to the list of inconsistent indices (in another class)
        
        for p in self.match_indices:
            incons_count_map[p] = 0
            incons_index_map[p] = []
        for p in self.nonmatch_indices:
            incons_count_map[p] = 0
            incons_index_map[p] = []
        
        all_nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for match_index in self.match_indices:
            match_features = self.feature_vector[match_index]
            nonmatch_squares = self.get_inconsistent_squares( self.get_square(match_features), all_nonmatch_squares)
            
            for square in nonmatch_squares:
                for nonmatch_index in self.nonmatch_squares_map[square]:
                    if self.compare(match_features, self.feature_vector[nonmatch_index]) == True:
                        
                        incons_count_map[match_index] += 1
                        #incons_index_map[match_index].append(nonmatch_index)
                        
                        incons_count_map[nonmatch_index] += 1
                        #incons_index_map[nonmatch_index].append(match_index)
             
        return incons_count_map, incons_index_map