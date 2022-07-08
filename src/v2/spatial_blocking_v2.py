'''
Created on Oct 25, 2016

@author: hzhang0418
'''
import numpy as np

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

    feature_vector = np.empty(shape = (nrow, len(feature_names)), dtype = np.float32) # each row is for one pair
    match_pairs = [] # list of match pairs with their row index
    nonmatch_pairs = [] # list of nonmatch pairs with their row index
    labels = {} # map pairs to their labels
    
    for row_index, t in enumerate(tuples):
        id1= t[ feature_map['ltable.id'] ];
        id2 = t[ feature_map['rtable.id'] ];
        label = t[ feature_map['label'] ]
        pair = (id1, id2)
        
        if int(label) == 1:
            match_pairs.append( (pair, row_index) )
        elif int(label) == 0:
            nonmatch_pairs.append( (pair, row_index) )
            
        for i, name in enumerate(feature_names):
            val = t[ feature_map[name] ]
            val = float("{0:.2f}".format(float(val))) # only 2 decimal points?
            feature_vector[row_index, i] = val
            
        labels[pair] = label
        
    print("Num of match:", len(match_pairs))
    print("Num of nonmatch:", len(nonmatch_pairs))
    print("Total:", len(labels))
    
    return feature_vector, nfeatures, match_pairs, nonmatch_pairs, labels 

class SpatialBlockV2():
    '''
    Changes in this version:
    1. use numpy 2-d array to store feature vectors
    2. distribute both match and nonmatch pairs to squares
    3. separate squares that are not consistent (to a given match square) to two types
        (a) inconsistent square (no need to compare with pairs inside this square)
        (b) suspiciously inconsistent square (need to compare with pairs inside this square)
    '''
    
    def __init__(self, feature_vector, nfeatures, match_pairs, nonmatch_pairs, npartitions, min_cons_dim):
        '''
        feature_vector: 2-d numpy array, (pair_index, feature_index)
        nfeatures
        match_pairs: list of tuples (match_pair, pair_index) 
        nonmatch_pairs: list of tuples (nonmatch_pair, pair_index) 
        npartitions: number of partitions for each dimesion
        min_cons_dim: minimum number of dimensions for consistency between a match pair and a nonmatch pair 
        '''
        
        self.feature_vector = feature_vector
        self.nfeatures = nfeatures
        self.match_pairs = match_pairs
        self.nonmatch_pairs = nonmatch_pairs
        self.npartitions = npartitions
        self.min_cons_dim = min_cons_dim
        
        self.delta = 0.0000001
        
        self.square_len = 1/float(npartitions) # length of the side of each square
        
        self.match_squares_map = {} # map each match square to list of match pairs (with pair index)
        self.nonmatch_squares_map = {} # map each nonmatch square to list of nonmatch pairs (with pair index)
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
        
    def map_to_squares(self, pairs, squares_map):
        '''
        distribute pairs into squares
        '''
        for pair in pairs:
            square = self.get_square(self.feature_vector[pair[1], :])
            if square in squares_map:
                squares_map[square].append(pair)
            else: 
                tmp = [pair]
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
            
    
    def compare(self, match_pair, nonmatch_pair):
        '''
        Check whether the given match_pair and nonmatch_pair are consistent
        '''
        
        match_features = self.feature_vector[match_pair[1], :]
        nonmatch_features = self.feature_vector[nonmatch_pair[1], :]
        
        inconsistent = True
        num_cons_dim = 0
        for i in range(self.nfeatures):
            if ( nonmatch_features[i] < match_features[i]):
                num_cons_dim += 1
            if num_cons_dim >= self.min_cons_dim:
                inconsistent = False
                break
            
        return inconsistent
    
    def get_inconsistent(self):
        '''
        Compute the number of inconsistent pairs with each pair
        '''
        # distribute match pairs to match squares
        self.map_to_squares(self.match_pairs, self.match_squares_map)
        # distribute nonmatch pairs to nonmatch squares
        self.map_to_squares(self.nonmatch_pairs, self.nonmatch_squares_map);  
        
        # compute inconsistent and suspiciously inconsistent squares for each match square
        self.compute_incons_and_susp_squares() 
        
        incons_count_map = {} # map pair to the number of inconsistency
        incons_pair_map = {} # map pair to the list of inconsistent pairs (in another class), NOT USED!!!
        
        for p in self.match_pairs:
            incons_count_map[p] = 0
            #incons_pair_map[p] = []
        for p in self.nonmatch_pairs:
            incons_count_map[p] = 0
            #incons_pair_map[p] = []
        
        # handle inconsistent squares
        for ms, ns_list in self.incons_square_map.items() :
            mp_list = self.match_squares_map[ms]
            #mps = [ p[0] for p in mp_list ]
            for ns in ns_list:
                np_list = self.nonmatch_squares_map[ns]
                #nps = [ p[0] for p in np_list ]
                # each pair in ms is inconsistent with each pair in ns
                for mp in mp_list:
                    incons_count_map[mp] += len(np_list)
                    #incons_pair_map[mp].extend( nps )
                    
                for np in np_list:
                    incons_count_map[np] += len(mp_list)
                    #incons_pair_map[np].extend( mps )
               
        # handle suspiciously inconsistent squares
        for ms, ns_list in self.susp_square_map.items() :
            mp_list = self.match_squares_map[ms]
            for ns in ns_list:
                np_list = self.nonmatch_squares_map[ns]
                # check each pair of mp and np
                for match_pair in mp_list:
                    for nonmatch_pair  in np_list:
                        inconsistent = self.compare(match_pair, nonmatch_pair)
                        if inconsistent == True:
                            incons_count_map[match_pair] += 1
                            #incons_pair_map[match_pair].append(nonmatch_pair[0])
                            
                            incons_count_map[nonmatch_pair] += 1
                            #incons_pair_map[nonmatch_pair].append(match_pair[0])  
             
        return incons_count_map, incons_pair_map
        
        
    def get_inconsistent_pairs(self):
        '''
        Compute the number of inconsistent pairs with each pair, and the list of inconsistent pairs with each pair
        '''
        # distribute match pairs to match squares
        self.map_to_squares(self.match_pairs, self.match_squares_map)
        # distribute nonmatch pairs to nonmatch squares
        self.map_to_squares(self.nonmatch_pairs, self.nonmatch_squares_map);  
        
        # compute inconsistent and suspiciously inconsistent squares for each match square
        self.compute_incons_and_susp_squares() 
        
        incons_count_map = {} # map pair to the number of inconsistency
        incons_pair_map = {} # map pair to the list of inconsistent pairs (in another class)
        
        for p in self.match_pairs:
            incons_count_map[p] = 0
            incons_pair_map[p] = []
        for p in self.nonmatch_pairs:
            incons_count_map[p] = 0
            incons_pair_map[p] = []
        
        # handle inconsistent squares
        for ms, ns_list in self.incons_square_map.items() :
            mp_list = self.match_squares_map[ms]
            mps = [ p[0] for p in mp_list ]
            for ns in ns_list:
                np_list = self.nonmatch_squares_map[ns]
                nps = [ p[0] for p in np_list ]
                # each pair in ms is inconsistent with each pair in ns
                for mp in mp_list:
                    incons_count_map[mp] += len(np_list)
                    incons_pair_map[mp].extend( nps )
                    
                for np in np_list:
                    incons_count_map[np] += len(mp_list)
                    incons_pair_map[np].extend( mps )
               
        # handle suspiciously inconsistent squares
        for ms, ns_list in self.susp_square_map.items() :
            mp_list = self.match_squares_map[ms]
            for ns in ns_list:
                np_list = self.nonmatch_squares_map[ns]
                # check each pair of mp and np
                for match_pair in mp_list:
                    for nonmatch_pair  in np_list:
                        inconsistent = self.compare(match_pair, nonmatch_pair)
                        if inconsistent == True:
                            incons_count_map[match_pair] += 1
                            incons_pair_map[match_pair].append(nonmatch_pair[0])
                            
                            incons_count_map[nonmatch_pair] += 1
                            incons_pair_map[nonmatch_pair].append(match_pair[0])  
             
        return incons_count_map, incons_pair_map
    
    '''
    
    '''
    def get_inconsistent_squares(self, match_square, nonmatch_squares):
        
        incons_squares = []
        
        for nonmatch_square in nonmatch_squares:
            count = 0
            for i in range(self.nfeatures):
                if nonmatch_square[i] < match_square[i]:
                    count += 1
                if count >= self.min_cons_dim:
                    break
            if count < self.min_cons_dim:
                incons_squares.append(nonmatch_square)
                
        return incons_squares
    
    def get_inconsistent_origin(self):
        '''
        Compute the number of inconsistent pairs with each pair, and the list of inconsistent pairs with each pair
        '''
        # distribute nonmatch pairs to nonmatch squares
        self.map_to_squares(self.nonmatch_pairs, self.nonmatch_squares_map);  
        
        # compute inconsistent and suspiciously inconsistent squares for each match square
        #self.compute_incons_and_susp_squares() 
        
        incons_count_map = {} # map pair to the number of inconsistency
        incons_pair_map = {} # map pair to the list of inconsistent pairs (in another class)
        
        for p in self.match_pairs:
            incons_count_map[p] = 0
            incons_pair_map[p] = []
        for p in self.nonmatch_pairs:
            incons_count_map[p] = 0
            incons_pair_map[p] = []
        
        all_nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for match_pair in self.match_pairs:
            match_square = self.get_square( self.feature_vector[match_pair[1], :] )
            nonmatch_squares = self.get_inconsistent_squares( match_square, all_nonmatch_squares)
            
            for square in nonmatch_squares:
                np_list = self.nonmatch_squares_map[square]
                for nonmatch_pair in np_list:
                    if self.compare(match_pair, nonmatch_pair) == True:
                        incons_count_map[match_pair] += 1
                        incons_pair_map[match_pair].append(nonmatch_pair[0])
                        
                        incons_count_map[nonmatch_pair] += 1
                        incons_pair_map[nonmatch_pair].append(match_pair[0])  
             
        return incons_count_map, incons_pair_map
        