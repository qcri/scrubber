'''
Created on Oct 27, 2016

@author: hzhang0418
'''

class SpatialBlockV4():
    '''
    
    '''
    
    def __init__(self, feature_vector, nfeatures, match_indices, nonmatch_indices, npartitions, min_cons_dim):
        '''
        feature_vector: (pair_index, features)
        nfeatures: number of features
        match_indices: list of indices for match pairs 
        nonmatch_indices: list of indices for nonmatch pairs 
        npartitions: number of partitions for each dimesion
        min_cons_dim: minimum number of dimensions for consistency between a match pair and a nonmatch pair 
        '''
        
        self.feature_vector = feature_vector
        self.nfeatures = nfeatures
        self.match_indices = match_indices
        self.nonmatch_indices = nonmatch_indices
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
    
    def compare_features(self, match_features, nonmatch_features):
        '''
        Check whether the given match_features and nonmatch_features are consistent
        '''
        
        inconsistent = True
        num_cons_dim = 0
        
        i = 0
        for nf in nonmatch_features:
            if nf < match_features[i]:
                num_cons_dim += 1
                if num_cons_dim >= self.min_cons_dim:
                    inconsistent = False
                    break
            i += 1
            
        return inconsistent
    
    '''
    
    '''
    def get_inconsistent_squares(self, match_square, nonmatch_squares):
        
        incons_squares = []
        
        for nonmatch_square in nonmatch_squares:
            count = 0
            
            i=0
            for nsf in nonmatch_square:
                if nsf < match_square[i]:
                    count += 1
                    if count >= self.min_cons_dim:
                        break
                i += 1
                
            if count < self.min_cons_dim:
                incons_squares.append(nonmatch_square)
                
        return incons_squares
    
    def count_inconsistent(self):
        '''
        Compute the number of inconsistent indices with each index
        '''
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        incons_count_map = {} # map index to the number of inconsistency
        
        for p in self.match_indices:
            incons_count_map[p] = 0
        for p in self.nonmatch_indices:
            incons_count_map[p] = 0
        
        all_nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for match_index in self.match_indices:
            match_features = self.feature_vector[match_index]
            nonmatch_squares = self.get_inconsistent_squares( self.get_square(match_features), all_nonmatch_squares)
            
            for square in nonmatch_squares:
                for nonmatch_index in self.nonmatch_squares_map[square]:
                    if self.compare_features(match_features, self.feature_vector[nonmatch_index]) == True:
                        incons_count_map[match_index] += 1
                        incons_count_map[nonmatch_index] += 1
             
        return incons_count_map
    
    def get_inconsistent_indices(self):
        '''
        Compute the list of inconsistent indices with each index
        '''
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        incons_index_map = {} # map index to the list of inconsistent indices (in another class)
        
        for p in self.match_indices:
            incons_index_map[p] = []
        for p in self.nonmatch_indices:
            incons_index_map[p] = []
        
        all_nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for match_index in self.match_indices:
            match_features = self.feature_vector[match_index]
            nonmatch_squares = self.get_inconsistent_squares( self.get_square(match_features), all_nonmatch_squares)
            
            for square in nonmatch_squares:
                for nonmatch_index in self.nonmatch_squares_map[square]:
                    if self.compare_features(match_features, self.feature_vector[nonmatch_index]) == True:
                        incons_index_map[match_index].append(nonmatch_index)
                        incons_index_map[nonmatch_index].append(match_index)
             
        return incons_index_map
    
    def count_inconsistent_for_dense_dataset(self):
        '''
        Compute the number of inconsistent indices with each index
        '''
        # distribute match indices to match squares
        self.map_to_squares(self.match_indices, self.match_squares_map)
        
        # distribute nonmatch indices to nonmatch squares
        self.map_to_squares(self.nonmatch_indices, self.nonmatch_squares_map);  
        
        incons_count_map = {} # map index to the number of inconsistency
        
        for p in self.match_indices:
            incons_count_map[p] = 0
        for p in self.nonmatch_indices:
            incons_count_map[p] = 0
        
        all_nonmatch_squares = self.nonmatch_squares_map.keys()
        
        for match_square, mis in self.match_squares_map.items():
            nonmatch_squares = self.get_inconsistent_squares( match_square, all_nonmatch_squares)
        
            for match_index in mis:
                match_features = self.feature_vector[match_index]
                for square in nonmatch_squares:
                    for nonmatch_index in self.nonmatch_squares_map[square]:
                        if self.compare_features(match_features, self.feature_vector[nonmatch_index]) == True:
                            incons_count_map[match_index] += 1
                            incons_count_map[nonmatch_index] += 1
             
        return incons_count_map
        