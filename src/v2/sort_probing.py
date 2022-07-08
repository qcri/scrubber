'''
Created on Nov 10, 2016

@author: hzhang0418
'''

'''
Only work for 1-dimension consistency

For each feature, cluster matches and nonmathes according to the feature value

For each feature, for each cluster of matches, get the list of nonmatches clusters that are consistent with the matches in the cluster at this feature

for each match, union the lists of nonmatches that are consistent with it at each feature,
then the compliment is the list of nonmatches that are inconsistent. 
'''

import numpy as np


class SortProbing:
    
    def __init__(self, feature_vector, nfeatures, match_indices, nonmatch_indices):
        '''
        feature_vector: (pair_index, features)
        nfeatures: number of features
        match_indices: list of indices for match pairs 
        nonmatch_indices: list of indices for nonmatch pairs 
        '''
        
        self.feature_vector = feature_vector
        self.nfeatures = nfeatures
        self.match_indices = match_indices
        self.nonmatch_indices = nonmatch_indices
        self.min_cons_dim = 1 # it is only designed for 1-dimension consistency
        
        self.feature_to_sorted_nonmatch = []
        
        self.clusters_to_match_poses = []
        self.clusters_to_nonmatch_poses = []
        
        self.feature_to_sorted_clusters = []
        
        self.match_pos_to_index = {}
        self.nonmatch_pos_to_index = {}
        
    def find_end_index(self, match_clusters, nonmath_clusters):
        end_index = {}
        
        i = 0
        k = 0
        
        while i<len(match_clusters):
            while k<len(nonmath_clusters) and match_clusters[i]>nonmath_clusters[k]:
                k += 1
            end_index[ match_clusters[i] ] = k
            i += 1
            
        return end_index

    def cluster_indices_for_all_features(self):
        
        for i in range(self.nfeatures):
            self.clusters_to_match_poses.append( { } )
            self.clusters_to_nonmatch_poses.append( { } )
                
        for pos, index in enumerate(self.match_indices):
            self.match_pos_to_index[pos] = index
            features = self.feature_vector[index]
            
            for i in range(self.nfeatures):
                f = features[i]
                if f in self.clusters_to_match_poses[i]:
                    self.clusters_to_match_poses[i][f].append(pos)
                else:
                    self.clusters_to_match_poses[i][f] = [pos]
                    
        for pos, index in enumerate(self.nonmatch_indices):
            self.nonmatch_pos_to_index[pos] = index
            features = self.feature_vector[index]
            for i in range(self.nfeatures):
                f = features[i]
                if f in self.clusters_to_nonmatch_poses[i]:
                    self.clusters_to_nonmatch_poses[i][f].append(pos)
                else:
                    self.clusters_to_nonmatch_poses[i][f] = [pos]
                    
        for i in range(self.nfeatures):
            match_clusters = list(self.clusters_to_match_poses[i].keys())
            match_clusters.sort()
            nonmatch_clusters = list(self.clusters_to_nonmatch_poses[i].keys())
            nonmatch_clusters.sort()
            self.feature_to_sorted_clusters.append( (match_clusters, nonmatch_clusters, self.find_end_index(match_clusters, nonmatch_clusters) ) )
    
    '''
    def sort_indices_for_all_features(self):
        for i in range(self.nfeatures):
            list_of_index_feature = [ (index, self.feature_vector[index][i]) for index in self.nonmatch_indices ]
            list_of_index_feature.sort(key=lambda t: t[1]) # ascending order of feature value
            self.feature_to_sorted_nonmatch.append(list_of_index_feature)
    '''
            
    def mark_cons_poses(self, pos_list, cluster_value, feature_index):
        tmp = self.feature_to_sorted_clusters[feature_index]
        nonmatch_clusters = tmp[1]
        end_index = tmp[2][cluster_value]
        
        cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
        for i in range(end_index):
            for pos in cluster_to_nonmatch_pos[ nonmatch_clusters[i] ]:
                pos_list[pos] = 1
    
    def get_list_of_inconsistent_indices(self, match_pair_pos):
        pos_list = np.zeros(len(self.nonmatch_indices)) # all init to zero
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        for feature_index in range(self.nfeatures):
            self.mark_cons_poses(pos_list, self.feature_vector[match_index][feature_index], feature_index)
            
        incons_nonmatch_indices = []
        for i in range(len(self.nonmatch_indices)):
            if pos_list[i] == 0:
                incons_nonmatch_indices.append(self.nonmatch_pos_to_index[i])
                
        return incons_nonmatch_indices
    
    def get_inconsistency_indices(self):
        
        # clustering
        self.cluster_indices_for_all_features()
        
        incons_indices = {}
        for index in self.match_indices:
            incons_indices[index] = []
        for index in self.nonmatch_indices:
            incons_indices[index] = []
        
        # probing
        for match_pair_pos, match_index in self.match_pos_to_index.items():
            incons_nonmatch_indices = self.get_list_of_inconsistent_indices(match_pair_pos)
            
            incons_indices[match_index] = incons_nonmatch_indices
            for nonmatch_index in incons_nonmatch_indices:
                incons_indices[nonmatch_index].append(match_index)
        
        return incons_indices
