'''
Created on Feb 16, 2017

@author: hzhang0418

Sort Probing algorithm

Basic idea:
For each feature, sort the list of matches and the list of nonmatches, 
then for each match (f), find the list of nonmatch with same or better value in this feature (L_f)

Then for a match, its list of nonmatches that violate mono property in at least one feature
is the intersection of all L_f's

1. First implementation use bitarrays
2. Second implementation use lists
3. Third implementation uses both lists and bitarrays 

'''

'''
Only work for 1-dimension consistency

This uses bitarray to speedup

For each feature, cluster matches and nonmathes according to the feature value

For each feature, for each cluster of matches, get the list of nonmatches clusters that are inconsistent with the matches in the cluster at this feature

for each match, intersect the lists of nonmatches that are inconsistent with it at each feature
'''

import bitarray as bar

from v3.mono import Mono

class SortProbing(Mono):
    
    def __init__(self, features, labels, min_con_dim, use_mvc, fs_alg):
        assert(min_con_dim == 1)
        super(SortProbing, self).__init__(features, labels, 1, use_mvc, fs_alg)
        
        self.feature_to_sorted_nonmatch = []
        
        # map pos to its index
        self.match_pos_to_index = {}
        self.nonmatch_pos_to_index = {}
        
        # list of dicts of lists, where the size of the first list == self.nfeatures
        # for each dict, a key is a feature value, and the associated value 
        # is the list of poses of matches (or nonmatches)  
        self.clusters_to_match_poses = [] 
        self.clusters_to_nonmatch_poses = []
        
        # list of tuples
        # each tuple is (match_clusters, nonmatch_clusters, the end_index map)
        self.feature_to_sorted_clusters = []
        
        self.match_clusters_to_nonmatch_bitarray = []
        
    def find_end_index(self, match_clusters, nonmatch_clusters, feature_index):
        # for each match cluster, find the first nonmatch cluster with same or greater value
        end_index = {}
        # count the number of nonmatch indices with same or greater value
        index_count = {}
        
        i = 0
        k = 0
        
        cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
        count = len(self.nonmatch_indices)
        
        while i<len(match_clusters):
            while k<len(nonmatch_clusters) and match_clusters[i]>nonmatch_clusters[k]:
                count -= len(cluster_to_nonmatch_pos[ nonmatch_clusters[k] ])
                k += 1
            end_index[ match_clusters[i] ] = k
            index_count[ match_clusters[i] ] = count
            i += 1
            
        return end_index, index_count

    def cluster_indices_for_all_features(self):
        
        for i in range(self.nfeatures):
            self.clusters_to_match_poses.append( { } )
            self.clusters_to_nonmatch_poses.append( { } )
                
        for pos, index in enumerate(self.match_indices):
            self.match_pos_to_index[pos] = index
            features = self.feature_vector_map[index]
            
            for i in range(self.nfeatures):
                f = features[i]
                if f in self.clusters_to_match_poses[i]:
                    self.clusters_to_match_poses[i][f].append(pos)
                else:
                    self.clusters_to_match_poses[i][f] = [pos]
                    
        for pos, index in enumerate(self.nonmatch_indices):
            self.nonmatch_pos_to_index[pos] = index
            features = self.feature_vector_map[index]
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
            self.feature_to_sorted_clusters.append( (match_clusters, nonmatch_clusters, self.find_end_index(match_clusters, nonmatch_clusters, i) ))
    
    '''
    def sort_indices_for_all_features(self):
        for i in range(self.nfeatures):
            list_of_index_feature = [ (index, self.feature_vector[index][i]) for index in self.nonmatch_indices ]
            list_of_index_feature.sort(key=lambda t: t[1]) # ascending order of feature value
            self.feature_to_sorted_nonmatch.append(list_of_index_feature)
    '''
            
    def create_mixes(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, index_counts = triple[2]
                
                end_index = end_indices[cluster_value]
                count = index_counts[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                
                if count>2000: # create bitarray
                    a = bar.bitarray(len(self.nonmatch_indices))
                    a.setall(True)
                    
                    # set bits for incons nonmatches
                    for i in range(end_index, len(nonmatch_clusters)):
                        for pos in cluster_to_nonmatch_pos[ nonmatch_clusters[i] ]:
                            a[pos] = False
                    
                    tmp[cluster_value] = a
                else:
                    # use list instead
                    a = []
                    for i in range(end_index, len(nonmatch_clusters)):
                        a.extend(cluster_to_nonmatch_pos[ nonmatch_clusters[i] ])
                        
                    tmp[cluster_value] = a
                    
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
    
    
    def get_list_of_inconsistent_indices_mixes(self, match_pair_pos):
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        cluster_value = self.feature_vector_map[match_index][0]
        tmp = self.match_clusters_to_nonmatch_bitarray[0][cluster_value]
        
        if isinstance(tmp, list):
            pos_bitarray = []
            pos_bitarray.extend(tmp)
        else:
            pos_bitarray = bar.bitarray(len(self.nonmatch_indices)) # all init to zero
            pos_bitarray.setall(False)
            pos_bitarray |= tmp
        
        for feature_index in range(1, self.nfeatures):
            cluster_value = self.feature_vector_map[match_index][feature_index]
            incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
            if isinstance(incons_bitarray, list):
                if isinstance(pos_bitarray, list):
                    pos_bitarray = [ pos for pos in pos_bitarray if pos in incons_bitarray ]        
                else:
                    pos_bitarray = [ pos for pos in incons_bitarray if pos_bitarray[pos]==False ]
            else:
                if isinstance(pos_bitarray, list):
                    pos_bitarray = [ pos for pos in pos_bitarray if incons_bitarray[pos]==False ]
                else:
                    pos_bitarray |= incons_bitarray
            
        incons_nonmatch_indices = []
        
        if isinstance(pos_bitarray, list):
            for pos in pos_bitarray:
                incons_nonmatch_indices.append(self.nonmatch_pos_to_index[pos])
        else:
            for i in range(len(self.nonmatch_indices)):
                if pos_bitarray[i] == False:
                    incons_nonmatch_indices.append(self.nonmatch_pos_to_index[i])
                
        return incons_nonmatch_indices
    
    def get_inconsistency_indices(self):
        
        # clustering
        self.cluster_indices_for_all_features()
        
        self.create_mixes()
        #self.create_lists()
        #self.create_bitarrays()
        
        # probing
        for match_pair_pos, match_index in self.match_pos_to_index.items():
            incons_nonmatch_indices = self.get_list_of_inconsistent_indices_mixes(match_pair_pos)
            #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vlist(match_pair_pos)
            #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vbitarray(match_pair_pos)
            
            if len(incons_nonmatch_indices) == 0:
                continue
            
            self.index2incons[match_index] = incons_nonmatch_indices
            for nonmatch_index in incons_nonmatch_indices:
                if nonmatch_index in self.index2incons:
                    self.index2incons[nonmatch_index].append(match_index)
                else:
                    self.index2incons[nonmatch_index] = [ match_index ] 
                    
                    
    '''
    using list
    '''
    def create_lists(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, index_counts = triple[2]
                
                end_index = end_indices[cluster_value]
                count = index_counts[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                
                # use list instead
                a = []
                for i in range(end_index, len(nonmatch_clusters)):
                    a.extend(cluster_to_nonmatch_pos[ nonmatch_clusters[i] ])

                tmp[cluster_value] = a
                    
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
    
    
    def get_list_of_inconsistent_indices_vlist(self, match_pair_pos):
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        pos_bitarray = []
        
        cluster_value = self.feature_vector_map[match_index][0]
        incons_bitarray = self.match_clusters_to_nonmatch_bitarray[0][cluster_value]
        
        pos_bitarray.extend(incons_bitarray)
        
        for feature_index in range(1, self.nfeatures):
            cluster_value = self.feature_vector_map[match_index][feature_index]
            incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
            pos_bitarray = [ pos for pos in pos_bitarray if pos in incons_bitarray ]        

        incons_nonmatch_indices = []
        
        for pos in pos_bitarray:
            incons_nonmatch_indices.append(self.nonmatch_pos_to_index[pos])
                
        return incons_nonmatch_indices
    
    '''
    using bitarray 
    '''
    def create_bitarrays(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                a = bar.bitarray(len(self.nonmatch_indices))
                a.setall(True)
                
                # set bits for incons nonmatches
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, _ = triple[2]
                end_index = end_indices[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                for i in range(end_index, len(nonmatch_clusters)):
                    for pos in cluster_to_nonmatch_pos[ nonmatch_clusters[i] ]:
                        a[pos] = False
                
                tmp[cluster_value] = a
                 
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
        
            
    def mark_incons_poses(self, pos_bitarray, cluster_value, feature_index):
        incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
        pos_bitarray |= incons_bitarray
    
    def get_list_of_inconsistent_indices_vbitarray(self, match_pair_pos):
        pos_bitarray = bar.bitarray(len(self.nonmatch_indices)) # all init to zero
        pos_bitarray.setall(False)
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        for feature_index in range(self.nfeatures):
            self.mark_incons_poses(pos_bitarray, self.feature_vector_map[match_index][feature_index], feature_index)
            
        incons_nonmatch_indices = []
        for i in range(len(self.nonmatch_indices)):
            if pos_bitarray[i] == False:
                incons_nonmatch_indices.append(self.nonmatch_pos_to_index[i])
                
        return incons_nonmatch_indices
        