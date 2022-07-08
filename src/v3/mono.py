'''
Created on Feb 16, 2017

@author: hzhang0418

The template for different implementations of Mono approach

'''

import numpy as np

from operator import itemgetter
from v3.label_debugger import LabelDebugger

import v3.feature_selection as fs
import v3.mvc as mvc
import v3.gvc as gvc

class Mono(LabelDebugger):
    
    def __init__(self, features, labels, min_con_dim, use_mvc=True, fs_alg='none'):
        super(Mono, self).__init__()
        
        self.min_con_dim = min_con_dim
        self.use_mvc = use_mvc
        self.feature_vector_map = {}
        self.match_indices = []
        self.nonmatch_indices = []
        self.index2incons = {}
        self.num_iter = 0
        
        (nrow, ncol) = features.shape
        print("Number of available features: ", ncol)
        
        # feature selection
        if fs_alg=='rf':
            selected_features = fs.select_features(features, labels, 5, 3)
        elif fs_alg=='model':            
            selected_features = fs.select_features_from_model(features, labels)
        else:
            selected_features = features
            
        (nrow, ncol) = selected_features.shape    
        index = 0
        for row in selected_features:
                self.feature_vector_map[index] = list(row)
                if labels[index] == 1:
                    self.match_indices.append(index)
                else:
                    self.nonmatch_indices.append(index)
                index += 1
                        
        self.nfeatures = ncol
        print("Number of used features: ", self.nfeatures)


    def detect_and_rank(self):
        if self.num_iter == 0:
            self.get_inconsistency_indices()
            count = 0
            for _,v in self.index2incons.items():
                count+= len(v)
            print("Total number of inconsistencies: ", count)
            
        if self.use_mvc:
            indices = self.getMVC()
            #indices = self.getGVC()
            
            incons_counts = []
            for index in indices:
                incons_counts.append((index,-len(self.index2incons[index]))) # note that negative count is used
        
            incons_counts = sorted(incons_counts, key=itemgetter(1,0))
            
            ranked = incons_counts
        
        else:
            ranked = self.rank_inconsistency_indices()
        
        ranked_indices = []
        for t in ranked:
            if t[1]<0:
                ranked_indices.append(t[0])
            else:
                break
        
        self.num_iter += 1
        
        return ranked_indices
    
    
    def use_feedback(self, user_fb):
        corrected_match_indices = []
        corrected_nonmatch_indices = []
        
        for index, label in user_fb.items():
            if label == 1:
                corrected_match_indices.append(index)
            else:
                corrected_nonmatch_indices.append(index)
        
        self.update_inconsistency_indices(corrected_match_indices, corrected_nonmatch_indices)
        
    
    def get_inconsistency_indices(self):
        print("Child class must override this function! ")
        
    
    def rank_inconsistency_indices(self):
        incons_counts = []
        for k,v in self.index2incons.items():
            incons_counts.append((k,-len(v))) # note that negative count is used
        
        incons_counts = sorted(incons_counts, key=itemgetter(1,0))
        
        return incons_counts
    
    
    def update_inconsistency_indices(self, corrected_match_indices, corrected_nonmatch_indices):
        '''
        Params:
        corrected_match_indices: the list of indices that were wrongly labeled as nonmatch
        corrected_nonmatch_indices: the list of indices that were wrongly labeled as match
        
        to update index2incons:
        First, update nonmatch_indices and match_indices
        
        For each index (x) in corrected_match_indices:
        1. get its list of incon indices,
        2. for each index (y) in the above list, remove x from y's list of incon indices
        
        For each index (x) in corrected_nonmatch_indices:
        1. get its list of incon indices,
        2. for each index (y) in the above list, remove x from y's list of incon indices

        
        For each index (x) in corrected_match_indices:
        3. compute the new list of incon indices for x by comparing x with all nonmatch indices 
        4. for each index (z) in the new list, add x to z's list of incon indices
        
        For each index (x) in corrected_nonmatch_indices:
        3. compute the new list of incon indices for x by comparing x with all match indices 
        4. for each index (z) in the new list, add x to z's list of incon indices
        '''
        
        # update nonmatch_indices and match_indices
        self.match_indices = [ index for index in self.match_indices if index not in corrected_nonmatch_indices ]
        self.match_indices.extend(corrected_match_indices)
        
        self.nonmatch_indices = [ index for index in self.nonmatch_indices if index not in corrected_match_indices ]
        self.nonmatch_indices.extend(corrected_nonmatch_indices)
        
        # process x in corrected_match_indices
        for x in corrected_match_indices:
            x_incons = self.index2incons[x]
            
            for y in x_incons:
                self.index2incons[y].remove(x)
                
        # process x in corrected_nonmatch_indices
        for x in corrected_nonmatch_indices:
            x_incons = self.index2incons[x]
            for y in x_incons:
                self.index2incons[y].remove(x)
                
        #compute incons for those in feedback
        for x in corrected_match_indices:
            x_incons = []
            match_features = self.feature_vector_map[x]
            for z in self.nonmatch_indices:
                is_incon = self.compare_features(match_features, self.feature_vector_map[z], self.min_con_dim)
                if is_incon:
                    x_incons.append(z)
                    if z in self.index2incons:
                        self.index2incons[z].append(x)
                    else:
                        self.index2incons[z] = [x]
            self.index2incons[x] = x_incons 
            
        for x in corrected_nonmatch_indices:
            x_incons = []
            nonmatch_features = self.feature_vector_map[x]
            for z in self.match_indices:
                is_incon = self.compare_features(self.feature_vector_map[z], nonmatch_features, self.min_con_dim)
                if is_incon:
                    x_incons.append(z)
                    if z in self.index2incons:
                        if x not in self.index2incons[z]:
                            self.index2incons[z].append(x)
                    else:
                        self.index2incons[z] = [x]
            self.index2incons[x] = x_incons 
        
    
    def compare_features(self, match_features, nonmatch_features, min_con_dim):
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
    
    
    def getMVC(self):
            
        matches = {}
        for pos in self.match_indices:
            matches[pos] = 1;
        
        left = {}
        right = {}
         
        for index, incon_list in self.index2incons.items():
            if len(incon_list)==0:
                continue
            if index in matches:
                left[index] = list(incon_list)
            else:
                right[index] = list(incon_list);

        print("Before MVC: ", len(left)+len(right))
        return mvc.min_vertex_cover(left, right)
    
    
    def getGVC(self):
        print("Before GVC: ", len([index for index, incon_list in self.index2incons.items() if len(incon_list)>0]))
        
        return gvc.greedy_vertex_cover(self.index2incons)
    
    