'''
Created on Mar 1, 2017

@author: hzhang0418
'''
from v3.label_debugger import LabelDebugger

import v3.fpfn as fpfn
import v3.brute_force as bf
import v3.sort_probing as sp

class Hybrid(LabelDebugger):
    
    def __init__(self, features, labels, max_ranked, use_mvc=True):
        super(Hybrid, self).__init__()
        
        self.mono = sp.SortProbing(features, labels, 1, use_mvc)
        #self.mono = bf.BruteForce(features, labels, 1)
        self.fpfn = fpfn.FPFN(features, labels, 5)
        self.max_ranked = max_ranked
        
        self.mono_ranked = []
        self.fpfn_ranked = []
    
    def detect_and_rank(self):
        #'''
        tmp = self.mono.detect_and_rank()
        self.mono_ranked = [t for t in tmp if t not in self.checked_indices ]
        
        tmp = self.fpfn.detect_and_rank()
        self.fpfn_ranked = [t for t in tmp if t not in self.checked_indices ]
        #'''
        
        '''
        self.mono_ranked = self.mono.detect_and_rank()
        self.fpfn_ranked = self.fpfn.detect_and_rank()
        '''
        
        #combined = self.combine_ranked_lists(self.mono_ranked[: self.max_ranked], 0.5, self.fpfn_ranked[ : self.max_ranked], 0.5)
        combined = self.combine_ranked_lists_v2(self.mono_ranked[: self.max_ranked], 0.5, self.fpfn_ranked[ : self.max_ranked], 0.5)
        
        return combined
        
    def use_feedback(self, user_fb):
        
        mono_fb = {}
        fpfn_fb = {}
        
        for index in self.mono_ranked:
            if index in user_fb:
                mono_fb[index] = user_fb[index]
                
        for index in self.fpfn_ranked:
            if index in user_fb:
                fpfn_fb[index] = user_fb[index]
        
        self.mono.use_feedback(mono_fb)
        self.fpfn.use_feedback(fpfn_fb)
    
    
    def combine_ranked_lists(self, first, first_weight, second, second_weight):
        all_pairs = set(first + second)
        ranking = {}
        for t in all_pairs:
            ranking[t] = [len(first)+1,len(second)+1,0]
        for i, t in enumerate(first,1):
            ranking[t][0] = i
        for i, t in enumerate(second, 1):
            ranking[t][1] = i
        for t in all_pairs:
            ranking[t][2] = ranking[t][0]*first_weight + ranking[t][1]*second_weight
        
        tmp = [ (k,v[2]) for k,v in ranking.items() ]
        tmp.sort(key=lambda x: x[1])
        
        return [ t[0] for t in tmp] 
    
    def combine_ranked_lists_v2(self, first, first_weight, second, second_weight):
        all_pairs = set(first + second)
        ranking = {}
        for i, t in enumerate(first,1):
            ranking[t] = [i,0,0]
            
        r = len(first)
        for t in all_pairs:
            if t not in ranking:
                ranking[t] = [r,0,0]
                r += 1
            
        for i, t in enumerate(second, 1):
            ranking[t][1] = i
            
        r = len(second)
        tmp = set(second)
        for t in all_pairs:
            if t not in tmp:
                ranking[t][1] = r
                r += 1
            
        for t in all_pairs:
            ranking[t][2] = ranking[t][0]*first_weight + ranking[t][1]*second_weight
        
        tmp = [ (k,v[2]) for k,v in ranking.items() ]
        tmp.sort(key=lambda x: x[1])
        
        return [ t[0] for t in tmp] 
    
    def combine_ranked_lists_method2(self, first, second):
        tmp = []
        num = min(len(first), len(second))
        for i in range(num):
            if first[i] not in tmp:
                tmp.append(first[i])
            if second[i] not in tmp:
                tmp.append(second[i])
        
        return tmp 