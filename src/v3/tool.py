'''
Created on Mar 1, 2017

@author: hzhang0418
'''

import v3.fpfn as fpfn
import v3.fpfn_irf as irf
import v3.brute_force as bf
import v3.spatial_blocking as sb
import v3.sort_probing as sp
import v3.hybrid as hybrid

class Tool(object):

    def __init__(self, params, features, labels):
        
        self.params = params
        self.features = features
        self.labels = labels
        
        self.alg = params['approach']
        self.top_k = params['top_k']
        self.max_iter = params['max_iter']
        
        self.iteration = 0
        
        self.fpfn = params['fpfn'] if 'fpfn' in params else 'fpfn'
        self.mono = params['mono'] if 'mono' in params else 'sp'

    def create_debugger(self):
        if self.alg == 'fpfn':
            if self.fpfn == 'fpfn':
                self.debugger = fpfn.FPFN(self.features, self.labels, 5, fs_alg=self.params['fs_alg'])
            elif self.fpfn == 'irf':
                self.debugger = irf.FPFN_IRF(self.features, self.labels, 5, fs_alg=self.params['fs_alg'])
            
        elif self.alg == 'mono': 
            if self.mono == 'sp':
                self.debugger = sp.SortProbing(self.features, self.labels, 1, self.params['use_mvc'], self.params['fs_alg'])  
            elif self.mono == 'sb':
                self.debugger = sb.SpatialBlocking(self.features, self.labels, 1, 3, self.params['use_mvc'], self.params['fs_alg'])
            elif self.mono == 'bf':
                self.debugger = bf.BruteForce(self.features, self.labels, 1, self.params['use_mvc'], self.params['fs_alg'])
        
        elif self.alg == 'hybrid':
            self.debugger = hybrid.Hybrid(self.features, self.labels, 100, self.params['use_mvc'])
    
    def detect(self):
        self.iteration += 1
        
        indices = self.debugger.detect(self.top_k)

        return indices
        
    def get_feedback(self, user_fb):
        
        self.debugger.use_feedback(user_fb)
        
        