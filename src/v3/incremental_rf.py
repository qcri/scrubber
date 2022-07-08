'''
Created on Mar 22, 2017

@author: hzhang0418
'''

import irf

def load_from_file(input_file):
    clf = IncrementalRF(0, [], [], [])
    clf.f = irf.load(input_file)
    return clf

class IncrementalRF(object):
    
    def __init__(self, ntrees, indices, features, labels):
        self.f = irf.IRF(ntrees)
        
        for i, index in enumerate(indices):
            self.f.add(str(index), { k:v for k,v in enumerate(features[i]) }, labels[i])
    
    def add(self, indices, new_sample_features, new_sample_labels):
        for i, index in enumerate(indices):
            self.f.add(str(index), { k:v for k,v in enumerate(new_sample_features[i]) }, new_sample_labels[i])
    
    def remove(self, sampleIDs):
        for sID in sampleIDs:
            self.f.remove(str(sID))
    
    def predict(self, test_set_features):
        nrows, _ = test_set_features.shape
        
        probs = []
        for i in range(nrows):
            probs.append( self.f.classify({ k:v for k,v in enumerate(test_set_features[i]) }) )
            
        return probs
    
    def save_to_file(self, output_file):
        self.f.save(output_file)