'''
Created on Dec 10, 2018

@author: hzhang0418
'''


import numpy as np
from operator import itemgetter

from v3.label_debugger import LabelDebugger
from v3.incremental_rf import IncrementalRF

class KLooFPFN_IRF(LabelDebugger):
    
    def __init__(self, features, labels, nmax):
        super(KLooFPFN_IRF, self).__init__()
        
        self.features = features
        self.labels = labels
        
        self.irfs = []
        self.indices2folds = {}
        
        self.incon_indices = {}
        
        self.num_iter = 0
        
        self.nmax = nmax # max number of add/remove
        
    def detect_and_rank(self):
        if self.num_iter==0:
            self.get_inconsistency_indices()
            
        ranked = self.rank_inconsistency_indices()
        
        self.num_iter += 1
        
        return [ t[0] for t in ranked ]
    
    def use_feedback(self, user_fb):
        print("Not supported yet!")
        return None
    
        self.incon_indices.clear()
        
        fold_indices = {}
        for i in range(self.nfolds):
            fold_indices[i] = []
        
        for k,v in user_fb.items():
            self.labels[k] = v
            fold_indices[ self.indices2folds[k] ].append(k)
            
        # update each irf
        mismatching = {}
        
        for i in range(self.nfolds):
            clf, test_index, test_set_features = self.irfs[i]
            indices = fold_indices[i]
            clf.remove(indices) # remove from training due to label changes
            new_sample_features, new_sample_labels = self.features[indices], self.labels[indices]
            clf.add(indices, new_sample_features, new_sample_labels) # add to training
            
            proba = clf.predict(test_set_features)
            
            test_set_labels = self.labels[test_index]
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
        
        # update last irf
        test_index = list(mismatching.keys())
        # remove samples that was in training but now in testing
        indices = [ k for k in test_index if k not in self.last_test_index ]
        self.last_irf.remove(indices)
        # add samples that was in testing but now in training
        indices = [ k for k in self.last_test_index if k not in test_index ]
        new_sample_features = self.features[indices]
        new_sample_labels = self.labels[indices]
        self.last_irf.add(indices, new_sample_features, new_sample_labels)
        
        test_set_features = self.features[test_index]
        test_set_labels = self.labels[test_index] 
        
        proba = self.last_irf.predict(test_set_features)
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            self.incon_indices[ test_index[k] ] = v
        
    def get_inconsistency_indices(self):
        
        # cross validation
        mismatching = self.loo_cross_validation(self.features, self.labels)
        
        # samples with matching labels as train set
        # samples with mismatching labels as test set
        test_index = list(mismatching.keys())
        train_index = [ i for i in range(len(self.features)) if i not in mismatching ]
        
        train_set_features, test_set_features = self.features[train_index], self.features[test_index]
        train_set_labels, test_set_labels = self.labels[train_index], self.labels[test_index]
        
        # predict again
        self.last_irf = IncrementalRF(20, train_index, self.features, self.labels)
        self.last_test_index = test_index
        proba = self.last_irf.predict(test_set_features)
    
        # find samples with mismatching labels in test set
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            self.incon_indices[ test_index[k] ] = v
    
    def rank_inconsistency_indices(self):
        
        incons_prob = []
        for k,v in self.incon_indices.items():
            incons_prob.append( (k, -np.max(v)) )
        
        incons_prob = sorted(incons_prob, key=itemgetter(1,0))
        
        return incons_prob
    
    
    def loo_cross_validation(self, features, labels):
        
        mismatching = {}
        
        # to predict first, train on all remaining
        train_index = range(1,len(labels))
        test_index = [0]
        
        train_set_features, test_set_features = features[train_index], features[test_index]
        train_set_labels, test_set_labels = labels[train_index], labels[test_index]
        
        clf = IncrementalRF(20, train_index, features, labels)
            
        proba = clf.predict(test_set_features)
        
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            mismatching[ test_index[k] ] = v
            
        nmodifications = 0
        
        for index in range(1, len(labels)):
            test_index = [index]
            
            if nmodifications>=self.nmax:
                train_index = list(range(index)) + list(range(index+1, len(labels))) 
                clf = IncrementalRF(20, train_index, features, labels)
                nmodifications = 0
            else:
                # remove example at index
                clf.remove(test_index)
                # add previous example
                prev_index = index-1
                clf.add([prev_index], [ self.features[prev_index] ], [ labels[prev_index] ])
                nmodifications += 1
            
            test_set_features = features[test_index]
            test_set_labels = labels[test_index]
            
            proba = clf.predict(test_set_features)
            
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
        
        return mismatching
        
    def find_mismatching(self, proba, labels):
        
        # find predicted class
        #predicted = np.argmax(proba, axis=1)
        predicted = [ int(round(p)) for p in proba ]
        
        # find those mismatching index
        diff = np.where(predicted!=labels)[0]
        
        mismatching = {}
        for index in diff:
            mismatching[index] = proba[index]
            
        return mismatching 