'''
Created on Mar 22, 2017

@author: hzhang0418
'''

import numpy as np
from operator import itemgetter

from sklearn.model_selection import KFold

from v3.label_debugger import LabelDebugger
from v3.incremental_rf import IncrementalRF

import v3.feature_selection as fs

class FPFN_IRF(LabelDebugger):
    
    def __init__(self, features, labels, nfolds, rank_alg='ml', fs_alg='none'):
        super(FPFN_IRF, self).__init__()
        
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
        self.nfeatures = ncol
        print("Number of used features: ", self.nfeatures)
        
        self.features = selected_features
        self.labels = labels
        self.nfolds = nfolds
        self.rank_alg = rank_alg
        
        self.irfs = []
        self.indices2folds = {}
        
        self.incon_indices = {}
        
        self.num_iter = 0
        
    def detect_and_rank(self):
        if self.num_iter==0:
            self.get_inconsistency_indices()
            
        ranked = self.rank_inconsistency_indices()
        
        print("Number of suspicious examples after ranking:", len(ranked))
        
        self.num_iter += 1
        
        return [ t[0] for t in ranked ]
    
    def use_feedback(self, user_fb):
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
        mismatching = self.cross_validation(self.features, self.labels, self.nfolds)
        
        print("Number of suspicious examples after CV:", len(mismatching))
        
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
            #incons_prob.append( (k, -np.max(v)) )
            incons_prob.append( (k, abs(0.5-v)) )
        
        incons_prob = sorted(incons_prob, key=itemgetter(1,0))
        
        return incons_prob
    
    
    def cross_validation(self, features, labels, nfolds):
        
        kf = KFold(nfolds, shuffle=True, random_state = 0)
        
        mismatching = {}
        
        for train_index, test_index in kf.split(features):
            
            train_set_features, test_set_features = features[train_index], features[test_index]
            train_set_labels, test_set_labels = labels[train_index], labels[test_index]
            
            clf = IncrementalRF(20, train_index, features, labels)
            
            proba = clf.predict(test_set_features)
            
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
                
            rf_index = len(self.irfs)
            for index in train_index:
                self.indices2folds[index] = rf_index
                
            self.irfs.append( (clf, test_index, test_set_features) )
        
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
        