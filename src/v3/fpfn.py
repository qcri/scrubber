'''
Created on Feb 22, 2017

@author: hzhang0418

'''

import numpy as np
from operator import itemgetter

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from v3.label_debugger import LabelDebugger

import v4.count_incon as ci
import v3.feature_selection as fs

class FPFN(LabelDebugger):
    
    def __init__(self, features, labels, nfolds, rank_alg='ml', fs_alg='none'):
        super(FPFN, self).__init__()
        
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
        
        self.clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
        self.features = selected_features
        self.labels = labels
        self.nfolds = nfolds
        self.rank_alg = rank_alg
        
        self.incon_indices = {}
        
        self.num_iter = 0
        
    def detect_and_rank(self):
        self.get_inconsistency_indices()
        ranked = self.rank_inconsistency_indices()
        
        print("Number of suspicious examples after ranking:", len(ranked))
        
        self.num_iter += 1
        
        return [ t[0] for t in ranked ]
    
    def use_feedback(self, user_fb):
        for k,v in user_fb.items():
            self.labels[k] = v
            
        self.incon_indices.clear()

    def get_inconsistency_indices(self):
        
        # cross validation
        mismatching = self.cross_validation(self.clf, self.features, self.labels, self.nfolds)
        
        print("Number of suspicious examples after CV:", len(mismatching))
        
        # samples with matching labels as train set
        # samples with mismatching labels as test set
        test_index = list(mismatching.keys())
        train_index = [ i for i in range(len(self.features)) if i not in mismatching ]
        
        train_set_features, test_set_features = self.features[train_index], self.features[test_index]
        train_set_labels, test_set_labels = self.labels[train_index], self.labels[test_index]
        
        # predict again
        proba = self.train_and_test(self.clf, train_set_features, train_set_labels, test_set_features)
    
        # find samples with mismatching labels in test set
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            self.incon_indices[ test_index[k] ] = v
    
    def rank_inconsistency_indices(self):
        incons_prob = []
            
        if self.rank_alg=='ml':
            for k,v in self.incon_indices.items():
                incons_prob.append( (k, -np.max(v)) )
        else:
            suspicious_indices = list(self.incon_indices)
            golden_indices = [ i for i in range(len(self.features)) if i not in self.incon_indices]
            index2count = ci.count(self.features, self.labels, suspicious_indices, golden_indices, 1)
            for k,v in index2count.items():
                incons_prob.append( (k, -v))
        
        incons_prob = sorted(incons_prob, key=itemgetter(1,0))
        
        return incons_prob
    
    
    def cross_validation(self, classifier, features, labels, nfolds):
        
        kf = KFold(nfolds, shuffle=True, random_state = 0)
        
        mismatching = {}
        
        for train_index, test_index in kf.split(features):
            train_set_features, test_set_features = features[train_index], features[test_index]
            train_set_labels, test_set_labels = labels[train_index], labels[test_index]
            
            proba = self.train_and_test(self.clf, train_set_features, train_set_labels, test_set_features)
            
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
        
        return mismatching
    
    
    def train_and_test(self, classifier, train_set_features, train_set_labels, test_set_features):
        # train
        self.clf.fit(train_set_features, train_set_labels)
        # predict
        return self.clf.predict_proba(test_set_features)
        
    def find_mismatching(self, proba, labels):
        # find predicted class
        predicted = np.argmax(proba, axis=1)
        #predicted = [ int(round(p)) for p in proba ]
        
        # find those mismatching index
        diff = np.where(predicted!=labels)[0]
        
        mismatching = {}
        for index in diff:
            mismatching[index] = proba[index]
            
        return mismatching