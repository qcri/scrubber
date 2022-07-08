'''
Created on Feb 16, 2017

@author: hzhang0418

Feature selection using Random Forest

'''

import numpy as np
import operator

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_features_from_model(features, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(features, labels)
    #print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(features)

def select_features(features, labels, nfolds, min_num_trees):
    
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    
    kf = KFold(nfolds, shuffle=True, random_state = 0)
        
    feature_info = []
    for train_index, _ in kf.split(features):
        train_set_features  = features[train_index]
        train_set_labels = labels[train_index]
        
        clf.fit(train_set_features, train_set_labels)
        
        rf_features = []
        for tree in clf.estimators_:
            tree_features = []
            for f in tree.tree_.feature:
                if f != -2:
                    tree_features.append(f)
            rf_features.append(tree_features)
        
        feature_info.append(rf_features)
        
    feature_indices = get_features_for_monotonicity(feature_info, min_num_trees)
    
    return features[:, feature_indices]

def count_trees_for_features(rf_features):
    trees_counts = {}
    for tree_features in rf_features:
        for f in tree_features:
            if f in trees_counts:
                trees_counts[f] += 1
            else:
                trees_counts[f] = 1
    
    return trees_counts  
        
def get_features_for_monotonicity(feature_info, thresh):
    filtered = []

    for rf_features in feature_info:
        trees_counts = count_trees_for_features(rf_features)
        filtered_features = []
        for f, c in trees_counts.items():
            if c >= thresh:
                filtered_features.append(f)
        filtered.append(set(filtered_features))
        
    final_set = filtered[0]
    for myset in filtered[1:]:
        final_set = final_set & myset
        
    features = list(final_set)
    features.sort()

    return features
        
    
