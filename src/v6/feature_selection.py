'''
Created on Mar 5, 2019

@author: hzhang0418
'''

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2

def select_features_from_model(features, labels):
    clf = RandomForestClassifier(n_estimators=10, random_state=0, criterion='gini', n_jobs=-1)
    #clf = LinearSVC(C=.01, penalty='l1', dual=False)
    clf = clf.fit(features, labels)
    #print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    selected = model.get_support(True)
    print('Selected features: ', len(selected), ' coefficients: ', clf.feature_importances_)
    return model.transform(features), selected

def select_lr_features(features, labels):
    clf = LogisticRegression(C=0.01, dual=False).fit(features, labels)
    model = SelectFromModel(clf, prefit=True)
    selected = model.get_support(True)
    print('Selected features: ', len(selected), ' coefficients: ', clf.coef_)
    return model.transform(features), selected

def select_boosting_features(features, labels):
    clf = GradientBoostingClassifier(random_state=42).fit(features, labels)
    model = SelectFromModel(clf, prefit=True)
    #print('Selected features: ', model.get_support())
    return model.transform(features), model.get_support(True)

def select_k_best_features(features, labels):
    # setting k=20%
    k = int(.2 * features.shape[1])
    return SelectKBest(chi2, k=k).fit_transform(features, labels)

def select_features(features, labels, alg='none'):
    if alg=='none':
        return features
    elif alg=='model':
        return select_features_from_model(features, labels)
    elif alg=='kbest' :
        return select_k_best_features(features, labels)
    elif alg=='lr' :
        return select_lr_features(features, labels)
    elif alg=='xgboost':
        return select_boosting_features(features, labels)
    else:
        raise Exception('Unsupported algorithm: '+alg)
