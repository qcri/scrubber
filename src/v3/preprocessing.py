'''
Created on Feb 16, 2017

@author: hzhang0418

data preprocessing:
- read the dataset of features
- convert features into proper data structures for other modules
- read the dataset with golden labels

'''

import os
import numpy as np
import pandas as pd

class Preprocessing:
    
    def __init__(self, params):
        self.params = params
        
        # dataset
        self.dataset_name = params['dataset_name']
        # base dir
        self.basedir = params['basedir']
        
    
    def read_features(self):
        '''
        read the file for feature vectors
        
        return
            header: list of header attributes
            feature_vectors: list of feature vectors, each vector is a tuple of strings
        '''
        
        feature_vector_file = os.path.join(self.basedir, self.params['hpath'])
        
        f = open(feature_vector_file)
        lines = f.read().splitlines()
        f.close()
        
        k=0
        while lines[k][0]=='#':
            k = k+1
            continue
        
        tmp = lines[k]
        header = tmp.split(',')
        tuples = [ line.split(',') for line in lines[k+1:] ]
        
        print("Number of headers: ", len(header))
        print("Number of feature vectors: ", len(tuples))
        
        return header, tuples
        
        
    def read_golden(self):
        '''
        read the file with golden labels
        
        return
            pair2golden = {} # map each pair to its golden label
        '''
        
        pair2golden = {} # map each pair to its golden label
        
        golden_file = os.path.join(self.basedir, 'golden.csv')
        
        f = open(golden_file)
        lines = f.read().splitlines()
        f.close()
        
        k=0
        while lines[k][0]=='#':
            k = k+1
            continue
        
        tmp = lines[k]
        header = tmp.split(',')
        
        header2col = {}
        for i,h in enumerate(header):
            header2col[h] = i
        
        for line in lines[k+1:]:
            t = line.split(',') 
        
            id1= t[ header2col['ltable.id'] ]
            id2 = t[ header2col['rtable.id'] ]
            label = int(t[ header2col['golden'] ])
            pair = (id1, id2)
            
            pair2golden[pair] = label
            
        return pair2golden
            
    def convert_from_dataframe(self, feature_table, feature_names):
        '''
        index2pair = {} # map index to the corresponding pair
        features = 2-d numpy array # each row is a feature vector
        labels = 1-d numpy array # each row is a label
        '''
    
        nfeatures = len(feature_names)
    
        index2pair = {} # map index to the corresponding pair
        features = np.zeros( (len(feature_table), nfeatures), dtype=np.float64 )
        labels = np.zeros( len(feature_table), dtype=np.int )
        
        row_index = 0
        
        for _, row in feature_table.iterrows():
            id1= row['ltable.id']
            id2 = row['rtable.id']
            label = int(row['label'])
            pair = (id1, id2)
            
            index2pair[row_index] = pair
            
            labels[row_index] = label
            features[row_index] = np.array([ float( "{0:.2f}".format(float(row[name]))) for name in feature_names ]) 
            
            row_index += 1
            
        return index2pair, features, labels
        
    
    def convert_from_tuples(self, feature_header, feature_tuples, feature_names):
        '''
        index2pair = {} # map index to the corresponding pair
        features = 2-d numpy array # each row is a feature vector
        labels = 1-d numpy array # each row is a label
        '''
        
        feature_map = {}
        for i, attribute in enumerate(feature_header):
            feature_map[attribute] = i
    
        nfeatures = len(feature_names)
    
        index2pair = {} # map index to the corresponding pair
        features = np.zeros( (len(feature_tuples), nfeatures), dtype=np.float64 )
        labels = np.zeros( len(feature_tuples), dtype=np.int )
        
        for row_index, t in enumerate(feature_tuples):
            id1= t[ feature_map['ltable.id'] ]
            id2 = t[ feature_map['rtable.id'] ]
            label = int(t[ feature_map['label'] ])
            pair = (id1, id2)
            
            index2pair[row_index] = pair
            
            labels[row_index] = label
            features[row_index] = np.array([ float( "{0:.2f}".format( float( t[ feature_map[name] ] ) ) ) for name in feature_names ]) 
            
        return index2pair, features, labels
        
    
    def convert_for_mono(self, feature_header, feature_tuples, feature_names):
        '''
        index2pair = {} # map index to the corresponding pair
        feature_vector_map = {} # map index to the feature vector of the corresponding pair
        nfeatures # number of features
        match_indices = [] # list of indices of match pairs
        nonmatch_indices = [] # list of indices of nonmatch pairs
        index2label = {} # map index to the label of the corresponding pair
        '''
        
        feature_map = {}
        for i, attribute in enumerate(feature_header):
            feature_map[attribute] = i
    
        nfeatures = len(feature_names)
    
        index2pair = {} # map index to the corresponding pair
        feature_vector_map = {} # map index to the feature vector of the corresponding pair
        match_indices = [] # list of indices of match pairs
        nonmatch_indices = [] # list of indices of nonmatch pairs
        index2label = {} # map index to the label of the corresponding pair
        
        for row_index, t in enumerate(feature_tuples):
            id1= t[ feature_map['ltable.id'] ]
            id2 = t[ feature_map['rtable.id'] ]
            label = int(t[ feature_map['label'] ])
            pair = (id1, id2)
            
            index2pair[row_index] = pair
            
            if label == 1:
                match_indices.append( row_index )
            elif label == 0:
                nonmatch_indices.append( row_index )
                
            feature_vector_map[row_index] = [ float( "{0:.2f}".format( float( t[ feature_map[name] ] ) ) ) for name in feature_names ]
                
            index2label[row_index] = label
            
        print("Num of match:", len(match_indices))
        print("Num of nonmatch:", len(nonmatch_indices))
        print("Total:", len(index2label)) 
        
        return index2pair, feature_vector_map, match_indices, nonmatch_indices, index2label, nfeatures  
