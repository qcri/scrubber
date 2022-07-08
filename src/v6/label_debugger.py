'''
Created on Mar 6, 2019

@author: hzhang0418
'''

import time

import feature_selection as fs

import v6.fpfn_irf as irf
import v6.fpfn as fpfn
import v6.sort_probing as sp
import v6.cleanlab_detector as cl

import combiner

class LabelDebugger(object):
    
    def __init__(self, features, labels, params):
        self.labels = labels
        self.features = features #fs.select_features(features, labels, params.get('fs_alg', 'none'))
        #print('Num features after selection: ', self.features.shape)
        self.max_list_len = params.get('max_list_len', 20)
        self._start_detectors(params)
        
        self.iter_count = 0
        # indices whose label has been verified by the analyst
        self.verified_indices = set()
        # verified error indices
        self.match_error_indices = set()
        self.nonmatch_error_indices = set()

    def _start_detectors(self, params):
        self.detectors = []
        detector_types = params.get('detectors', 'both')
        
        if detector_types=='fpfn':
            #det = irf.FPFN_IRF(self.features, self.labels, params)
            det = fpfn.FPFN(self.features, self.labels, params)
            #det.set_num_cores(1)
            self.detectors.append(det)
            self.ndetectors = 1
        elif detector_types=='mono':
            det = sp.SortProbing(self.features, self.labels, params)
            self.detectors.append(det)
            self.ndetectors = 1
        elif detector_types=='cleanlab': ## mayuresh: cl detector
            det = cl.CL(self.features, self.labels, params)
            self.detectors.append(det)
            self.ndetectors = 1
        elif detector_types=='both':
            det = irf.FPFN_IRF(self.features, self.labels, params)
            #det = fpfn.FPFN(self.features, self.labels, params)
            #params['confusion'] = True
            #det = cl.CL(self.features, self.labels, params) ## mayuresh: For accuracy using CL with confusion=True, for lower latency, use FPFN_IRF
            #det.set_num_cores(1)
            self.detectors.append(det)
            det = sp.SortProbing(self.features, self.labels, params)
            #det.set_num_cores(1)
            self.detectors.append(det)
            self.ndetectors = 2
    
    
    def find_suspicious_labels(self, top_k):
        self.iter_count += 1
        
        self.ranked_lists = []
        for det in self.detectors:
            start = time.clock()
            tmp = det.detect_and_rank()
            ##TEMP: filter false non-matches
            #print('detected before: ', len(tmp))
            #tmp = [t for t in tmp if self.labels[t]==1]
            #print('detected false matches: ', len(tmp))
            end = time.clock()
            #if self.iter_count==1:
            #    print('First iteration time:', end-start)
            self.ranked_lists.append([t for t in tmp if t not in self.verified_indices][:self.max_list_len] )
        
        if self.ndetectors == 1:
            top_suspicious_indices = self.ranked_lists[0][:top_k]
        #elif self.ndetectors == 2:
        #    top_suspicious_indices = combiner.combine_two_lists(ranked_lists[0], ranked_lists[1])[:top_k]
        else:
            top_suspicious_indices = combiner.combine_all_lists(self.ranked_lists)[:top_k]
            
        self.verified_indices.update(top_suspicious_indices)
            
        return top_suspicious_indices
    
    
    def correct_labels(self, index2correct_label):
        error_index2correct_label = {}
        for index, label in index2correct_label.items():
            if label!=self.labels[index]:
                error_index2correct_label[index] = label
            
            self.labels[index] = label
        
        for detector in self.detectors:
            #start = time.clock()
            detector.use_feedback(error_index2correct_label)
            #end = time.clock()
            #print('Incremental update time:', end-start)
            
    
    # it's used to analyze current iteration
    # must be called before correct_labels
    def analyze(self, index2correct_label):
        num_errors = 0
        error_indices = []
        error_indices_matches = [] # labels mislabeled as nonmatches, analyst corrects to matches
        error_indices_nonmatches = [] # labels mislabeled as matches, analyst corrects to nonmatches
        for index, label in index2correct_label.items():
            if label!=self.labels[index]:
                num_errors += 1
                error_indices.append(index)
                if label == 1:
                     error_indices_matches.append(index)
                else:
                     error_indices_nonmatches.append(index)
        #print('Added match errors: ', len(error_indices_matches), ' non-match errors: ', len(error_indices_nonmatches))
        self.match_error_indices.update(error_indices_matches)
        self.nonmatch_error_indices.update(error_indices_nonmatches)
       
        det_error_poses = []
        for rlist in self.ranked_lists:
            index_pos = [] # pair of (error_index, pos_in_list)
            det_error_count = 0 # number of errors detected
            for index in error_indices:
                found = False
                for pos, v in enumerate(rlist):
                    if v == index:
                        found = True
                        index_pos.append( (index, pos) )
                        det_error_count += 1
                        break
                if not found:
                    index_pos.append( (index, -1) )
            det_error_poses.append( (det_error_count, index_pos) )
            
        return self.iter_count, num_errors, error_indices, error_indices_matches, det_error_poses    

    ## mayuresh: Train classifier one type of errors identified, call it after correcting labels
    def explain_errors(self, match_errors=False):
        if match_errors:
           indices = [i for i in self.match_error_indices]
        else:
           indices = [i for i in self.nonmatch_error_indices]
        error_count = len(indices)
        print('Explaining errors size: ', error_count, ' indices: ', indices)
        if match_errors:
           indices.extend([i for i in self.verified_indices if self.labels[i]==0])
        else:
           indices.extend([i for i in self.verified_indices if self.labels[i]==1]) 
        print('Total candidates considered after adding true verified labels size: ', len(indices))
        if error_count == 0 or len(indices) == error_count:
           print('Only one class of data is available, no explanation possible!')
           return

        ## Extract features and labels
        explain_features = [f for i, f in enumerate(self.features) if i in indices]
        explain_labels = [l for i, l in enumerate(self.labels) if i in indices]
        #print('Labels used are: ', explain_labels)

        ## Use LR to find important features
        sf, sfi = fs.select_features(explain_features, explain_labels, 'lr')
        print('Selected important features are: ', sfi)

    def set_num_cores(self, num_cores):
        for det in self.detectors:
            det.set_num_cores(num_cores)          
        
