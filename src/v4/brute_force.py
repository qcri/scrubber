'''
Updated on Feb 17, 2019

@author: hzhang0418

Naive algorithm to compute all pairs of match/nonmatch that violate the monotonicity property in at least one feature

'''
from v4.mono import Mono

class BruteForce(Mono):
    
    def __init__(self, features, labels, min_con_dim, use_mvc):
        super(BruteForce, self).__init__(features, labels, min_con_dim, use_mvc)
    
    def get_inconsistency_indices(self):
        for mi in self.match_indices:
            match_features = self.feature_vector_map[mi]
            incons_indices = []
            for ni in self.nonmatch_indices:
                inconsistent = self.compare_features(match_features, self.feature_vector_map[ni], self.min_con_dim)
                if inconsistent == True:
                    incons_indices.append(ni)
                    
            if len(incons_indices)>0:
                self.index2incons[mi] = incons_indices
                for ni in incons_indices:
                    if ni in self.index2incons:
                        self.index2incons[ni].append(mi)
                    else:
                        self.index2incons[ni] = [mi]

        return self.index2incons