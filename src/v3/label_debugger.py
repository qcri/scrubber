'''
Created on Mar 1, 2017

@author: hzhang0418
'''

'''
Given a dataset, the label debugger will interact with the analyst to detect suspicious labels. 

Each iteration it will return top-K suspicious labels to the analyst, get feedback from
the analyst, then perform incremental update internally.

'''

class LabelDebugger(object):
    
    def __init__(self):
        self.checked_indices = set()
    
    def detect(self, top_k):
        ranked_indices = self.detect_and_rank()
        
        tops = []
        count = 0        
        for index in ranked_indices:
            if index in self.checked_indices:
                continue
            
            tops.append(index)
            self.checked_indices.add(index)
            count += 1
            
            if count>=top_k:
                break
        
        return tops
    
    def detect_and_rank(self):
        print("Child class must override this function! ")
        pass
    
    def use_feedback(self, user_fb):
        print("Child class must override this function! ")
        pass
    
    
    