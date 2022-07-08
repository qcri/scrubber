'''
Created on Mar 3, 2019

@author: hzhang0418
'''

from operator import itemgetter

'''
Greedy Vertex Cover
'''

def greedy_vertex_cover(vertex2neighbors):
    
    print("Greedy Vertex Cover...")
    
    v2count = { v:len(neighbors) for v, neighbors in vertex2neighbors.items() }
    
    selected = set()
    tmp = []
    
    while(True):
        # select current best
        vc = [ (v,c) for v, c in v2count.items() if c>0]
        if len(vc)==0:
            break
        vc_sorted = sorted(vc, key=itemgetter(1,0), reverse=True)
        best = vc_sorted[0][0]
        selected.add(best)
        tmp.append(best)
        # remove edges
        for v in vertex2neighbors[best]:
            if v in selected:
                continue
            v2count[v] -= 1
            
        v2count[best] = 0
        
    return tmp
        