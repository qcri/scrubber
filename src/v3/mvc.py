'''
Created on Mar 3, 2017

@author: hzhang0418
'''


# Hopcroft-Karp bipartite max-cardinality matching and max independent set
# David Eppstein, UC Irvine, 27 Apr 2002

def bipartiteMatch(graph):
    '''
    Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.  The output is a triple (M,A,B) where M is a
    dictionary mapping members of V to their matches in U, A is the part
    of the maximum independent set in U, and B is the part of the MIS in V.
    The same object may occur in both U and V, and is treated as two
    distinct vertices if this happens.
    '''

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break
    
    while 1:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            newLayer = {}
            for u in layer:
                for v in graph[u]:
                        if v not in preds:
                            newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None

            return (matching, list(pred), list(unlayered))

        # recursively search backward through layers to find alternating paths
        # recursion returns true if found path, false otherwise
        def recurse(v):
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return 1
            return 0

        for v in unmatched: recurse(v)

# Find a minimum vertex cover
def min_vertex_cover(left_v, right_v):
    '''
    Use the Hopcroft-Karp algorithm to find a maximum
    matching or maximum independent set of a bipartite graph.
    Next, find a minimum vertex cover by finding the 
    complement of a maximum independent set.

    The function takes as input two dictionaries, one for the
    left vertices and one for the right vertices. Each key in 
    the left dictionary is a left vertex with a value equal to 
    a list of the right vertices that are connected to the key 
    by an edge. The right dictionary is structured similarly.

    The output is a dictionary with keys equal to the vertices
    in a minimum vertex cover and values equal to lists of the 
    vertices connected to the key by an edge.

    For example, using the following simple bipartite graph:

    1000 2000
    1001 2000

    where vertices 1000 and 1001 each have one edge and 2000 has 
    two edges, the input would be:

    left = {1000: [2000], 1001: [2000]}
    right = {2000: [1000, 1001]}

    and the ouput or minimum vertex cover would be:

    {2000: [1000, 1001]}

    with vertex 2000 being the minimum vertex cover.

    The code can also generate a bipartite graph with an arbitrary
    number of edges and vertices, write the graph to a file, and 
    read the graph and convert it to the appropriate format.
    '''

    data_hk = bipartiteMatch(left_v)
    left_mis = data_hk[1]
    right_mis = data_hk[2]
    mvc = left_v.copy()
    mvc.update(right_v)

    #print(left_v)
    #print(right_v)
    #print(left_mis)
    #print(right_mis)

    for v in left_mis:
        del(mvc[v])
    for v in right_mis:
        del(mvc[v])

    return mvc