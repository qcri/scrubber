'''
Created on Aug 23, 2016

@author: haojun

updated to py_entitymatching

'''

import py_entitymatching as em

# Sample matched and non-matched pairs 
# if number of matched pairs is no more than half of the required number of samples, then all matched pairs will be included
def sample(table_G, num_samples):
    pos_records = table_G[table_G.apply(lambda r: r['label']==1, 1)].copy()
    half = num_samples/2
    
    if len(pos_records)>half:
        pos_sampled = pos_records.sample(half)
    else:
        pos_sampled = pos_records
    
    neg_records = table_G[table_G.apply(lambda r: r['label']==0, 1)].copy()
    neg_sampled = neg_records.sample(num_samples - len(pos_sampled))
    
    return pos_sampled.append(neg_sampled)

# find corresponding feature vectors
def retrieve_feature_vectors(table_H, table_G_sampled):
    rows = table_G_sampled['_id'].tolist()
    table_H_sampled = table_H[ table_H['_id'].isin(rows) ].copy()
    return table_H_sampled


'''
def prepare_restaurant():
    # read tables, compute feature table
    table_A, table_B, table_G, table_features, dataset_name = restaurant.get_data_set()
   
    hpath = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/restaurant/H.csv"
    table_H = em.read_csv_metadata(hpath, ltable=table_A, rtable=table_B)
    
    # sample 500 from golden
    sampled_g_path = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/restaurant/SampledG_500.csv"
    table_G_sampled = sample(table_G, 500)
    table_G_sampled.to_csv(sampled_g_path, index=False)

    # find their feature vectors from table_H
    sampled_h_path = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/restaurant/SampledH_500.csv"
    table_H_sampled = retrieve_feature_vectors(table_H, table_G_sampled)
    table_H_sampled.to_csv(sampled_h_path, index=False) 
    
    return None

def prepare_product():
    # read tables, compute feature table
    table_A, table_B, table_G, table_features, dataset_name = product.get_data_set()
    
    # sample 500 from golden
    sampled_g_path = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/walmart_amazon_1/SampledG_500.csv"
    table_G_sampled = sample(table_G, 500)
    table_G_sampled.to_csv(sampled_g_path, index=False)

    # find their feature vectors from table_H
    hpath = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/walmart_amazon_1/SampledH_500.csv"
    compute_feature_vector(table_G, table_features, hpath)
    return None

def prepare_cora():
    # read tables, compute feature table
    table_A, table_B, table_G, table_features, dataset_name = cora.get_data_set()
   
    hpath = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/cora/H.csv"
    table_H = em.read_csv_metadata(hpath, ltable=table_A, rtable=table_B)
    
    # sample 500 from golden
    sampled_g_path = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/cora/SampledG_500.csv"
    table_G_sampled = sample(table_G, 500)
    table_G_sampled.to_csv(sampled_g_path, index=False)

    # find their feature vectors from table_H
    sampled_h_path = "/scratch/workspace2/Magellan/Magellan-0.1/magellan/datasets/cora/SampledH_500.csv"
    table_H_sampled = retrieve_feature_vectors(table_H, table_G_sampled)
    table_H_sampled.to_csv(sampled_h_path, index=False) 
    
    return None
'''

def get_restaurant(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes1['addr'] = 'str_bt_5w_10w'
    atypes2 = em.get_attr_types(table_B)
    atypes2['addr'] = 'str_bt_5w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_product(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes1['title'] = 'str_gt_10w'
    atypes2 = em.get_attr_types(table_B)
    atypes2['title'] = 'str_gt_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_cora(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('class','class'))
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_cora_large(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('class','class'))
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def compute_feature_vector(table_G, table_features, hpath): 
    table_H = em.extract_feature_vecs(table_G, attrs_before=None, feature_table=table_features, attrs_after=["label"])
    table_H.fillna(0, inplace=True)
    
    tmp = ['_id', 'ltable.id', 'rtable.id', 'label']
    
    # normalize
    for col in table_H:
        if col in tmp:
            continue
        
        # normalize column
        cmax, cmin = table_H[col].max(), table_H[col].min()
        
        if cmax>1 or cmin<0: 
            print (col, cmax, cmin)
        
            diff = cmax - cmin
            if diff<0.001: # move them to around 0.5
                t = 0.5 - (cmax+cmin)/2
                table_H[col] = table_H[col] + t
            else:
                table_H[col] = (table_H[col] - cmin)/diff
            
        if col.endswith('dist') or col.endswith('sw') or col.endswith('nmw'):
            table_H[col] = 1.0 - table_H[col]
    
    table_H.to_csv(hpath, index=False)
    
    
def get_beer(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_bike(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes1['color'] = 'str_eq_1w'
    atypes2 = em.get_attr_types(table_B)
    atypes2['color'] = 'str_eq_1w'
    match_c = em.get_attr_corres(table_A,table_B)
    #match_c['corres'].remove(('id','id'))
    
    match_c['corres'].remove(('color','color'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features
    
def get_books1(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes1['authors'] = 'str_bt_1w_5w'
    atypes1['edition'] = 'str_bt_1w_5w'
    atypes2 = em.get_attr_types(table_B)
    atypes2['authors'] = 'str_bt_1w_5w'
    atypes2['edition'] = 'str_bt_1w_5w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_B['edition'] = table_B['edition'].astype(str)
    
    #match_c['corres'].remove(('authors', 'authors'))
    match_c['corres'].remove(('edition', 'edition'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_citations(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes1['author'] = 'str_bt_1w_5w'
    atypes1['journal'] = 'str_bt_1w_5w'
    atypes1['author'] = 'str_bt_1w_5w'
    atypes1['volume'] = 'str_eq_1w'
    atypes1['pages'] = 'str_bt_1w_5w'
    atypes1['booktitle'] = 'str_gt_10w'
    atypes1['number'] = 'str_bt_1w_5w'
    atypes2 = em.get_attr_types(table_B)
    atypes2['author'] = 'str_bt_1w_5w'
    atypes2['journal'] = 'str_bt_1w_5w'
    atypes2['author'] = 'str_bt_1w_5w'
    atypes2['volume'] = 'str_eq_1w'
    atypes2['pages'] = 'str_bt_1w_5w'
    atypes2['booktitle'] = 'str_gt_10w'
    atypes2['number'] = 'str_bt_1w_5w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_B['volume'] = table_B['volume'].astype(str)
    
    match_c['corres'].remove(('volume', 'volume'))
    match_c['corres'].remove(('number', 'number'))
    match_c['corres'].remove(('pages', 'pages'))
    match_c['corres'].remove(('journal', 'journal'))
    match_c['corres'].remove(('booktitle', 'booktitle'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_movies1(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    del table_A['RatingValue']
    del table_B['RatingValue']
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['RatingValue'] = 'str_eq_1w'
    atypes1['Genre'] = 'str_bt_1w_5w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['RatingValue'] = 'str_eq_1w'
    atypes2['Genre'] = 'str_bt_1w_5w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    #table_A['RatingValue'] = table_A['RatingValue'].astype(str)
    match_c['corres'].remove(('Genre', 'Genre'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_restaurants4(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_citations_large(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    del table_A['month']
    del table_B['month']
    
    del table_A['journal']
    del table_B['journal']
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['journal'] = 'str_bt_1w_10w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['journal'] = 'str_bt_1w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_data(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # remove useless columns 
    # del table_A['column']
    # del table_B['column']
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    # do type conversion (to string)
    # #table_A['column'] = table_A['column'].astype(str)
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def retrieve(dataset_name, apath, bpath, gpath):
    
    if dataset_name == 'restaurant':
        return get_restaurant(apath, bpath, gpath)
    elif dataset_name == 'cora':
        return get_cora(apath, bpath, gpath)
    elif dataset_name == 'cora_large':
        return get_cora_large(apath, bpath, gpath)
    elif dataset_name == 'product':
        return get_product(apath, bpath, gpath)
    elif dataset_name == 'beer':
        return get_beer(apath, bpath, gpath)
    elif dataset_name == 'bike':
        return get_bike(apath, bpath, gpath)
    elif dataset_name == 'books1':
        return get_books1(apath, bpath, gpath)
    elif dataset_name == 'citations':
        return get_citations(apath, bpath, gpath)
    elif dataset_name == 'movies1':
        return get_movies1(apath, bpath, gpath)
    elif dataset_name == 'restaurants4':
        return get_restaurants4(apath, bpath, gpath)
    elif dataset_name == 'citations_large':
        return get_citations_large(apath, bpath, gpath)
    elif dataset_name == 'citations_20k':
        return get_citations_large(apath, bpath, gpath)
    elif dataset_name == 'citations_50k':
        return get_citations_large(apath, bpath, gpath)
    elif dataset_name == 'citations_100k':
        return get_citations_large(apath, bpath, gpath)
    
    elif dataset_name == 'clothing':
        return get_clothing(apath, bpath, gpath)
    elif dataset_name == 'home':
        return get_home(apath, bpath, gpath)
    elif dataset_name == 'electronics':
        return get_electronics(apath, bpath, gpath)
    elif dataset_name == 'tools':
        return get_tools(apath, bpath, gpath)
    
    elif dataset_name == 'trunc_clothing':
        return get_trunc_clothing(apath, bpath, gpath)
    elif dataset_name == 'trunc_home':
        return get_trunc_home(apath, bpath, gpath)
    elif dataset_name == 'trunc_electronics':
        return get_trunc_electronics(apath, bpath, gpath)
    elif dataset_name == 'trunc_tools':
        return get_trunc_tools(apath, bpath, gpath)
    
    elif dataset_name == 'citations_new':
        return get_citations_new(apath, bpath, gpath)
    
    # datasets with only matches
    elif dataset_name == 'walmart_amazon':
        return get_walmart_amazon(apath, bpath, gpath)
    elif dataset_name == 'abt_buy':
        return get_abt_buy(apath, bpath, gpath)
    elif dataset_name == 'dblp_acm':
        return get_dblp_acm(apath, bpath, gpath)
    elif dataset_name == 'dblp_googlescholar':
        return get_dblp_googlescholar(apath, bpath, gpath)
    elif dataset_name == 'amazon_google':
        return get_amazon_google(apath, bpath, gpath)
    elif dataset_name == 'fodors_zagats':
        return get_fodors_zagats(apath, bpath, gpath)
    
    # cora new
    elif dataset_name == 'cora_new':
        return get_cora_new(apath, bpath, gpath)
    
    # songs 1m
    elif dataset_name == 'songs_1m':
        return get_songs_1m(apath, bpath, gpath)
    # songs small
    elif dataset_name == 'songs_small':
        return get_songs_small(apath, bpath, gpath)
    
    # citations_500k
    elif dataset_name == 'citations_500k':
        return get_citations_500k(apath, bpath, gpath)
    
    # umerics
    elif dataset_name == 'umetrics_300':
        return get_umetrics_300(apath, bpath, gpath)
    elif dataset_name == 'umetrics_400':
        return get_umetrics_400(apath, bpath, gpath)
    
    else:
        raise Exception('Dataset not found!')
        return get_data(apath, bpath, gpath)
    

def fix_cora_labels(table_G, table_H, table_cora):
    
    table_G_fixed = table_G.copy()
    table_H_fixed = table_H.copy()
    
    tmp = table_cora[['id', 'correct_class']]
    
    # 
    id2class = {}
    
    for t in tmp.itertuples(index=False):
        id2class[ t[0] ] = t[1]
        
    
    num_errors = 0    
    for index, row in table_G.iterrows():
        id1 = row['ltable.id']
        id2 = row['rtable.id']
        label = row['label']
        if id1 in id2class and id2 in id2class:
            if id2class[id1] == id2class[id2]:
                table_G_fixed.set_value(index, 'label', 1)
                if int(label) != 1:
                    num_errors += 1
            else:
                table_G_fixed.set_value(index, 'label', 0)
                if int(label) != 0:
                    num_errors += 1
        else:
            table_G_fixed.set_value(index, 'label', 0)
            if int(label) != 0:
                num_errors += 1
            
    for index, row in table_H.iterrows():
        id1 = row['ltable.id']
        id2 = row['rtable.id']
        label = row['label']
        if id1 in id2class and id2 in id2class:
            if id2class[id1] == id2class[id2]:
                table_H_fixed.set_value(index, 'label', 1)
            else:
                table_H_fixed.set_value(index, 'label', 0)
        else:
            table_H_fixed.set_value(index, 'label', 0)
            
    print("Number of errors in cora: ", num_errors)
    
    return table_G_fixed, table_H_fixed


'''
Data sets from Han
'''

def get_clothing(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # remove useless columns 
    # del table_A['column']
    # del table_B['column']
    
    table_A.drop(columns=['assembled_product_height', 'assembled_product_length'], inplace=True)
    table_B.drop(columns=['assembled_product_height', 'assembled_product_length'], inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    # do type conversion (to string)
    # #table_A['column'] = table_A['column'].astype(str)
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def get_home(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # remove useless columns 
    # del table_A['column']
    # del table_B['column']
    
    table_A.drop(columns=['assembled_product_width', 'assembled_product_length'], inplace=True)
    table_B.drop(columns=['assembled_product_width', 'assembled_product_length'], inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    # do type conversion (to string)
    # #table_A['column'] = table_A['column'].astype(str)
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def get_electronics(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # remove useless columns 
    # del table_A['column']
    # del table_B['column']
    
    cols_to_be_removed = ['assembled_product_width', 'assembled_product_length', 'assembled_product_height', 'depth', 
                          'number_of_holes', 'number_of_rack_units', 'size']
    
    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    # do type conversion (to string)
    # #table_A['column'] = table_A['column'].astype(str)
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def get_tools(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # remove useless columns 
    # del table_A['column']
    # del table_B['column']
    
    cols_to_be_removed = ['assembled_product_width', 'assembled_product_length', 'assembled_product_height', 'alphanumeric_character'] 
    #['assembled_product_width', 'assembled_product_length', 'assembled_product_height', 'depth', 
    #                      'number_of_holes', 'number_of_rack_units', 'size']
    
    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    # do type conversion (to string)
    # #table_A['column'] = table_A['column'].astype(str)
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


'''
Truncated datasets
'''

def get_trunc_clothing(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features
    

def get_trunc_home(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def get_trunc_electronics(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


def get_trunc_tools(apath, bpath, gpath):
    '''
    A template for new dataset
    '''
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    # correct the type of some column
    # atypes1['column'] = 'type'
    atypes2 = em.get_attr_types(table_B)
    # correct the type of some column
    # atypes2['column'] = 'type'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

'''
New Citations Dataset
'''
def get_citations_new(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['journal', 'month', 'publication_type']
    
    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['journal'] = 'str_bt_1w_10w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['journal'] = 'str_bt_1w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


'''
    Datasets with only matches
'''
def get_walmart_amazon(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['modelno', 'price']
    
    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_abt_buy(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    #cols_to_be_removed = []

    #table_A.drop(columns=cols_to_be_removed, inplace=True)
    #table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_dblp_acm(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    #cols_to_be_removed = []

    #table_A.drop(columns=cols_to_be_removed, inplace=True)
    #table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_dblp_googlescholar(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    #cols_to_be_removed = []

    #table_A.drop(columns=cols_to_be_removed, inplace=True)
    #table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_amazon_google(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    #cols_to_be_removed = []

    #table_A.drop(columns=cols_to_be_removed, inplace=True)
    #table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_fodors_zagats(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    #cols_to_be_removed = []

    #table_A.drop(columns=cols_to_be_removed, inplace=True)
    #table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_cora_new(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('class','class'))
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_songs_1m(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['artist_familiarity','artist_hotttnesss']

    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_songs_small(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['artist_familiarity','artist_hotttnesss']

    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    atypes2 = em.get_attr_types(table_B)
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

'''
Citations_500k Dataset
'''
def get_citations_500k(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['journal', 'month', 'publication_type']
    
    table_A.drop(columns=cols_to_be_removed, inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['journal'] = 'str_bt_1w_10w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['journal'] = 'str_bt_1w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features


'''
UMERICS Dataset
'''
def get_umetrics_300(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['AwardNumber', 'AccessionNumber']
    
    table_A.drop(columns=['AwardNumber'], inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['journal'] = 'str_bt_1w_10w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['journal'] = 'str_bt_1w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features

def get_umetrics_400(apath, bpath, gpath):
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')
    
    cols_to_be_removed = ['AwardNumber', 'AccessionNumber']
    
    table_A.drop(columns=['AwardNumber'], inplace=True)
    table_B.drop(columns=cols_to_be_removed, inplace=True)
    
    # compute feature table
    match_t = em.get_tokenizers_for_matching(q = [3,5])
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(table_A) 
    #atypes1['journal'] = 'str_bt_1w_10w'
    atypes2 = em.get_attr_types(table_B)
    #atypes2['journal'] = 'str_bt_1w_10w'
    match_c = em.get_attr_corres(table_A,table_B)
    match_c['corres'].remove(('id','id'))
    
    table_features = em.get_features(table_A, table_B, atypes1, atypes2, match_c, match_t, match_s)
    
    return table_A, table_B, table_features