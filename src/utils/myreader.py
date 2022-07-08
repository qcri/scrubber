'''
Created on Jul 20, 2015

@author: hzhang0418
'''

import csv

def read_lines(input_file):
    '''
    Args:
        input_file (string): file name or path of the input file.
    
    Returns:
       lines (list[string]): all lines from the file, stored in a list. Each element is a line.
    
    Read all lines in a text file into a list of strings.
    
    Note: tailing EOLs are trimmed since it calls splitlines().
    '''
    
    with open(input_file) as f:
        lines = f.read().splitlines()
    
    return lines


def read_tuples(input_file, delim='\t'):
    '''
    Args:
        input_file (string): file name or path of the input file.
        delim (char): the delimiter between elements in a tuple. Default value is '\t'. 
    
    Returns:
        tuples (list[tuple]): list of tuples. Each line is one tuple.
    
    Read all lines in a text file into a list of tuples.
    Each tuple is a list of tokens split from its line with the given delimiter
    '''
    
    with open(input_file) as f:
        lines = f.read().splitlines()
        tuples = [ line.split(delim) for line in lines ]
    
    return tuples


def read_dicts(input_file, dict_delim='\t', key_value_delim='='):
    '''
    Args:
        input_file (string): file name or path of the input file.
        delim (char): the delimiter between elements in a tuple. Default value is '\t'. Each element is a key-value pair.
        key_vakye_delim (char): the delimiter between key and value. Default value is '='.
        
    Returns:
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
    
    Read all lines in a text file into a list of dictionaries
    Each dictionary is a dictionary of key-value pairs split from its line with given delimiter.
    The delimiter between key and value are also as given.
    '''
    
    with open(input_file) as f:
        lines = f.read().splitlines()
        
        dicts = []
        for line in lines:
            obj = {}
            pairs = line.split(dict_delim)
            for p in pairs:
                index = p.index(key_value_delim)
                key = p[:index]
                value = p[index+1:]
                obj[key] = value
            dicts.append(dict)
    
    return dicts


def read_dicts_with_projection(input_file, dict_delim='\t', key_value_delim='=', attributes=None):
    '''
    Args:
        input_file (string): file name or path of the input file.
        delim (char): the delimiter between elements in a tuple. Default value is '\t'. Each element is a key-value pair.
        key_vakye_delim (char): the delimiter between key and value. Default value is '='.
        attributes (list[string]): list of keys to be projected. Default value is None.
        
    Returns:
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
        
    Read all lines in a text file into a list of dictionaries
    Note only required attributes are saved into dictionaries while others are discarded
    Each dictionary is a dictionary of key-value pairs split from its line with given delimiter
    The delimiter between key and value are also as given
    '''
    
    if attributes is None: # no projection
        return read_dicts(input_file, dict_delim, key_value_delim)
    
    required = set(attributes) # the required attributes
    
    with open(input_file) as f:
        lines = f.read().splitlines()
        
        dicts = []
        for line in lines:
            obj = {}
            pairs = line.split(dict_delim)
            for p in pairs:
                index = p.index(key_value_delim)
                key = p[:index]
                if key in required:
                    value = p[index+1:]
                    obj[key] = value
            dicts.append(dict)
    
    return dicts


def read_csv_table(input_file, delim='\t'):
    '''
    Args:
        input_file (string): file name or path of the input file in CSV format.
        delim (char): the delimiter between elements in a tuple. Default value is '\t'. 
    
    Returns:
        header (list[string]): list of attribute names.
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
        
    Read table stored in a CSV file into a list of dictionaries.
    It returns the header as a list of string, and a list of dictionaries for all tuples.
    '''
    
    with open(input_file) as f:
        r = csv.reader(f, delimiter=delim)
        header = next(r)
    
        dicts = []
        for row in r:
            obj = {}
            for k, v in zip(header, row):
                obj[k] = v
            dicts.append(obj)
            
    return header, dicts


def read_csv_table_with_projection(input_file, delim='\t', attributes=None):
    '''
    Args:
        input_file (string): file name or path of the input file in CSV format.
        delim (char): the delimiter between elements in a tuple. Default value is '\t'. 
        attributes (list[string]): list of keys to be projected. Default value is None.
    
    Returns:
        header (list[string]): list of attribute names.
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
        
    Read the table stored in the given CSV file into a list of dictionaries.
    It returns the header as a list of string, and a list of dictionaries for all tuples.
    Note only required attributes are saved into dictionaries while others are discarded.
    '''
    
    if attributes is None:
        return read_csv_table(input_file, delim='\t')
    
    required = set(attributes) # required attributes
    
    with open(input_file) as f:
        r = csv.reader(f, delimiter=delim)
        header = next(r)
        
        dicts = []
        for row in r:
            obj = {}
            for a in attributes:
                obj[a] = row[ required[a] ]
            dicts.append(obj)
            
    return header, dicts
               
'''
    Functions below are for reading input files from walmart.
    The format of each line is a string for attribute 'id', then followed by key-value pairs.
    The delimiter is '\t'. 
'''                

def read_walmart_products(input_file):
    '''
    Args:
        input_file (string): file name or path of the input file.
        
    Returns:
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
        
    Read all Walmart products in a text file into a list of dictionaries.
    Each line is a product, where attributes are separated by tab key.
    The first attribute is 'id', while remaining attributes are in the format of key=value.
    '''
    
    with open(input_file) as f:
        lines = f.read().splitlines()
        
        dicts = []
        for line in lines:
            obj = {}
            pairs = line.split('\t')
            obj['id'] = pairs[0]
            for p in pairs[1:]:
                index = p.index('=')
                key = p[:index]
                value = p[index+1:]
                obj[key] = value
            dicts.append(obj)
    
    return dicts


def read_walmart_products_with_projection(input_file, attributes=None):
    '''
    Args:
        input_file (string): file name or path of the input file.
        attributes (list[string]): list of keys to be projected. Default value is None.
        
    Returns:
        dicts (list[dict]): list of dictionaries. Each line is one dictionary.
        
    Read all Walmart products in a text file into a list of dictionaries
    Note only required attributes are saved into dictionaries while others are discarded
    Each line is a product, where attributes are separated by tab key
    The first attribute is ID, while remaining attributes are in the format of key=value
    '''
    
    if attributes is None:
        return read_walmart_products(input_file)
    
    required = set(attributes) # the required attributes
    
    with open(input_file) as f:
        lines = f.read().splitlines()
        
        dicts = []
        for line in lines:
            obj = {}
            pairs = line.split('\t')
            obj['id'] = pairs[0]
            for p in pairs[1:]:
                index = p.index('=')
                key = p[:index]
                value = p[index+1:]
                if key in required:
                    obj[key] = value
            dicts.append(obj)
    
    return dicts        