'''
Created on Jul 20, 2015

@author: hzhang0418
'''

import os

def write_lines(lines, output_file):
    '''
    Args:
        lines (list[string]): list of strings to be written. Each element will be one line.
        output_file (string): file name or path of the output file.
    
    Write all strings in a list into the given file, adding newline at the end of each string.
    '''
    
    f = open(output_file, 'w')
    for line in lines:
        f.write(str(line))
        f.write(os.linesep)
    f.close()
    
    
def write_tuples(tuples, output_file, delim='\t'):
    '''
    Args:
        tuples (list[tuple]): list of tuples to be written. Each tuple will be one line.
        output_file (string): file name or path of the output file.
        delim (char): the delimiter character between elements in each tuple.
        
    Write a list of tuples into the given file, adding newline at the end of each tuple.
    Elements in each tuple are separated by the given delimiter.
    '''
    
    f = open(output_file, 'w')
    for t in tuples:
        f.write(str(t[0]))
        for v in t[1:]: 
            f.write(delim)
            f.write(str(v))
        f.write(os.linesep)
    f.close()
    
def write_tuples_as_simple_csv_table(tuples, header, output_file, delim='\t'):
    '''
    Args:
        tuples (list[tuple]): list of tuples to be written. Each tuple will be one line.
        header (list[string]): list of attributes. Number of attributes should match length of each tuple.
        output_file (string): file name or path of the output file.
        delim (char): the delimiter character between elements in each tuple.
        
    Write a list of tuples into the given file, adding newline at the end of each tuple.
    Elements in each tuple are separated by the given delimiter.
    '''
    
    f = open(output_file, 'w')
    
    f.write(str(header[0]))
    tail = header[1:]
    for h in tail:
        f.write(delim)
        f.write(str(h))
    f.write(os.linesep)
    
    for t in tuples:
        f.write(str(t[0]))
        for v in t[1:]: 
            f.write(delim)
            f.write(str(v))
        f.write(os.linesep)
    f.close()
    
    
def write_dicts_as_key_value_pairs(dicts, output_file, dict_delim='\t', key_value_delim='='):
    '''
    Args:
        dicts (list[dict]): list of dictionaries to be written. Each dictionary will be one line.
        output_file (string): file name or path of the output file.
        dict_delim (char): the delimiter character between key-value pairs in each dictionary.
        key_value_delim (char): the delimiter character between key and value of a pair.
        
    Write a list of dictionaries into the given file, adding newline at the end of each dictionary.
    Also, add given delimiter between key-value pairs in a dictionary.
    Note: Dictionaries may have different number of pairs. There is no guarantee about the ordering of pairs.
    '''
    
    f = open(output_file, 'w')
    for t in dicts:
        tmp = t.items()
        count = 0
        for k, v in tmp: 
            count = count + 1
            f.write(str(k))
            f.write(key_value_delim)
            f.write(str(v))
            if count < len(tmp):
                f.write(dict_delim)
        f.write(os.linesep)
    f.close()
    
    
def write_dicts_as_simple_csv_table(dicts, header, output_file, delim='\t'):
    '''
    Args:
        dicts (list[dict]): list of dictionaries to be written. Each dictionary will be one line.
        header (list[string]): list of attributes.
        output_file (string): file name or path of the output file.
        delim (char): the delimiter character between key-value pairs in each dictionary.
        
    Write header, and all dictionaries in a list into the given file in 'simple' CSV format.
    Add newline at the end of each dictionary.
    Also, add given delimiter between key-value pairs in a dictionary.
    
    Note: The assumption is that each cell has no special characters to handle.
    '''
    f = open(output_file, 'w')
    
    f.write(str(header[0]))
    tail = header[1:]
    for h in tail:
        f.write(delim)
        f.write(str(h))
    f.write(os.linesep)
    
    for d in dicts:
        f.write( str(d[header[0]]) )
        for h in tail: 
            f.write(delim)
            f.write( str(d[h]) )
        f.write(os.linesep)
        
    f.close()

