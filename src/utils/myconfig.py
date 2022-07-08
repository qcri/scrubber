'''
Created on Jul 26, 2015

@author: zhj
'''

def read_config(config_file, delim='='):
    '''
    Args:
        config_file (string): file name or path of the configuration file
        delim (char): the delimiter character. Default value is '='.
    
    Returns:
        params (dict): a dictionary storing configuration pairs
    
    Read configurations from a simple file.
    
    Each line (except comment line and empty line) contains a key-value pair, separated by the given delimiter.
    Comment line starts with '#', and empty line must not have whitespace or tabular key.
    
    Note: leading and tailing whitespaces in keys and values are trimmed.
    '''
    params = {}
    
    with open(config_file) as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if len(line) == 0: # empty line
                continue
            elif line[0] == '#': # comment line
                continue
            else:
                index = line.find(delim)
                if index == -1:
                    raise Exception("Delimiter character was not found in line " + str(i))
                key = line[:index].strip() # key
                value = line[index+1:].strip() #value
                params[key] = value 
    
    return params