'''
Created on Nov 4, 2016

@author: hzhang0418
'''

import datetime

class MyLogger:
    
    def __init__(self, output_file, is_append=True):
        self.output_file = output_file
        self.is_append = is_append
        
    def open(self):
        if self.is_append:
            self.f = open(self.output_file, 'a')
        else:
            self.f = open(self.output_file, 'w')
        
    def close(self):
        self.f.close()
        
    def flush(self):
        self.f.flush()
        
    def log_string(self, string):
        self.f.write(string)
        self.f.write('\n')
        
    def log_tuple(self, t, delim='\t'):
        self.f.write(str(t[0]))
        for v in t[1:]: 
            self.f.write(delim)
            self.f.write(str(v))
        self.f.write('\n')
        
    def log_datetime(self):
        self.f.write(datetime.datetime.now().strftime("%y %m %d %H %M %S"))
        self.f.write('\n')
        