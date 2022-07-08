'''
Created on Dec 17, 2017

@author: haojun
'''

import io
import csv

def read_csv_as_list(input_file):
    rows = []
    with io.open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', dialect=csv.excel_tab)
        rows = [ row for row in reader]
    return rows

def read_csv_as_dict(input_file, key_attr):
    rows = {}
    with io.open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', dialect=csv.excel_tab)
        for row in reader:
            rows[ row[key_attr] ] = row  
    return rows

def write_list_to_csv(rows_as_list, fieldnames, output_file):
    with open(output_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames, extrasaction='ignore', delimiter=',')
        writer.writeheader()
        writer.writerows(rows_as_list)

def write_dict_to_csv(rows_as_dict, fieldnames, output_file):
    tmp = list(rows_as_dict.values())
    write_list_to_csv(tmp, fieldnames, output_file)