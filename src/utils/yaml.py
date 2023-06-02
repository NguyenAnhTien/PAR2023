"""
@author: Tien Nguyen
@date  : 2023-04-26
"""
import yaml

def read_yaml(file_name):
    with open(file_name, 'r') as file:
        configs = yaml.safe_load(file)
    return configs
