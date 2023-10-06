import yaml
from os import path
import numpy as np

here = path.abspath(path.dirname(__file__))
path2Yaml = './config.yaml'
pathYaml = path.join(here, path2Yaml)

def config_reader_yaml():
    with open(pathYaml, 'r') as file:
        config_dict = yaml.safe_load(file)

    for k,v in config_dict['Volume_Distributions'].items():
        config_dict['Volume_Distributions'][k] = np.array(v) / sum(v)
    
    return config_dict