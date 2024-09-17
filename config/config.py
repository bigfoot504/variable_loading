import yaml
from os import path
import numpy as np

here = path.abspath(path.dirname(__file__))
path2Yaml = './config.yaml'
pathYaml = path.join(here, path2Yaml)

def config_reader_yaml():
    with open(pathYaml, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Convert block volume distribution across weeks into a probability distribution
    for time_tier, vol_distr in config_dict['Volume_Distributions'].items():
        config_dict['Volume_Distributions'][time_tier] = np.array(vol_distr) / sum(vol_distr)

    # Convert weekly volume distribution across days into a probability distribution
    for lift, lift_dict in config_dict['Lifts'].items():
        weekly_distribution = config_dict['Lifts'][lift]['Weekly Distribution']
        config_dict['Lifts'][lift]['Weekly Distribution'] = np.array(weekly_distribution) / sum(weekly_distribution)
    
    return config_dict