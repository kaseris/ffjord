import numpy as np
import os
import sys
import pickle
import torch.nn as nn

sys.path.append(os.getcwd())
from datasets.human36m.euler import read_all_data, define_actions

if __name__ == '__main__':
    actions = define_actions("all")

    prefix_len = 50
    pred_len = 25
    data_dir = './Human3.6M'
    omit_one_hot = True

    print('Start Reading data')

    train_set, test_set, \
    data_mean, data_std, \
    dim_to_ignore, dim_to_use = read_all_data(  actions, 
                                                prefix_len, 
                                                pred_len, 
                                                data_dir, 
                                                not omit_one_hot)

    data = {}
    data['train'] = train_set
    data['test'] = test_set
    data['mean'] = data_mean
    data['std'] = data_std
    data['dim_to_ignore'] = dim_to_ignore
    data['dim_to_use'] = dim_to_use

    pickle.dump(data, open('./Human3.6M/h36m_euler.pkl', 'wb'))

    print("Finish to save processed human 3.6m")