#%%
'''
Created on Jan 13, 2020

'''
import torch

import sys, os

import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')


sys.path.append(os.path.abspath(__file__))

from utils import *
#%%
# try:
# # from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
#     from utils import *
# #     from sensitivity_analysis.linear_regression.evaluating_test_samples import *
# #     from Models.Data_preparer import *
#     
#     from generate_noise import *
# 
# except ImportError:
#     from Load_data import *
# # from sensitivity_analysis.logistic_regression.Logistic_regression import test_X
#     from utils import *
# #     from Models.Data_preparer import *
# #     from evaluating_test_samples import *
#     from generate_noise import *

def get_class_delta_ids(ratio,targets,class_id=3):
    all_ids = torch.arange(0,targets.shape[0])
    y_class = all_ids[targets==class_id]
    num_removes = int(len(y_class)*ratio)
    remove_indices = y_class[:num_removes]

    return remove_indices
if __name__ == '__main__':
    
#     sys_args = sys.argv

    parser = argparse.ArgumentParser('generate_rand_ids')

    
    parser.add_argument('--dataset',  help="dataset to be used")
    
    parser.add_argument('--ratio',  type=float, help="delete rate or add rate")
    
    parser.add_argument('--class-id', type=int, default=3, help="Id of class to remove samples from")

    parser.add_argument('--repo', default = gitignore_repo, help = 'repository to store the data and the intermediate results')

    args = parser.parse_args()

    git_ignore_folder = args.repo
    
    noise_rate = args.ratio
    
    
    dataset_name = args.dataset
    
    data_preparer = Data_preparer()
    
    
    function=getattr(Data_preparer, "prepare_" + dataset_name)
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    print(dataset_train.data.shape)
    
    train_data_len = len(dataset_train.data)
    
    delta_data_ids = get_class_delta_ids(args.ratio,dataset_train.labels,args.class_id)
        
    
    torch.save(train_data_len, git_ignore_folder + 'train_data_len')
        
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
    
    
    
    
    
    
# %%
