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

def sample_delta_ids(ratio,targets,sample_prob,sampler_seed,class_id=3):

    if len(targets.shape) >1 :
        y_train = targets.argmax(1)
    else:
        y_train = targets
    
    num_classes = len(torch.unique(y_train))
    weights = torch.ones(y_train.size(0))
    remove_class_weight = sample_prob
    for k in range(num_classes):
        if k == class_id:
            weights[y_train==k]=remove_class_weight
        else:
            weights[y_train==k]=(1-remove_class_weight)/(num_classes-1)
    torch.manual_seed(sampler_seed)
    num_removes = int(len(y_train)*ratio)
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights,replacement=False,num_samples=num_removes)
    remove_indices = torch.tensor(list(sampler))

    return remove_indices

def sample_cost_delta_ids(ratio,data,targets,sample_prob,sampler_seed,class_id=3):

    if len(targets.shape) >1 :
        y_train = targets.argmax(1)
    else:
        y_train = targets
    
    num_removes = int(len(y_train)*ratio)
    weights = torch.zeros(y_train.size(0))
    # sort in descending order of L2 norm    
    norms_cost_indices = data.norm(dim=1).argsort(descending=True)

    # sort labels by the norm cost
    y_train_sorted = y_train.index_select(dim=0,index=norms_cost_indices)
    class_mask = y_train_sorted==class_id
    # sorted costs of remove class
    class_cost = norms_cost_indices[class_mask]
    # the top costs
    top_class_cost_ids = class_cost[:num_removes]
    # the other costs 
    bottom_class_cost_ids = class_cost[num_removes:]
    
    # all other classes are 0 weight
    weights[y_train!=class_id]=0
    # sample top costs with sample probability
    weights[top_class_cost_ids] = sample_prob
    # other samples of same class have inverse probability
    weights[bottom_class_cost_ids] = 1-sample_prob
    torch.manual_seed(sampler_seed)
    num_removes = int(len(y_train)*ratio)
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights,replacement=False,num_samples=num_removes)
    remove_indices = torch.tensor(list(sampler))

    return remove_indices

if __name__ == '__main__':
    
#     sys_args = sys.argv

    parser = argparse.ArgumentParser('generate_dist_ids')

    parser.add_argument('removal_type', type=str, default="targeted", choices=["targeted","informed"], help="Method to sample removed points")
    
    parser.add_argument('--dataset',  help="dataset to be used")
    
    parser.add_argument('--ratio',  type=float, help="delete rate or add rate")

    parser.add_argument("--sample-prob", type=float, help= "sampling probability for chosen class")
    
    parser.add_argument("--sampler-seed", type=int, help= "Seed for the sampler")
        
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
    
    if args.removal_type == "targeted":
        delta_data_ids = sample_delta_ids(args.ratio,dataset_train.labels,args.sample_prob,args.sampler_seed,args.class_id)
    elif args.removal_type == "informed":
        delta_data_ids = sample_cost_delta_ids(args.ratio,dataset_train.data,dataset_train.labels,args.sample_prob,args.sampler_seed,args.class_id)
    else:
        raise ValueError("Check removal type argument")
        
    torch.save(train_data_len, git_ignore_folder + 'train_data_len')
        
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
# %%
