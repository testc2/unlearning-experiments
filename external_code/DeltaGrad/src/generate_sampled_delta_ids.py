#%%
'''
Created on Jan 13, 2020

'''
import torch
import sys, os
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')

project_dir = Path(__file__).resolve().parent.parent.parent.parent
code_dir = project_dir/"code"
sys.path.append(str(code_dir.resolve()))
sys.path.append(os.path.abspath(__file__))

from utils import *
from experiments.removal_ratio import sample_uniform_random,sample_uniform_informed,sample_targeted_random,sample_targeted_informed
#%%

def sample_delta_ids(sampling_type,remove_ratio,dataset,targets,sampler_seed,remove_class):

    if len(targets.shape) >1 :
        y_train = targets.argmax(1)
    else:
        y_train = targets
    
    num_removes = int(dataset.size(0)*remove_ratio)
    if sampling_type == "uniform_random":
        remove_indices = sample_uniform_random(dataset,sampler_seed,num_removes)
    elif sampling_type == "uniform_informed":
        remove_indices = sample_uniform_informed(dataset,num_removes)
    elif sampling_type == "targeted_random":
        remove_indices = sample_targeted_random(y_train,remove_class,sampler_seed,num_removes)
    elif sampling_type == "targeted_informed":
        remove_indices = sample_targeted_informed(dataset,y_train,remove_class,num_removes)
    else:
        raise ValueError("Wrong sampling type. Please Check Arguments")

    return remove_indices

if __name__ == '__main__':
    
#     sys_args = sys.argv

    parser = argparse.ArgumentParser('generate_dist_ids')

    parser.add_argument("sampling_type", type=str, help= "Type of sampling to perform", choices=["uniform_random","uniform_informed","targeted_random","targeted_informed"])
    
    parser.add_argument('--dataset',  help="dataset to be used")
    
    parser.add_argument('--ratio',  type=float, help="delete rate or add rate")
    
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
    
    delta_data_ids = sample_delta_ids(
        sampling_type=args.sampling_type,
        remove_ratio=args.ratio,
        dataset=dataset_train.data,
        targets=dataset_train.labels,
        sampler_seed=args.sampler_seed,
        remove_class=args.class_id
    )
        
    torch.save(train_data_len, git_ignore_folder + 'train_data_len')
        
    print(delta_data_ids)
    
    torch.save(delta_data_ids, git_ignore_folder + "delta_data_ids")
    
# %%
