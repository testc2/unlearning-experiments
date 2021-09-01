from functools import partial
from os import error, stat
from typing import Callable
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import (
    lr_optimize_sgd_batch,
    predict,
    predict_log_proba,
    lr_grad,
)
from methods.remove import remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from methods.scrub import scrub
from methods.common_utils import get_f1_score, get_roc_score
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from time import time
import torch.multiprocessing as mp
import numpy as np
from collections import OrderedDict
from experiments.removal_ratio import sample

# from tqdm.contrib.telegram import tqdm
from tqdm import tqdm, trange
header = [
    # removal method and regularization
    "strategy","lam","l2_norm",
    # deletion details
    "remove_ratio","deletion_batch_size","sampler_seed","remove_class","sampling_type",  
    "sgd_seed","optim","step_size","lr_schedule","batch_size","num_steps",  # training details
    "running_time","unlearning_time","retraining_time","other_time",  # running time details
    "test_accuracy","cum_remove_accuracy","batch_remove_accuracy","pipeline_acc_err",  #  metrics
    "num_deletions","retrained","batch_deleted_class_balance","cum_deleted_class_balance", # additional metrics
    "threshold"
]

rows = OrderedDict({k: None for k in header})

def dict_2_string(row,args,params,metrics_list):
    strings = []
    method_row = row.copy()
    args_dict = vars(args)
    for metrics in metrics_list:
        method_row.update((k,params[k]) for k in params.keys() & method_row.keys())
        method_row.update((k,args_dict[k]) for k in args_dict.keys() & method_row.keys())
        method_row.update((k,metrics[k]) for k in metrics.keys() & method_row.keys())
        
        print_str = ",".join([str(x) for x in method_row.values()])
        print_str += "\n"
        strings.append(print_str)
    return strings

def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = np.array((numerator==0)&(denominator==0),ndmin=1)
    sae = np.array(numerator/denominator,ndmin=1)
    sae[both_zero] = 1 
    return sae*100


def pipeline(w:torch.Tensor,strategy:Callable,args:dict,params:dict,data:dict):
    # clone the trained weights to ensure it is not changed 
    w_temp = w.clone()
    X_remove = data["X_remove"]
    X_prime = data["X_prime"]
    X_test = data["X_test"]
    y_remove = data["y_remove"]
    y_prime = data["y_prime"]
    y_test = data["y_test"]
    batch_size = args.deletion_batch_size
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    num_removes = X_remove.shape[0]
    metrics = []
    state_dict ={
        "test_acc_init":accuracy_score(y_test,predict(w,X_test)),
        "retrained":False,
        "unlearning_time":0,
        "retraining_time":0,
        "other_time":0
    }

    for batch in trange(0,num_removes,batch_size):
        _metrics = {}
        X_batch_remove = data["X_batch_remove"] = X_remove[batch:batch+batch_size]
        y_batch_remove = data["y_batch_remove"] = y_remove[batch:batch+batch_size]
        X_batch_prime = data["X_batch_prime"] = X_train_temp[batch+batch_size:]
        y_batch_prime = data["y_batch_prime"] = y_train_temp[batch+batch_size:]
        
        start = time()
        w_temp = strategy(w_temp,args,params,data,state_dict)
        running_time = time() - start 

        # compute time metrics 
        state_dict["other_time"] = running_time - (state_dict["retraining_time"]+state_dict["unlearning_time"])
        # update current metrics 
        _metrics.update(state_dict)

        # compute performance metrics
        test_accuracy = accuracy_score(y_test,predict(w_temp,X_test))
        acc_err_init = SAPE(test_accuracy,state_dict["test_acc_init"])[0]
        cum_remove_accuracy = accuracy_score(y_remove[:batch+batch_size],predict(w_temp,X_remove[:batch+batch_size]))
        batch_remove_accuracy = accuracy_score(y_batch_remove,predict(w_temp,X_batch_remove))
        num_deletions = min(batch+batch_size,num_removes)

        if args.ovr:
            #TODO implement logic for multi-class class balance 
            pass
        else:
            batch_class_balance = ((y_batch_remove == 0).sum()/y_batch_remove.shape[0]).item()
            cum_class_balance = ((y_remove[:batch+batch_size]==0).sum()/(num_deletions)).item()
        
        # add performance and other metrics
        _metrics.update({
            "running_time":running_time,
            "test_accuracy":test_accuracy,
            "batch_remove_accuracy":batch_remove_accuracy,
            "cum_remove_accuracy":cum_remove_accuracy,
            "pipeline_acc_err":acc_err_init,
            "num_deletions":num_deletions,
            "batch_deleted_class_balance":batch_class_balance,
            "cum_deleted_class_balance":cum_class_balance,
        })

        metrics.append(_metrics)
    return metrics


def do_nothing(w,args,params,data,state_dict):
    return w

def always_retrain(w,args,params,data,state_dict):
    state_dict["retrained"]=True
    start = time()
    w_prime =  lr_optimize_sgd_batch(
        data["X_batch_prime"],
        data["y_batch_prime"],
        params,
        args
    )
    state_dict["retraining_time"] = time() - start
    return w_prime

def always_unlearn_gol(w,args,params,data,state_dict):
    start = time()
    w_approx,_ = scrub(
        w,
        data["X_batch_prime"],
        data["y_batch_prime"],
        args.lam,
        noise=0,
        noise_seed=0
    )
    state_dict["unlearning_time"]= time() - start
    return w_approx

def gol_test_acc_thresh(w,args,params,data,state_dict):
    # always unlearn 
    start = time()
    w_approx,added_noise = scrub(
        w,
        data["X_batch_prime"],
        data["y_batch_prime"],
        args.lam,
        noise=0,
        noise_seed=0
    )
    state_dict["unlearning_time"] = time() - start
    
    # compute test accuracy
    test_accuracy = accuracy_score(data["y_test"],predict(w_approx,data["X_test"]))
    # find the SAPE wrt to test accuracy of last checkpoint
    acc_err_init = SAPE(test_accuracy,state_dict["test_acc_init"])[0]
    # if the unlearned model exceeds acc_test threshold then retrain 
    if acc_err_init > params["threshold"]:
        state_dict["retrained"]=True
        
        start = time()
        w_approx =  lr_optimize_sgd_batch(data["X_batch_prime"],data["y_batch_prime"],params,args)
        state_dict["retraining_time"] = time() - start

        # update checkpoint test accuracy
        state_dict["test_acc_init"] = accuracy_score(data["y_test"],predict(w_approx,data["X_test"]))
    return w_approx

def run_pipeline(w,args,params,data):
    # begin pipeline
    method = args.strategy
    if method == "nothing":
        strat_fn = do_nothing
    elif method == "retrain":
        strat_fn = always_retrain
    elif method == "golatkar":
        strat_fn = always_unlearn_gol
    elif method == "golatkar_test_thresh":
        strat_fn = gol_test_acc_thresh
    else:
        raise ValueError(f"Strategy {method} is not supported")
    
    metrics = pipeline(w,strat_fn,args,params,data)

    return dict_2_string(rows,args,params,metrics)




def execute_when_to_retrain(args,data):
    results_folder = args.results_dir/ args.dataset/ "when_to_retrain"
    
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    
    if args.ovr:
        num_classes = data["y_train"].size(1)
        remove_class = 3 # fix removed class as 3
        file_name = f"{args.strategy}_multi{args.suffix}.csv"
    else:
        num_classes = len(data["y_train"].unique())
        remove_class = 0
        file_name = f"{args.strategy}_binary{args.suffix}.csv"
    
    results_file = results_folder / file_name
    num_processes = args.num_processes
    torch.set_num_threads(num_processes)

    X_remove,y_remove,X_prime,y_prime = sample(
        data,
        args.remove_ratio,
        remove_class,
        sampler_seed=args.sampler_seed,
        sampling_type=args.sampling_type
    )

    data["X_remove"] =X_remove
    data["X_prime"] =X_prime
    data["y_remove"] =y_remove
    data["y_prime"] =y_prime

    param_grid_dict = {
            "remove_class":[remove_class],
            "sgd_seed":[args.sgd_seed]
    }

    if args.strategy == "golatkar_test_thresh":
        param_grid_dict.update({
            "threshold":args.thresholds,
        })
    
    param_grid = ParameterGrid(param_grid_dict)


    # train model with no noise
    if args.ovr: 
        pass
    else:
        w = lr_optimize_sgd_batch(data["X_train"],data["y_train"],{"sgd_seed":args.sgd_seed},args)

    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        for params in tqdm(param_grid,total=len(param_grid)):    
            strings = run_pipeline(w,args,params,data)
            [fp.write(s) for s in strings]
            fp.flush()