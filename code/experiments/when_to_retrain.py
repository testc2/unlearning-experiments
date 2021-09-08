from functools import partial
from os import error, stat
from typing import Callable, Optional
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import (
    lr_optimize_sgd_batch,
    predict,
    predict_log_proba,
    lr_grad,
)
from methods.remove import remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from methods.scrub import scrub, compute_noise, scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch
from methods.common_utils import SAPE, get_f1_score, get_roc_score
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
    "noise","noise_seed",  # privacy noise details
    "running_time","unlearning_time","retraining_time","other_time",  # running time details
    "test_accuracy","cum_remove_accuracy","batch_remove_accuracy","pipeline_acc_err","pipeline_acc_dis_est","pipeline_abs_err",  #  metrics
    "num_deletions","retrained","batch_deleted_class_balance","cum_deleted_class_balance", # additional metrics
    "threshold","prop_const"
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


def pipeline(w:torch.Tensor,strategy:Callable,args:dict,params:dict,data:dict,pre_computed_data:dict={}):
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
    }
    state_dict.update(pre_computed_data)
    for batch in trange(0,num_removes,batch_size):
        _metrics = {}
        # reset metrics 
        state_dict.update({
            "retrained":False,
            "unlearning_time":0,
            "retraining_time":0,
            "other_time":0
        })
        X_batch_remove = data["X_batch_remove"] = X_remove[batch:batch+batch_size]
        y_batch_remove = data["y_batch_remove"] = y_remove[batch:batch+batch_size]
        X_batch_prime = data["X_batch_prime"] = X_train_temp[batch+batch_size:]
        y_batch_prime = data["y_batch_prime"] = y_train_temp[batch+batch_size:]
        X_remove_cum = data["X_remove_cum"] = X_remove[:batch+batch_size]
        y_remove_cum = data["y_remove_cum"] = y_remove[:batch+batch_size]
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
        abs_err_init = np.abs(test_accuracy-state_dict["test_acc_init"])
        cum_remove_accuracy = accuracy_score(y_remove_cum,predict(w_temp,X_remove_cum))
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
            "pipeline_abs_err":abs_err_init,
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
    # compute noise to be added 
    noise_scrub = compute_noise(
        w_prime,
        data["X_batch_prime"],
        data["y_batch_prime"],
        args.lam,
        params["noise"],
        params["noise_seed"]
    )
    w_prime += noise_scrub
    state_dict["retraining_time"] = time() - start
    return w_prime

def always_unlearn_gol(w,args,params,data,state_dict):
    start = time()
    w_approx,_ = scrub(
        w,
        data["X_batch_prime"],
        data["y_batch_prime"],
        args.lam,
        noise=params["noise"],
        noise_seed=params["noise_seed"]
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
        noise=params["noise"],
        noise_seed=params["noise_seed"]
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
        # compute noise to be added
        noise_scrub = compute_noise(
            w_approx,
            data["X_batch_prime"],
            data["y_batch_prime"],
            args.lam,
            params["noise"],
            params["noise_seed"]
        )
        w_approx += noise_scrub
        state_dict["retraining_time"] = time() - start

        # update checkpoint test accuracy
        state_dict["test_acc_init"] = accuracy_score(data["y_test"],predict(w_approx,data["X_test"]))
    return w_approx

def gol_disparity_thresh(w,args,params,data,state_dict,v2=False):
    # always unlearn 
    start = time()
    w_approx,added_noise = scrub(
        w,
        data["X_batch_prime"],
        data["y_batch_prime"],
        args.lam,
        noise=params["noise"],
        noise_seed=params["noise_seed"]
    )
    state_dict["unlearning_time"] = time() - start
    
    # compute test accuracy
    test_accuracy = accuracy_score(data["y_test"],predict(w_approx,data["X_test"]))
    if v2:
        # find the absolute error wrt to test accuracy of last checkpoint
        abs_err_init = np.abs(test_accuracy-state_dict["test_acc_init"])
        # find accuracy on deleted sample (cumulative) of the unlearned model
        acc_del_u = accuracy_score(data["y_remove_cum"],predict(w_approx,data["X_remove_cum"]))


        # estimate the deleted sample accuracy of retrained model using the proportionality const
        term = state_dict["prop_const"]*abs_err_init
        acc_del_retrained = [acc_del_u+term,acc_del_u-term]
        # compute the disparity as SAPE of both cases 
        sape_del = SAPE(acc_del_u,acc_del_retrained)
        # select the largest SAPE as the disparity  
        acc_dis_est = max(sape_del)
    
    else:
        # find the SAPE wrt to test accuracy of last checkpoint
        acc_err_init = SAPE(test_accuracy,state_dict["test_acc_init"])[0]
        # estimate the AccDis using the proportionality const
        acc_dis_est = acc_err_init * state_dict["prop_const"]

    # if the estimated disparity of the unlearned model exceeds the threshold then retrain 
    if acc_dis_est > params["threshold"]:
        state_dict["retrained"]=True
        
        start = time()
        w_approx =  lr_optimize_sgd_batch(data["X_batch_prime"],data["y_batch_prime"],params,args)
        # compute noise to be added
        noise_scrub = compute_noise(
            w_approx,
            data["X_batch_prime"],
            data["y_batch_prime"],
            args.lam,
            params["noise"],
            params["noise_seed"]
        )
        w_approx += noise_scrub
        state_dict["retraining_time"] = time() - start

        # update checkpoint test accuracy
        state_dict["test_acc_init"] = accuracy_score(data["y_test"],predict(w_approx,data["X_test"]))
        acc_dis_est = 0
    
    state_dict["pipeline_acc_dis_est"] = acc_dis_est
    return w_approx


def obtain_proportionality_const(w,args,params,data,v2=False):
    """Function that computes the proportionality constant for the estimation of accuracy disparity

    Args:
        w (torch.tensor): The initial trained model
        args (argparse): the arguments for the method
        params (dict): the paramters for the current run of the experiment
        data (dict): A collection of the test, train and other data
        v2 (bool, optional): To choose the absolute difference based estimation or not. Defaults to False.

    Returns:
        [type]: [description]
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    # For AccDis Strategy retrain at large deletion ratio and compute proportionality
    if args.ovr:
        deletion_ratio = 0.09
    else :
        deletion_ratio = 0.45
    
    # generate most adversarial examples 
    X_remove,y_remove,X_prime,y_prime = sample(
        data,
        deletion_ratio,
        params["remove_class"],
        sampler_seed=0,
        sampling_type="targeted_random"
    )

    # obtain unlearned model 
    if args.ovr:
        w_approx, _ = scrub_ovr_minibatch_pytorch(
            w,
            data,
            minibatch_size=X_remove.shape[0],
            args=args,
            noise=params["noise"],
            noise_seed=params["noise_seed"],
            X_remove=X_remove,
            X_prime=X_prime,
            y_remove=y_remove,
            y_prime=y_prime 
        )
    else:
        w_approx,_ = scrub_minibatch_pytorch(
            w,
            data,
            minibatch_size=X_remove.shape[0],
            args=args,
            noise=params["noise"],
            noise_seed=params["noise_seed"],
            X_remove=X_remove,
            X_prime=X_prime,
            y_remove=y_remove,
            y_prime=y_prime
            
        )

    # obtain retrained model 
    if args.ovr:
        pass
    else:
        w_prime = lr_optimize_sgd_batch(X_prime,y_prime,params,args)
        # compute noise to be added
        if params["noise"]>0:
            _,noise_scrub = scrub_minibatch_pytorch(
                w_prime,
                data,
                X_remove.shape[0],
                args,
                params["noise"],
                params["noise_seed"],
                X_remove=X_remove,
                X_prime=X_prime,
                y_remove=y_remove,
                y_prime=y_prime
            )
        else:
            noise_scrub = torch.zeros_like(w_prime)
        w_prime += noise_scrub
    
    # compute metrics
    if args.ovr:
        pass
    else:
        # get test accuracies
        retrain_test_acc = accuracy_score(y_test,predict(w_prime,X_test))
        gol_test_acc = accuracy_score(y_test,predict(w_approx,X_test))
        init_test_acc = accuracy_score(y_test,predict(w,X_test))
        # get deleted samples accuracy 
        retrain_del_acc = accuracy_score(y_remove,predict(w_prime,X_remove))
        gol_del_acc = accuracy_score(y_remove,predict(w_approx,X_remove))
    
    if v2:
        # compute AbsDis and AbsErr_init
        abs_dis = np.abs(retrain_del_acc-gol_del_acc)
        abs_err_init = np.abs(init_test_acc-gol_test_acc)
        c = abs_dis/abs_err_init
    else:
        # compute AccDis and AccErr_init
        acc_dis = SAPE(retrain_del_acc,gol_del_acc)[0]
        acc_err_init = SAPE(init_test_acc,gol_test_acc)[0]

        c = acc_dis/acc_err_init
    return c


def run_pipeline(args,params,data):
    # train model with no noise
    if args.ovr: 
        pass
    else:
        w = lr_optimize_sgd_batch(data["X_train"],data["y_train"],params,args)
        # compute noise to be added
        noise_scrub = compute_noise(
            w,
            data["X_train"],
            data["y_train"],
            args.lam,
            params["noise"],
            params["noise_seed"]
        )
        w += noise_scrub
    
    pre_computed_data={}
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
    elif method == "golatkar_disparity_thresh_v1":
        strat_fn = gol_disparity_thresh
        c = obtain_proportionality_const(w,args,params,data,v2=False)
        pre_computed_data["prop_const"]=c
    elif method == "golatkar_disparity_thresh_v2":
        strat_fn = gol_disparity_thresh
        c = obtain_proportionality_const(w,args,params,data,v2=True)
        pre_computed_data["prop_const"]=c
    else:
        raise ValueError(f"Strategy {method} is not supported")
    
    metrics = pipeline(w,strat_fn,args,params,data,pre_computed_data)

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
            "sgd_seed":[args.sgd_seed],
            "noise_seed":args.noise_seeds,
            "noise":args.noise_levels
    }

    if args.strategy in ["golatkar_test_thresh","golatkar_disparity_thresh_v1","golatkar_disparity_thresh_v2"]:
        param_grid_dict.update({
            "threshold":args.thresholds,
        })
    
    param_grid = ParameterGrid(param_grid_dict)

    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        for params in tqdm(param_grid,total=len(param_grid)):    
            strings = run_pipeline(args,params,data)
            [fp.write(s) for s in strings]
            fp.flush()