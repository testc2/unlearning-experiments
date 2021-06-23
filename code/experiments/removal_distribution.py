from functools import partial
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import lr_optimize_sgd, lr_optimize_sgd_batch,predict,predict_log_proba,lr_grad
from methods.remove import remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from methods.scrub import scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch
from methods.common_utils import get_f1_score,get_roc_score
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from time import time
import torch.multiprocessing as mp
import numpy as np
from collections import OrderedDict
# from tqdm.contrib.telegram import tqdm
from tqdm import tqdm,trange
from experiments.removal_ratio import sample

header  = [
    "method","lam","l2_norm", # removal method and regularization
    "remove_ratio","sampler_seed","minibatch_size","remove_class","sample_prob","num_removes","minibatch_fraction","sampling_type", # removal details 
    "sgd_seed","optim","step_size","lr_schedule","training_batch_size","num_steps", # training details
    "training_time","removal_time", # timings
    "norm","grad_residual","model_diff", # paramteric metrics,
    "test_accuracy","test_f1_score","test_roc_score", # test metrics
    "remove_accuracy","remove_f1_score","remove_roc_score", # removed samples metrics
    "prime_accuracy","prime_f1_score","prime_roc_score", # remaining samples metrics
]

rows = OrderedDict({k:None for k in header})
def get_metrics(X_test,X_remove,X_prime,y_test,y_remove,y_prime,w_star,w,args):

    metrics ={}
    metrics["norm"] = float(w.norm())
    if not args.ovr:
        test_preds = predict(w,X_test)
        remove_preds = predict(w,X_remove)
        prime_preds = predict(w,X_prime)
        test_preds_log_proba = predict_log_proba(w,X_test)
        remove_preds_log_proba = predict_log_proba(w,X_remove)
        prime_preds_log_proba = predict_log_proba(w,X_prime)
        metrics["test_accuracy"] = accuracy_score(y_test,test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test,test_preds)
        metrics["test_roc_score"] = get_roc_score(y_test,test_preds_log_proba)
        
        metrics["remove_accuracy"] = accuracy_score(y_remove,remove_preds)
        metrics["remove_f1_score"] = get_f1_score(y_remove,remove_preds)
        metrics["remove_roc_score"] = get_roc_score(y_remove,remove_preds_log_proba)
        
        metrics["prime_accuracy"] = accuracy_score(y_prime,prime_preds)
        metrics["prime_f1_score"] = get_f1_score(y_prime,prime_preds)
        metrics["prime_roc_score"] = get_roc_score(y_prime,prime_preds_log_proba)

        metrics["grad_residual"] = float(lr_grad(w,X_prime,y_prime,args.lam).norm())
        metrics["model_diff"] = (w-w_star).norm(p=2).item()

    else:
        test_preds = predict_ovr(w,X_test)
        remove_preds = predict_ovr(w,X_remove)
        prime_preds = predict_ovr(w,X_prime)
        test_proba = predict_proba_ovr(w,X_test)
        remove_proba = predict_proba_ovr(w,X_remove)
        prime_proba = predict_proba_ovr(w,X_prime)
        y_test_ = y_test.argmax(1)
        y_remove_ = y_remove.argmax(1)
        y_prime_ = y_prime.argmax(1)
        metrics["test_accuracy"] = accuracy_score(y_test_,test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test_,test_preds,average="weighted")
        metrics["test_roc_score"] = get_roc_score(y_test_,test_proba,average="macro",multi_class="ovr")
        
        metrics["remove_accuracy"] = accuracy_score(y_remove_,remove_preds)
        metrics["remove_f1_score"] = get_f1_score(y_remove_,remove_preds,average="weighted")
        metrics["remove_roc_score"] = get_roc_score(y_remove_,remove_proba,average="macro",multi_class="ovr")
        
        metrics["prime_accuracy"] = accuracy_score(y_prime_,prime_preds)
        metrics["prime_f1_score"] = get_f1_score(y_prime_,prime_preds,average="weighted")
        metrics["prime_roc_score"] = get_roc_score(y_prime_,prime_proba,average="macro",multi_class="ovr")
    
        metrics["grad_residual"] = np.mean([float(lr_grad(w[:,k],X_prime,y_prime[:,k],args.lam).norm())for k in range(y_prime.size(1))])
        metrics["model_diff"] = (w-w_star).norm(p=2).item()
    return metrics


def dict_2_string(row,args,params,metrics,training_time):
    row.update(params)
    row["lam"] = args.lam
    row["optim"] = args.optim
    row["step_size"] = args.step_size
    row["num_steps"] = args.num_steps
    row["training_batch_size"] = args.batch_size
    row["lr_schedule"] = args.lr_schedule
    row["l2_norm"] = args.l2_norm
    row["training_time"]=training_time
    row.update(metrics)
    print_str = ",".join([str(x) for x in row.values()])
    print_str += "\n"
    return print_str

def train(data,args,sgd_seed=0):
    X_train = data["X_train"]
    y_train  = data["y_train"]
    param = dict(sgd_seed=sgd_seed)
    start = time()
    if args.ovr:
        w_orig = lr_ovr_optimize_sgd(X_train,y_train,param,args)
    else:
        w_orig = lr_optimize_sgd_batch(X_train,y_train,param,args)
    training_time = time()-start
    return w_orig,training_time

def retrain(params,args,data,w_orig):
    X_remove,y_remove,X_prime,y_prime = sample(
        data,
        params["remove_ratio"],
        params["remove_class"],
        params["sampler_seed"],
        params["sampling_type"]
    )
    start = time()
    if args.ovr:
        w_baseline = lr_ovr_optimize_sgd(X_prime,y_prime,params,args)
    else:
        w_baseline = lr_optimize_sgd_batch(X_prime,y_prime,params,args)
    retraining_time = time()-start
    params.update({
        "removal_time":retraining_time,
        "num_removes":X_remove.size(0),
    })
    metrics = get_metrics(
        data["X_test"],
        X_remove,
        X_prime,
        data["y_test"],
        y_remove,
        y_prime,
        w_orig,
        w_baseline,
        args
    )
    return metrics, params

def removal_step(params,args,data,w_orig):
    X_remove,y_remove,X_prime,y_prime = sample(
        data,
        params["remove_ratio"],
        params["remove_class"],
        params["sampler_seed"],
        params["sampling_type"]
    )
    num_removes = int(data["X_train"].size(0)*params["remove_ratio"])
    minibatch_size = num_removes//params["minibatch_fraction"]
    # Guo 
    if params["method"] == "Guo":
        start = time()
        if args.ovr:
            w_approx = remove_ovr_minibatch_pytorch(w_orig,data,minibatch_size,args,X_remove,X_prime,y_remove,y_prime)
        else:
            w_approx = remove_minibatch_pytorch(w_orig,data,minibatch_size,args,X_remove,X_prime,y_remove,y_prime)
        removal_time = time()-start
    elif params["method"] == "Golatkar":
        start = time()
        if args.ovr:
            w_approx,_ = scrub_ovr_minibatch_pytorch(w_orig,data,minibatch_size,args,noise=0,noise_seed=0,X_remove=X_remove,X_prime=X_prime,y_remove=y_remove,y_prime=y_prime)
        else:
            w_approx,_ = scrub_minibatch_pytorch(w_orig,data,minibatch_size,args,noise=0,noise_seed=0,X_remove=X_remove,X_prime=X_prime,y_remove=y_remove,y_prime=y_prime)
        removal_time = time()-start

    params.update({
        "removal_time":removal_time,
        "num_removes":X_remove.size(0),
        "minibatch_size":minibatch_size,
    })
    metrics = get_metrics(
        data["X_test"],
        X_remove,
        X_prime,
        data["y_test"],
        y_remove,
        y_prime,
        w_orig,
        w_approx,
        args
    )
    return metrics, params

def func(params,args,data,w_orig):
    if params["method"] == "baseline":
        return retrain(params,args,data,w_orig)
    else:
        return removal_step(params,args,data,w_orig)

def execute_distribution_exp(args,data):
    results_folder = args.results_dir/ args.dataset
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    
    if args.ovr:
        num_classes = data["y_train"].size(1)
        remove_class = 3 # fix removed class as 3
        file_name = f"Remove_Dist_multi{args.suffix}.csv"
    else:
        num_classes = len(data["y_train"].unique())
        remove_class = 0
        file_name = f"Remove_Dist_binary{args.suffix}.csv"
    
    results_file = results_folder / file_name
    num_processes = args.num_processes
    
    # remove_ratios = np.array([1]+list(range(5,55,5)))/100
    removal_step_grid ={
        "minibatch_fraction": args.minibatch_fractions,
        "method":["Guo","Golatkar"]
    }
    baseline_grid={"method":["baseline"]}
    

    base_grid_dict = {
        "uniform_random":
        {
            "remove_ratio":args.remove_ratios,
            "remove_class":[-1],# no removal class for uniform
            "sampling_type":["uniform_random"],
            "sampler_seed":range(args.num_sampler_seed),
            "sgd_seed":args.sgd_seed,

        },
        "targeted_random":
        {
            "remove_ratio":args.remove_ratios,
            "remove_class":[remove_class], # only chosen remove class
            "sampling_type":["targeted_random"],
            "sampler_seed":range(args.num_sampler_seed),
            "sgd_seed":args.sgd_seed,
        },
        "uniform_informed":
        {
            "remove_ratio":args.remove_ratios,
            "remove_class":[-1],
            "sampling_type":["uniform_informed"],
            "sampler_seed":[0],# no sampling seed for informed
            "sgd_seed":args.sgd_seed,
        },
        "targeted_informed":
        {
            "remove_ratio":args.remove_ratios,
            "remove_class":[remove_class], # only chosen remove class
            "sampling_type":["targeted_informed"],
            "sampler_seed":[0],
            "sgd_seed":args.sgd_seed,
        }
    }

    # Choose only the sampling distributions required
    base_grid = [base_grid_dict[sampling_type] for sampling_type in args.sampling_types]
    # Add minibatch fraction only for removal steps parameters and add baseline as is 
    param_grid_list = []
    for method_grid in [removal_step_grid,baseline_grid]:
        for grid in base_grid:
            d = grid.copy()
            d.update(method_grid)
            param_grid_list.append(d)
    
    param_grid = ParameterGrid(param_grid_list)

    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        torch.set_num_threads(1)
        with mp.Pool(num_processes) as pool:
            # train original model on whole training data
            w_orig,training_time = train(data,args)
            retrain_partial = partial(func,data=data,args=args,w_orig=w_orig)
            print("Starting Parallel")
            for metrics, params in tqdm(pool.imap_unordered(retrain_partial,param_grid),total=len(param_grid)):    
                fp.write(dict_2_string(rows.copy(),args,params,metrics,training_time))
                fp.flush()            
