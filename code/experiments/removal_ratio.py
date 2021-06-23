from functools import partial
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import lr_optimize_sgd, lr_optimize_sgd_batch,predict,predict_log_proba,lr_grad
from methods.remove import remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from methods.scrub import scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch
from methods.common_utils import get_f1_score,get_roc_score
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import ParameterGrid
from time import time
import torch.multiprocessing as mp
import numpy as np
from collections import OrderedDict
# from tqdm.contrib.telegram import tqdm
from tqdm import tqdm,trange


header  = [
    "lam","l2_norm", # removal method and regularization
    "remove_ratio","sampler_seed","minibatch_size","remove_class","sample_prob","num_removes","minibatch_fraction","sampling_type", # removal details 
    "sgd_seed","optim","step_size","lr_schedule","training_batch_size","num_steps", # training details
    "training_time","removal_time", # timings
    "norm","grad_residual","model_diff_orig", # paramteric metrics,
    "test_accuracy","test_f1_score","test_roc_score", # test metrics
    "remove_accuracy","remove_f1_score","remove_roc_score", # removed samples metrics
    "prime_accuracy","prime_f1_score","prime_roc_score", # remaining samples metrics
]

rows = OrderedDict({k:None for k in header})

def get_metrics(X_test,X_remove,X_prime,y_test,y_remove,y_prime,w_orig,w,args):

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
        metrics["model_diff_orig"] = (w-w_orig).norm(p=2).item()

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
        metrics["model_diff_orig"] = (w-w_orig).norm(p=2).item()
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

def sample_uniform_random(X_train,sampler_seed,num_removes):
    """Samples removed points uniformly from all classes and at random

    Args:
        data (dict): training data dictionary
        sampler_seed (int): seed for randomness
        num_removes (int): number of points to remove
    """
    torch.manual_seed(sampler_seed)
    perm = torch.randperm(X_train.size(0)) 
    remove_indices = perm[:num_removes]
    return remove_indices

def sample_targeted_random(y_train,remove_class,sampler_seed,num_removes):
    """Samples removed points from a particular class at random

    Args:
        data (dict): training data dictionary
        remove_class (int): The class_id to be targeted 
        sampler_seed (int): seed for randomness
        num_removes (int): number of points to be removed
    """
    # indices of class to be removed
    remove_class_indices = (y_train==remove_class).nonzero().flatten()
    n_remove_class = remove_class_indices.size(0)
    torch.manual_seed(sampler_seed)
    class_perm = torch.randperm(n_remove_class)
    # indices of elements to be removed in class
    remove_indices_class = class_perm[:num_removes]
    remove_indices = remove_class_indices[remove_indices_class]
    return remove_indices

def sample_uniform_informed(X_train,num_removes):
    """Samples removed points uniformly from all classes and based on norm

    Args:
        remove_class (int): The class_id to be targeted 
        sampler_seed (int): seed for randomness
        num_removes (int): number of points to be removed
    """    
    # find the L2 norms of all the samples in descending order 
    norms_cost_indices = X_train.norm(dim=1).argsort(descending=True)
    # remove the top indoces irrespective of class
    remove_indices = norms_cost_indices[:num_removes]
    return remove_indices

def sample_targeted_informed(X_train,y_train,remove_class,num_removes):
    """Samples removed points from a particular class based on L2 norm

    Args:
        data (dict): training data dictionary
        remove_class (int): The class_id to be targeted 
        sampler_seed (int): seed for randomness
        num_removes (int): number of points to be removed
    """
    # find the L2 norms of all the samples in descending order 
    norms_cost_indices = X_train.norm(dim=1).argsort(descending=True)
    # find the class of in the order of decreasing norm
    y_train_sorted = y_train.index_select(dim=0,index=norms_cost_indices)
    class_mask = (y_train_sorted==remove_class)
    # filter norm costs based on class targeted
    class_norm_indices =  norms_cost_indices[class_mask]
    # choose top indices to be removed
    remove_indices = class_norm_indices[:num_removes]
    return remove_indices

def sample(data,remove_ratio,remove_class,sampler_seed,sampling_type):
    X_train = data["X_train"]
    y_train_orig = data["y_train"]
    if len(y_train_orig.shape) >1 :
        y_train = y_train_orig.argmax(1)
    else:
        y_train = y_train_orig
    
    num_removes = int(X_train.size(0)*remove_ratio)
    if sampling_type == "uniform_random":
        remove_indices = sample_uniform_random(X_train,sampler_seed,num_removes)
    elif sampling_type == "uniform_informed":
        remove_indices = sample_uniform_informed(X_train,num_removes)
    elif sampling_type == "targeted_random":
        remove_indices = sample_targeted_random(y_train,remove_class,sampler_seed,num_removes)
    elif sampling_type == "targeted_informed":
        remove_indices = sample_targeted_informed(X_train,y_train,remove_class,num_removes)
    else:
        raise ValueError("Wrong sampling type. Please Check Arguments")
    # find indices to keep
    keep_indices = torch.LongTensor(list(set(range(X_train.size(0)))-set(remove_indices.numpy())))
    X_remove = X_train.index_select(0,remove_indices)
    X_prime = X_train.index_select(0,keep_indices)
    y_remove = y_train_orig.index_select(0,remove_indices)
    y_prime = y_train_orig.index_select(0,keep_indices)

    return X_remove,y_remove,X_prime,y_prime

def execute_ratio_exp(args,data):
    results_folder = args.results_dir/ args.dataset
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    
    if args.ovr:
        num_classes = data["y_train"].size(1)
        file_name = f"Ratio_multi{args.suffix}.csv"
    else:
        num_classes = 2
        file_name = f"Ratio_binary{args.suffix}.csv"

    results_file = results_folder / file_name
    num_processes = args.num_processes

    # train original model on whole training data
    # remove_ratios = np.array([1]+list(range(5,55,5)))/100
    remove_ratios = np.r_[(np.array(range(50,90,5))/100),np.linspace(0.9,1,20,endpoint=False)]
    grid = ParameterGrid([
        {
            "remove_ratio":remove_ratios,
            "remove_class":[-1],# no removal class for uniform
            "sampling_type":["uniform_random"],
            "sampler_seed":range(args.num_sampler_seed),
            "sgd_seed":args.sgd_seed,
        },
        # {
        #     "remove_ratio":remove_ratios,
        #     "remove_class":range(num_classes),
        #     "sampling_type":["targeted_random"],
        #     "sampler_seed":range(args.num_sampler_seed),
        #     "sgd_seed":args.sgd_seed,
        # },
        {
            "remove_ratio":remove_ratios,
            "remove_class":[-1],
            "sampling_type":["uniform_informed"],
            "sampler_seed":[0],# no sampling seed for informed
            "sgd_seed":args.sgd_seed,
        },
        # {
        #     "remove_ratio":remove_ratios,
        #     "remove_class":range(num_classes),
        #     "sampling_type":["targeted_informed"],
        #     "sampler_seed":[0],
        #     "sgd_seed":args.sgd_seed,
        # }
    ])
    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        torch.set_num_threads(1)
        # ctx = mp.get_context('spawn')
        with mp.Pool(num_processes) as pool:
            w_orig,training_time = train(data,args)
            retrain_partial = partial(retrain,data=data,args=args,w_orig=w_orig)
            print("Starting Parallel")
            for metrics, params in tqdm(pool.imap_unordered(retrain_partial,grid),total=len(grid)):    
                fp.write(dict_2_string(rows.copy(),args,params,metrics,training_time))
                fp.flush()
            
