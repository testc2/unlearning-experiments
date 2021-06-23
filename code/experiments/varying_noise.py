from functools import partial
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import lr_optimize_sgd,predict,predict_log_proba,lr_grad
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
from tqdm import tqdm

header  = [
    "method","lam","l2_norm", # removal method and regularization
    "num_removes","remove_seed","remove_type","remove_class","minibatch_size", # removal details 
    "sgd_seed","optim","step_size","lr_schedule", # training details
    "noise","noise_seed", # privacy noise details 
    "training_time","removal_time", # timings
    "norm","grad_residual", # paramteric metrics,
    "test_accuracy","test_f1_score","test_roc_score", # test metrics
    "remove_accuracy","remove_f1_score","remove_roc_score", # removed samples metrics
    "prime_accuracy","prime_f1_score","prime_roc_score", # remaining samples metrics
]
rows = OrderedDict({k:None for k in header})

def get_metrics(w,data,args):
    X_test = data["X_test"]
    X_remove = data["X_remove"]
    X_prime = data["X_prime"]
    y_test = data["y_test"]
    y_remove = data["y_remove"]
    y_prime = data["y_prime"]

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
    return metrics
    
def dict_2_string(row,args,params,data,w):
    row.update(params)
    row["method"] = args.method
    row["lam"] = args.lam
    row["optim"] = args.optim
    row["step_size"] = args.step_size
    row["lr_schedule"] = args.lr_schedule
    row["l2_norm"] = args.l2_norm
    row["num_removes"] = args.num_removes
    row["remove_seed"] = args.remove_seed
    row["remove_type"] = args.remove_type
    row["remove_class"] = args.remove_class
    metrics = get_metrics(w,data,args)
    row.update(metrics)
    print_str = ",".join([str(x) for x in row.values()])
    print_str += "\n"
    return print_str

def train_remove_guo(params,args,data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    torch.manual_seed(params["noise_seed"])
    guo_noise = params["noise"] * torch.randn(X_train.size(1),).float()
    start = time()
    w_guo = lr_optimize_sgd(X_train,y_train,params,args,guo_noise)
    training_time = time()-start
    start = time()
    w_guo_approx = remove_minibatch_pytorch(w_guo,data,params["minibatch_size"],args)
    removal_time = time()-start
    params.update({
        "training_time":training_time,
        "removal_time":removal_time,
    })
    return w_guo_approx, params

def train_scrub_golatkar(params,args,data):
    X_train = data["X_train"]
    y_train = data["y_train"] 
    start = time()
    w = lr_optimize_sgd(X_train,y_train,params,args)
    training_time = time()-start
    start = time()
    w_golatkar_approx,_ = scrub_minibatch_pytorch(w,data,params["minibatch_size"],args,params["noise"],params["noise_seed"])
    removal_time = time()-start
    params.update({
        "training_time":training_time,
        "removal_time":removal_time,
    })
    return w_golatkar_approx, params    

def train_remove_guo_ovr(params,args,data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    torch.manual_seed(params["noise_seed"])
    guo_noise = params["noise"] * torch.randn((X_train.size(1),y_train.size(1))).float()
    start = time()
    w_guo = lr_ovr_optimize_sgd(X_train,y_train,params,args,guo_noise)
    training_time = time()-start
    start = time()
    w_guo_approx = remove_ovr_minibatch_pytorch(w_guo,data,params["minibatch_size"],args)
    removal_time = time()-start
    params.update({
        "training_time":training_time,
        "removal_time":removal_time,
    })
    return w_guo_approx, params 

def train_scrub_golatkar_ovr(params,args,data):
    X_train = data["X_train"]
    y_train = data["y_train"] 
    start = time()
    w = lr_ovr_optimize_sgd(X_train,y_train,params,args)
    training_time = time()-start
    start = time()
    w_golatkar_approx,_ = scrub_ovr_minibatch_pytorch(w,data,params["minibatch_size"],args,params["noise"],params["noise_seed"])
    removal_time = time()-start
    params.update({
        "training_time":training_time,
        "removal_time":removal_time,
    })
    return w_golatkar_approx, params  
 
def execute_noise_exp(args,data):
    results_folder = args.results_dir/ args.dataset
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    if args.ovr:
        file_name = f"{args.method}_multi_{args.remove_type+args.suffix}.csv"
    else:
        file_name = f"{args.method}_binary_{args.remove_type+args.suffix}.csv"

    results_file = results_folder / file_name
    num_processes = args.num_processes
    grid = ParameterGrid({
        "noise":args.noise_levels,
        "noise_seed":range(args.num_seeds),
        "minibatch_size":args.minibatches,
        "sgd_seed":args.sgd_seed,
        }
        )
    if args.method == "Guo" and not args.ovr:
        train_func  = train_remove_guo
    elif args.method == "Guo" and args.ovr:
        train_func = train_remove_guo_ovr
    elif args.method == "Golatkar" and not args.ovr:
        train_func = train_scrub_golatkar
    elif args.method == "Golatkar" and args.ovr:
        train_func = train_scrub_golatkar_ovr
    else:
        raise ValueError(f"Method argument incorrect")
    
    torch.set_num_threads(1)
    train_partial = partial(train_func,data=data,args=args)
    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        with mp.Pool(num_processes) as pool:
            for w, params in tqdm(pool.imap_unordered(train_partial,grid),total=len(grid)):    
                fp.write(dict_2_string(rows.copy(),args,params,data,w))
                fp.flush()
        
