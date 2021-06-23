from functools import partial
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import lr_optimize_sgd_batch,predict,predict_log_proba,lr_grad
from methods.scrub import scrub
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

header  = [
    "method","lam","l2_norm", # removal method and regularization
    "sgd_seed","optim","step_size","lr_schedule","training_batch_size","num_steps", # training details
    "noise","noise_seed", # privacy noise details 
    "perturb_time",# perturbation details
    "norm","grad_residual","model_diff", # paramteric metrics,
    "test_accuracy","test_f1_score","test_roc_score", # test metrics
]

rows = OrderedDict({k:None for k in header})

def get_metrics(w,w_baseline,data,args):
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_train = data["X_train"]
    y_train = data["y_train"]

    metrics ={}
    metrics["norm"] = float(w.norm())
    if not args.ovr:
        test_preds = predict(w,X_test)
        test_preds_log_proba = predict_log_proba(w,X_test)
        metrics["test_accuracy"] = accuracy_score(y_test,test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test,test_preds)
        metrics["test_roc_score"] = get_roc_score(y_test,test_preds_log_proba)
        metrics["model_diff"] = (w-w_baseline).norm(p=2).item()
        metrics["grad_residual"] = float(lr_grad(w,X_train,y_train,args.lam).norm())

    else:
        test_preds = predict_ovr(w,X_test)
        test_proba = predict_proba_ovr(w,X_test)
        y_test_ = y_test.argmax(1)
        metrics["test_accuracy"] = accuracy_score(y_test_,test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test_,test_preds,average="weighted")
        metrics["test_roc_score"] = get_roc_score(y_test_,test_proba,average="macro",multi_class="ovr")
        metrics["model_diff"] = (w-w_baseline).norm(p=2).item()
        metrics["grad_residual"] = np.mean([float(lr_grad(w[:,k],X_train,y_train[:,k],args.lam).norm())for k in range(y_train.size(1))])
    return metrics


def dict_2_string(row,args,data,w_baseline,w_guo,w_golatkar,stats,params):
    strings = []
    method_row = row.copy()
    for weights, method in zip([w_baseline,w_guo,w_golatkar],["baseline","Guo","Golatkar"]):
        method_row.update(params)
        method_row["method"] = method
        method_row["lam"] = args.lam
        method_row["optim"] = args.optim
        method_row["step_size"] = args.step_size
        method_row["num_steps"] = args.num_steps
        method_row["training_batch_size"] = args.batch_size
        method_row["lr_schedule"] = args.lr_schedule
        method_row["l2_norm"] = args.l2_norm
        method_row["perturb_time"] = stats[f"{method}_perturb_time"]
        metrics = get_metrics(weights,w_baseline,data,args)
        method_row.update(metrics)
        print_str = ",".join([str(x) for x in method_row.values()])
        print_str += "\n"
        strings.append(print_str)
    return strings


def perturb(params,args,data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    # train a baseline model
    start = time()
    if args.ovr:
        w_baseline = lr_ovr_optimize_sgd(X_train,y_train,params,args,b=None)
    else:    
        w_baseline = lr_optimize_sgd_batch(X_train,y_train,params,args,b=None)
    baseline_train_time = time() - start
    # train a perturbed loss model
    torch.manual_seed(params["noise_seed"])
    guo_noise = params["noise"] * torch.randn_like(w_baseline).float()
    start = time()
    if args.ovr:
        w_guo = lr_ovr_optimize_sgd(X_train,y_train,params,args,guo_noise)
    else:
        w_guo = lr_optimize_sgd_batch(X_train,y_train,params,args,guo_noise)
    guo_perturb_time = time()-start
    start = time()
    if args.ovr:
        total_added_noise = torch.zeros_like(w_baseline)    
        for k in range(y_train.size(1)):
            _,noise_scrub = scrub(w_baseline[:,k].clone(),X_train,y_train[:,k],args.lam,params["noise"],params["noise_seed"])
            total_added_noise[:,k]=noise_scrub
        w_gol = w_baseline.clone()+total_added_noise
    else:
        _,noise_scrub = scrub(w_baseline.clone(),X_train,y_train,args.lam,params["noise"],params["noise_seed"])
        w_gol = w_baseline.clone()+noise_scrub
    gol_perturb_time = time() - start
    stats = {
        "Golatkar_perturb_time":gol_perturb_time,
        "Guo_perturb_time":guo_perturb_time,
        "baseline_perturb_time":baseline_train_time,
    }
    return w_baseline, w_guo, w_gol, stats, params



def execute_perturbation_exp(args,data):
    results_folder = args.results_dir/ args.dataset
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    
    if args.ovr:
        file_name = f"Perturbation_multi{args.suffix}.csv"
    else:
        file_name = f"Perturbation_binary{args.suffix}.csv"
 
    results_file = results_folder / file_name
    num_processes = args.num_processes
    grid = ParameterGrid([{
        "sgd_seed":args.sgd_seed,
        "noise":args.noise_levels,
        "noise_seed":range(args.num_seeds),                    }
    ])
    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        torch.set_num_threads(1)
        perturb_partial = partial(perturb,data=data,args=args)
        with mp.Pool(num_processes) as pool:
            for returns in tqdm(pool.imap_unordered(perturb_partial,grid),total=len(grid)):    
                strings = dict_2_string(rows.copy(),args,data,*returns)
                [fp.write(s) for s in strings]
                fp.flush()
            


