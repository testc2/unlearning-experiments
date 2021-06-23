from functools import partial
from os import error
from methods.multiclass_utils import predict_ovr, predict_proba_ovr, lr_ovr_optimize_sgd
from methods.pytorch_utils import (
    lr_optimize_sgd_batch,
    predict,
    predict_log_proba,
    lr_grad,
)
from methods.remove import remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from methods.scrub import scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch
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
    "method","lam","l2_norm",
    # removal details
    "remove_ratio","sampler_seed","minibatch_size","remove_class","sample_prob","num_removes","minibatch_fraction","sampling_type",  
    "sgd_seed","optim","step_size","lr_schedule","training_batch_size","num_steps",  # training details
    "noise","noise_seed",  # privacy noise details
    "training_time", "removal_time",  # running time details
    "norm","grad_residual",  # paramteric metrics,
    "test_accuracy","test_f1_score","test_roc_score",  # test metrics
    "remove_accuracy","remove_f1_score","remove_roc_score",  # removed samples metrics
    "prime_accuracy","prime_f1_score","prime_roc_score",  # remaining samples metrics
]

rows = OrderedDict({k: None for k in header})


def get_metrics(X_test, X_remove, X_prime, y_test, y_remove, y_prime, w, args):

    metrics = {}
    metrics["norm"] = float(w.norm())
    if not args.ovr:
        test_preds = predict(w, X_test)
        remove_preds = predict(w, X_remove)
        prime_preds = predict(w, X_prime)
        test_preds_log_proba = predict_log_proba(w, X_test)
        remove_preds_log_proba = predict_log_proba(w, X_remove)
        prime_preds_log_proba = predict_log_proba(w, X_prime)
        metrics["test_accuracy"] = accuracy_score(y_test, test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test, test_preds)
        metrics["test_roc_score"] = get_roc_score(y_test, test_preds_log_proba)

        metrics["remove_accuracy"] = accuracy_score(y_remove, remove_preds)
        metrics["remove_f1_score"] = get_f1_score(y_remove, remove_preds)
        metrics["remove_roc_score"] = get_roc_score(y_remove, remove_preds_log_proba)

        metrics["prime_accuracy"] = accuracy_score(y_prime, prime_preds)
        metrics["prime_f1_score"] = get_f1_score(y_prime, prime_preds)
        metrics["prime_roc_score"] = get_roc_score(y_prime, prime_preds_log_proba)

        metrics["grad_residual"] = float(lr_grad(w, X_prime, y_prime, args.lam).norm())

    else:
        test_preds = predict_ovr(w, X_test)
        remove_preds = predict_ovr(w, X_remove)
        prime_preds = predict_ovr(w, X_prime)
        test_proba = predict_proba_ovr(w, X_test)
        remove_proba = predict_proba_ovr(w, X_remove)
        prime_proba = predict_proba_ovr(w, X_prime)
        y_test_ = y_test.argmax(1)
        y_remove_ = y_remove.argmax(1)
        y_prime_ = y_prime.argmax(1)
        metrics["test_accuracy"] = accuracy_score(y_test_, test_preds)
        metrics["test_f1_score"] = get_f1_score(y_test_, test_preds, average="weighted")
        metrics["test_roc_score"] = get_roc_score(
            y_test_, test_proba, average="macro", multi_class="ovr"
        )

        metrics["remove_accuracy"] = accuracy_score(y_remove_, remove_preds)
        metrics["remove_f1_score"] = get_f1_score(
            y_remove_, remove_preds, average="weighted"
        )
        metrics["remove_roc_score"] = get_roc_score(
            y_remove_, remove_proba, average="macro", multi_class="ovr"
        )

        metrics["prime_accuracy"] = accuracy_score(y_prime_, prime_preds)
        metrics["prime_f1_score"] = get_f1_score(
            y_prime_, prime_preds, average="weighted"
        )
        metrics["prime_roc_score"] = get_roc_score(
            y_prime_, prime_proba, average="macro", multi_class="ovr"
        )

        metrics["grad_residual"] = np.mean(
            [
                float(lr_grad(w[:, k], X_prime, y_prime[:, k], args.lam).norm())
                for k in range(y_prime.size(1))
            ]
        )
    return metrics

def dict_2_string(row,args,params_list,metrics_list):
    strings = []
    method_row = row.copy()
    for params, metrics in zip(params_list,metrics_list):
        method_row.update(params)
        method_row["lam"] = args.lam
        method_row["optim"] = args.optim
        method_row["step_size"] = args.step_size
        method_row["num_steps"] = args.num_steps
        method_row["training_batch_size"] = args.batch_size
        method_row["lr_schedule"] = args.lr_schedule
        method_row["l2_norm"] = args.l2_norm
        method_row.update(metrics)
        # if len(method_row.keys())!=len(row.keys()):
            # error_columns = set(method_row.keys()).symmetric_difference(set(row.keys()))
            # raise ValueError(f"Extra columns, Check the error columns {error_columns}")
        print_str = ",".join([str(x) for x in method_row.values()])
        print_str += "\n"
        strings.append(print_str)
    return strings

def train_unlearn_guo(params, data, args):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_remove, y_remove, X_prime, y_prime = sample(
        data,
        params["remove_ratio"],
        params["remove_class"],
        params["sampler_seed"],
        params["sampling_type"],
    )
    num_removes = int(data["X_train"].size(0) * params["remove_ratio"])
    minibatch_size = num_removes // params["minibatch_fraction"]
    # train a perturbed loss model
    torch.manual_seed(params["noise_seed"])
    start = time()
    if args.ovr:
        guo_noise = (
            params["noise"] * torch.randn(X_train.size(1), y_train.size(1)).float()
        )
        w_guo = lr_ovr_optimize_sgd(X_train, y_train, params, args, guo_noise)
    else:
        guo_noise = (
            params["noise"]* torch.randn(X_train.size(1),).float()
        )
        w_guo = lr_optimize_sgd_batch(X_train, y_train, params, args, guo_noise)
    training_time = time() - start
    # Removal step
    start = time()
    if args.ovr:
        w_approx = remove_ovr_minibatch_pytorch(
            w_guo, data, minibatch_size, args, X_remove, X_prime, y_remove, y_prime
        )
    else:
        w_approx = remove_minibatch_pytorch(
            w_guo, data, minibatch_size, args, X_remove, X_prime, y_remove, y_prime
        )
    removal_time = time() - start

    params.update(
        {
            "training_time": training_time,
            "removal_time": removal_time,
            "num_removes": X_remove.size(0),
            "minibatch_size": minibatch_size,
        }
    )
    metrics = get_metrics(
        data["X_test"],
        X_remove,
        X_prime,
        data["y_test"],
        y_remove,
        y_prime,
        w_approx,
        args,
    )
    return metrics, params


def train_unlearn_gol(params, data, args):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_remove, y_remove, X_prime, y_prime = sample(
        data,
        params["remove_ratio"],
        params["remove_class"],
        params["sampler_seed"],
        params["sampling_type"],
    )
    num_removes = int(data["X_train"].size(0) * params["remove_ratio"])
    minibatch_size = num_removes // params["minibatch_fraction"]
    start = time()
    # train original model on whole dataset
    if args.ovr:
        w_orig = lr_ovr_optimize_sgd(X_train, y_train, params, args, b=None)
    else:
        w_orig = lr_optimize_sgd_batch(X_train, y_train, params, args, b=None)
    training_time = time() - start
    # Removal step
    start = time()
    if args.ovr:
        w_approx, _ = scrub_ovr_minibatch_pytorch(
            w_orig,
            data,
            minibatch_size,
            args,
            noise=params["noise"],
            noise_seed=params["noise_seed"],
            X_remove=X_remove,
            X_prime=X_prime,
            y_remove=y_remove,
            y_prime=y_prime,
        )
    else:
        w_approx, _ = scrub_minibatch_pytorch(
            w_orig,
            data,
            minibatch_size,
            args,
            noise=params["noise"],
            noise_seed=params["noise_seed"],
            X_remove=X_remove,
            X_prime=X_prime,
            y_remove=y_remove,
            y_prime=y_prime,
        )
    removal_time = time() - start

    params.update(
        {
            "training_time": training_time,
            "removal_time": removal_time,
            "num_removes": X_remove.size(0),
            "minibatch_size": minibatch_size,
        }
    )
    metrics = get_metrics(
        data["X_test"],
        X_remove,
        X_prime,
        data["y_test"],
        y_remove,
        y_prime,
        w_approx,
        args,
    )
    return metrics, params


def baseline(params, data, args):
    X_remove, y_remove, X_prime, y_prime = sample(
        data,
        params["remove_ratio"],
        params["remove_class"],
        params["sampler_seed"],
        params["sampling_type"],
    )
    num_removes = X_remove.size(0)
    if params["method"] == "baseline_Guo":
        # train a baseline with same amount of perturbation
        torch.manual_seed(params["noise_seed"])
        start = time()
        if args.ovr:
            guo_noise = (
                params["noise"] * torch.randn(X_prime.size(1), y_prime.size(1)).float()
            )
            w_guo = lr_ovr_optimize_sgd(X_prime, y_prime, params, args, guo_noise)
        else:
            guo_noise = (
                params["noise"]* torch.randn(X_prime.size(1),).float()
            )
            w_guo = lr_optimize_sgd_batch(X_prime, y_prime, params, args, guo_noise)
        training_time = time() - start

        params.update(
            {
                "training_time": training_time,
                "removal_time": training_time,
                "num_removes": num_removes,
            }
        )
        metrics = get_metrics(
            data["X_test"],
            X_remove,
            X_prime,
            data["y_test"],
            y_remove,
            y_prime,
            w_guo,
            args,
        )
        return metrics, params
    else:
        minibatch_size = num_removes // params["minibatch_fraction"]
        # train a baseline on the remaining samples with no noise
        start = time()
        if args.ovr:
            w_baseline = lr_ovr_optimize_sgd(X_prime, y_prime, params, args)
        else:
            w_baseline = lr_optimize_sgd_batch(X_prime, y_prime, params, args)
        training_time = time() - start
        # gather metrics for baseline without any noise 
        baseline_params = params.copy()
        baseline_params.pop("minibatch_fraction")
        baseline_params.pop("noise")
        baseline_params.pop("noise_seed")
        baseline_params.update(
            {
                "training_time": training_time,
                "removal_time": training_time,
                "num_removes": num_removes,
            }
        )
        baseline_metrics = get_metrics(
            data["X_test"],
            X_remove,
            X_prime,
            data["y_test"],
            y_remove,
            y_prime,
            w_baseline,
            args,
        )
        # at the same time gather gol baseline and metrics
        # at the required level of noise
        start = time()
        if args.ovr:
            w_gol_baseline, _ = scrub_ovr_minibatch_pytorch(
                w_baseline,
                data,
                minibatch_size,
                args,
                noise=params["noise"],
                noise_seed=params["noise_seed"],
                X_remove=X_remove,
                X_prime=X_prime,
                y_remove=y_remove,
                y_prime=y_prime,
            )
        else:
            w_gol_baseline, _ = scrub_minibatch_pytorch(
                w_baseline,
                data,
                minibatch_size,
                args,
                noise=params["noise"],
                noise_seed=params["noise_seed"],
                X_remove=X_remove,
                X_prime=X_prime,
                y_remove=y_remove,
                y_prime=y_prime,
            )
        removal_time = time() - start
        gol_training_time = training_time + removal_time
        gol_params = params.copy()
        gol_params["method"] = "baseline_Golatkar"
        gol_params.update(
            {
                "training_time": gol_training_time,
                "removal_time": gol_training_time,
                "num_removes": num_removes,
                "minibatch_size":minibatch_size,
            }
        )
        gol_baseline_metrics = get_metrics(
            data["X_test"],
            X_remove,
            X_prime,
            data["y_test"],
            y_remove,
            y_prime,
            w_gol_baseline,
            args,
        )
        # return all metrics and params
        return [baseline_metrics, gol_baseline_metrics], [baseline_params, gol_params]


def func(params, args, data):
    method = params["method"]
    if method == "Guo":
        metrics, params = train_unlearn_guo(params, data, args)
        return [metrics], [params]
    elif method == "Golatkar":
        metrics, params = train_unlearn_gol(params, data, args)
        return [metrics], [params]
    elif method == "baseline_Guo":
        metrics, params = baseline(params, data, args)
        return [metrics], [params]
    # collect baseline and Golatkar metrics at the same time
    elif method == "baseline":
        metrics_list, params_list = baseline(params, data, args)
        return metrics_list, params_list
    else:
        raise ValueError(f"Mehtod={method} is wrong. Check options")


def execute_unlearning_exp(args,data):
    results_folder = args.results_dir/ args.dataset
    if not results_folder.exists() : 
        results_folder.mkdir(parents=True,exist_ok=True)
    
    if args.ovr:
        num_classes = data["y_train"].size(1)
        remove_class = 3 # fix removed class as 3
        file_name = f"Unlearn_multi{args.suffix}.csv"
    else:
        num_classes = len(data["y_train"].unique())
        remove_class = 0
        file_name = f"Unlearn_binary{args.suffix}.csv"
    
    results_file = results_folder / file_name
    num_processes = args.num_processes
    param_grid_list= [
        {
            # removal sampling params
            "sampling_type":["targeted_informed"],
            "remove_ratio":args.remove_ratios,
            "remove_class":[remove_class],
            "sampler_seed":[0],
            "sgd_seed":args.sgd_seed,
            # removal step params
            "method":["Guo","Golatkar"],
            "minibatch_fraction":args.minibatch_fractions,
            "noise":args.noise_levels,
            "noise_seed":range(args.num_noise_seeds),
        },
        {
            "method":["baseline_Guo"],
            "remove_ratio":args.remove_ratios,
            "sampling_type":["targeted_informed"],
            "remove_class":[remove_class],
            "sampler_seed":[0],
            "sgd_seed":args.sgd_seed,
            "noise":args.noise_levels,
            "noise_seed":range(args.num_noise_seeds)
        },
        {
            "method":["baseline"],
            "remove_ratio":args.remove_ratios,
            "sampling_type":["targeted_informed"],
            "remove_class":[remove_class],
            "sampler_seed":[0],
            "sgd_seed":args.sgd_seed,
            "minibatch_fraction":args.minibatch_fractions,
            "noise":args.noise_levels,
            "noise_seed":range(args.num_noise_seeds)

        }
    
    ]
    
    param_grid = ParameterGrid(param_grid_list)
    print(len(param_grid),num_processes)
    with open(results_file,mode=args.overwrite_mode,encoding="utf-8") as fp:
        if args.overwrite_mode == "w":
            fp.write(",".join(rows.keys())+"\n")
        torch.set_num_threads(1)
        unlearn_partial = partial(func,data=data,args=args)
        with mp.Pool(num_processes) as pool:
            print("Starting Parallel")
            for metrics_list, params_list in tqdm(pool.imap_unordered(unlearn_partial,param_grid),total=len(param_grid)):    
                strings = dict_2_string(rows,args,params_list,metrics_list)
                [fp.write(s) for s in strings]
                fp.flush()