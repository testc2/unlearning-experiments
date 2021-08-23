#%%
from IPython import get_ipython

if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / "data"
code_dir = project_dir/ "code"

import sys
sys.path.append(str(code_dir.resolve()))
from run_exp import parser
from methods.common_utils import get_remove_prime_splits, load_cifar, load_epsilon, load_higgs, load_mnist, create_toy_dataset, load_rcv1, load_covtype, load_sensIT
from methods.pytorch_utils import lr_grad,lr_hessian_inv, lr_optimize_lbfgs,lr_optimize_sgd,predict,lr_optimize_lbfgs,lr_optimize_sgd_batch
from methods.multiclass_utils import predict_ovr,lr_ovr_optimize_sgd
from methods.scrub import scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch
from methods.remove import batch_remove, remove_minibatch_pytorch, remove_ovr_minibatch_pytorch
from sklearn.metrics import accuracy_score, classification_report
from methods.common_utils import get_f1_score
from experiments.removal_ratio import sample, train
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch.multiprocessing as  mp
import torch
import matplotlib.pyplot as plt
from collections import Counter
from time import perf_counter as time
import json
#%%
args = parser.parse_args(["--optim","SGD","--step-size","1","dist","MNIST","--l2-norm"])
# args = parser.parse_args(["--optim","Adam","remove","COVTYPE"])
if args.dataset == "MNIST":
    X_train, X_test, y_train, y_test = load_mnist(data_dir,ovr=args.ovr,l2_norm=args.l2_norm)
elif args.dataset == "COVTYPE":
    X_train, X_test, y_train, y_test = load_covtype(data_dir,ovr=args.ovr,l2_norm=args.l2_norm)
elif args.dataset == "HIGGS":
    X_train, X_test, y_train, y_test = load_higgs(data_dir,l2_norm=args.l2_norm)
elif args.dataset == "CIFAR":
    X_train, X_test, y_train, y_test = load_cifar(data_dir,l2_norm=args.l2_norm,ovr=args.ovr)
elif args.dataset == "EPSILON":
    X_train, X_test, y_train, y_test = load_epsilon(data_dir,l2_norm=args.l2_norm)

with open("training_config.json") as fp:
    training_configs = json.load(fp)
ovr_str = "binary" if not args.ovr else "ovr"
training_config = training_configs[f"{args.dataset}_{ovr_str}"]

params= dict(sgd_seed=0)
args.verbose = True
args.num_steps = training_config["epochs"]
args.batch_size = training_config["bz"]
args.step_size = training_config["lr"]
args.lr_schedule = False
if args.ovr:
    w = lr_ovr_optimize_sgd(X_train,y_train,params,args)
    print("Original Test F1 Score: ",get_f1_score(y_test.argmax(1),predict_ovr(w,X_test),average="weighted"))
    print("Original Test Accuracy: ",get_accuracy(y_test.argmax(1),predict_ovr(w,X_test)))
else:
    noise_std = 0
    torch.manual_seed(0)
    b = torch.randn(X_train.size(1))*noise_std
    w = lr_optimize_sgd_batch(X_train,y_train,params,args)
    print("Original Test Accuracy: ",accuracy_score(y_test,predict(w,X_test)))
    print("Original Test F1 Score: ",get_f1_score(y_test,predict(w,X_test)))

#%%
data = {
    "X_train":X_train,
    "X_test":X_test,
    "y_train":y_train,
    "y_test":y_test,
    }
remove_ratio = training_config["remove_ratios"][0]
num_removes = int(X_train.size(0)*remove_ratio)
sampling_type = "targeted_informed"
print(f"Deletion Ratio: {remove_ratio} Deletion #: {num_removes}")
if not args.ovr :
    remove_class = 0
    sample_prob = 1
    X_remove,y_remove,X_prime,y_prime = sample(data,remove_ratio,remove_class,sampler_seed=0,sampling_type=sampling_type)
    data["X_remove"] =X_remove
    data["X_prime"] =X_prime
    data["y_remove"] =y_remove
    data["y_prime"] =y_prime
else:
    num_classes = len(y_train.argmax(1).unique())
    remove_class = 0
    sample_prob = 1
    X_remove,y_remove,X_prime,y_prime = sample(data,remove_ratio,remove_class,sampler_seed=0,sampling_type="targeted_informed")
    data["X_remove"] =X_remove
    data["X_prime"] =X_prime
    data["y_remove"] =y_remove
    data["y_prime"] =y_prime

X_remove = data["X_remove"]
y_remove = data["y_remove"]
X_prime = data["X_prime"]
y_prime = data["y_prime"]
#%%
method = "Guo"
unlearning_bz = 10
w_approx = w.clone()
# concatenate the remove and remaining 
X_train_temp = torch.cat((X_remove,X_prime))
y_train_temp = torch.cat((y_remove,y_prime))
guo_rows =[]
retrain_rows =[]
args.verbose = False
for batch in trange(0,num_removes,unlearning_bz):
    X_batch_remove = X_remove[batch:batch+unlearning_bz]
    y_batch_remove = y_remove[batch:batch+unlearning_bz]
    X_batch_prime = X_train_temp[batch+unlearning_bz:]
    y_batch_prime = y_train_temp[batch+unlearning_bz:]
    start = time()
    w_approx = batch_remove(w_approx,X_batch_prime,X_batch_remove,y_batch_prime,y_batch_remove,args.lam)
    unlearn_time = time()-start
    guo_rows.append({
        "method":"Guo",
        "batch":batch,
        "time":unlearn_time,
        "test_accuracy":accuracy_score(y_test,predict(w_approx,X_test)),
        "cum_remove_accuracy":accuracy_score(y_remove[:batch+unlearning_bz],predict(w_approx,X_remove[:batch+unlearning_bz])),
        "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_approx,X_batch_remove)),
        }
    )
    start = time()
    w_prime =  lr_optimize_sgd_batch(X_batch_prime,y_batch_prime,params,args)
    retrain_time = time() - start
    retrain_rows.append({
        "method":"retrain",
        "batch":batch,
        "time":retrain_time,
        "test_accuracy":accuracy_score(y_test,predict(w_prime,X_test)),
        "cum_remove_accuracy":accuracy_score(y_remove[:batch+unlearning_bz],predict(w_prime,X_remove[:batch+unlearning_bz])),
        "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_prime,X_batch_remove)),
        }
    )


guo_df = pd.Dataframe(guo_rows)
retrain_df = pd.DataFrame(retrain_rows)

#%%
print(f"{method} full batch Remove Accuracy({unlearning_bz}): {accuracy_score(y_remove,predict(w_approx,X_remove)):>8.5f}")
print(f"{method} full batch Test Accuracy: {accuracy_score(y_test,predict(w_approx,X_test)):>14.5f}\n")

#%%
if args.ovr:
    args.optim = "Adam"
    args.step_size = 0.01
    args.num_steps = 20
    args.batch_size = 512
    w_prime =  lr_ovr_optimize_sgd(data["X_prime"],data["y_prime"],params,args)
    print("Prime Remove F1 Score: ",get_f1_score(y_remove.argmax(1),predict_ovr(w_prime,X_remove),average="weighted"))
    print("Prime Test F1 Score: ",get_f1_score(y_test.argmax(1),predict_ovr(w_prime,X_test),average="weighted"))
    w_approx = remove_ovr_minibatch_pytorch(w,data,num_removes,args)
    # w_approx,_ = scrub_ovr_minibatch_pytorch(w,data,num_removes,args,noise=0,noise_seed=0)
    print(f"Guo full batch ({num_removes}): ",get_f1_score(y_remove.argmax(1),predict_ovr(w_approx,X_remove),average="weighted"))
    print("Guo full batch Test F1 Score: ",get_f1_score(y_test.argmax(1),predict_ovr(w_approx,X_test),average="weighted"))
    w_approx_2 = remove_ovr_minibatch_pytorch(w,data,num_removes//2,args)
    # w_approx_2,_ = scrub_ovr_minibatch_pytorch(w,data,num_removes//2,args,noise=0,noise_seed=0)
    print(f"Guo batch/2 {num_removes//2}: ",get_f1_score(y_remove.argmax(1),predict_ovr(w_approx_2,X_remove),average="weighted"))
    print(f"Guo batch/2 {num_removes//2} test F1 Score:",get_f1_score(y_test.argmax(1),predict_ovr(w_approx_2,X_test),average="weighted"))
    w_approx_4 = remove_ovr_minibatch_pytorch(w,data,num_removes//4,args)
    # w_approx_4,_ = scrub_ovr_minibatch_pytorch(w,data,num_removes//4,args,noise=0,noise_seed=0)
    print(f"Guo batch/4 {num_removes//4}: ",get_f1_score(y_remove.argmax(1),predict_ovr(w_approx_4,X_remove),average="weighted"))
    print(f"Guo batch/4 {num_removes//4} test F1 Score:",get_f1_score(y_test.argmax(1),predict_ovr(w_approx_4,X_test),average="weighted"))
    w_approx_8 = remove_ovr_minibatch_pytorch(w,data,num_removes//8,args)
    # w_approx_8,_ = scrub_ovr_minibatch_pytorch(w,data,num_removes//8,args,noise=0,noise_seed=0)
    print(f"Guo batch/8 {num_removes//8}: ",get_f1_score(y_remove.argmax(1),predict_ovr(w_approx_8,X_remove),average="weighted"))
    print(f"Guo batch/8 {num_removes//8} test F1 Score:",get_f1_score(y_test.argmax(1),predict_ovr(w_approx_8,X_test),average="weighted"))
    # w_approx_16 = remove_ovr_minibatch_pytorch(w,data,num_removes//16,args)
    # w_approx_16,_ = scrub_ovr_minibatch_pytorch(w,data,num_removes//16,args,noise=0,noise_seed=0)
    # print(f"Guo batch/16 {num_removes//16}: ",accuracy_score(y_remove.argmax(1),predict_ovr(w_approx_16,X_remove)))
    # print(f"Guo batch/16 {num_removes//16} test accuracy:",accuracy_score(y_test.argmax(1),predict_ovr(w_approx_16,X_test)))
else:
    # args.optim = "Adam"
    # args.step_size = 0.01
    # args.num_steps = 100
    w_prime =  lr_optimize_sgd_batch(X_prime,y_prime,params,args)
    print("Prime Remove Accuracy: ",accuracy_score(y_remove,predict(w_prime,X_remove)))
    print("Prime Test Accuracy: ",accuracy_score(y_test,predict(w_prime,X_test)))
#%%    
    method = "Guo"
    if method == "Guo":
        w_approx = remove_minibatch_pytorch(w,data,num_removes,args)
        w_approx_2 = remove_minibatch_pytorch(w,data,num_removes//2,args)
        w_approx_4 = remove_minibatch_pytorch(w,data,num_removes//4,args)
        w_approx_8 = remove_minibatch_pytorch(w,data,num_removes//8,args)
        # w_approx_16 = remove_minibatch_pytorch(w,data,num_removes//16,args)
        # w_approx_100 = remove_minibatch_pytorch(w,data,100,args)
        # w_approx_50 = remove_minibatch_pytorch(w,data,50,args)
    elif method == "Golatkar":
        w_approx,_ = scrub_minibatch_pytorch(w,data,num_removes,args,noise=0,noise_seed=0)
        w_approx_2,_ = scrub_minibatch_pytorch(w,data,num_removes//2,args,noise=0,noise_seed=0)
        w_approx_4,_ = scrub_minibatch_pytorch(w,data,num_removes//4,args,noise=0,noise_seed=0)
        w_approx_8,_ = scrub_minibatch_pytorch(w,data,num_removes//8,args,noise=0,noise_seed=0)
        # w_approx_16,_ = scrub_minibatch_pytorch(w,data,num_removes//16,args,noise=0,noise_seed=0)
        # w_approx_100,_ = scrub_minibatch_pytorch(w,data,100,args,noise=0,noise_seed=0)
        # w_approx_50,_ = scrub_minibatch_pytorch(w,data,50,args,noise=0,noise_seed=0)        
    print(f"{method} full batch ({num_removes}): {accuracy_score(y_remove,predict(w_approx,X_remove)):>20.5f}")
    print(f"{method} full batch Test Accuracy: {accuracy_score(y_test,predict(w_approx,X_test)):>14.5f}\n")
    print(f"{method} batch/2 ({num_removes//2}): {accuracy_score(y_remove,predict(w_approx_2,X_remove)):>25.5f}")
    print(f"{method} batch/2 ({num_removes//2}) test accuracy:{accuracy_score(y_test,predict(w_approx_2,X_test)):>12.5f}\n")
    print(f"{method} batch/4 ({num_removes//4}): {accuracy_score(y_remove,predict(w_approx_4,X_remove)):>25.5f}")
    print(f"{method} batch/4 ({num_removes//4}) test accuracy:{accuracy_score(y_test,predict(w_approx_4,X_test)):>12.5f}\n")
    print(f"{method} batch/8 ({num_removes//8}): {accuracy_score(y_remove,predict(w_approx_8,X_remove)):>25.5f}")
    print(f"{method} batch/8 ({num_removes//8}) test accuracy:{accuracy_score(y_test,predict(w_approx_8,X_test)):>12.5f}\n")
    # print(f"{method} batch/16 ({num_removes//16}): {accuracy_score(y_remove,predict(w_approx_16,X_remove)):>25.5f}")
    # print(f"{method} batch/16 ({num_removes//16}) test accuracy:{accuracy_score(y_test,predict(w_approx_16,X_test)):>12.5f}\n")
    # print(f"{method} batch=100 : {accuracy_score(y_remove,predict(w_approx_100,X_remove)):>25.5f}")
    # print(f"{method} batch=100 test accuracy:{accuracy_score(y_test,predict(w_approx_100,X_test)):>12.5f}\n")
    # print(f"{method} batch=50 : {accuracy_score(y_remove,predict(w_approx_50,X_remove)):>25.5f}")
    # print(f"{method} batch=50 test accuracy:{accuracy_score(y_test,predict(w_approx_50,X_test)):>12.5f}\n")
    # %%