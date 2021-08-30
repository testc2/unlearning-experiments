#%%
from IPython import get_ipython
from torch.nn.functional import threshold

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
from methods.scrub import scrub_minibatch_pytorch, scrub_ovr_minibatch_pytorch, scrub
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
import math
import psutil
import os
#%%
def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = np.array((numerator==0)&(denominator==0),ndmin=1)
    sae = np.array(numerator/denominator,ndmin=1)
    sae[both_zero] = 1 
    return sae*100
#%%
# torch.set_num_threads(12)
# torch.set_num_interop_threads(1)
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss /1024 **2
#%%
args = parser.parse_args(["--optim","SGD","--step-size","1","dist","COVTYPE","--l2-norm"])
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
#%%
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
    start = time()
    w = lr_optimize_sgd_batch(X_train,y_train,params,args)
    training_time = time() - start
    print(f"Training Time: {training_time:.5f}s")
    print("Original Test Accuracy: ",accuracy_score(y_test,predict(w,X_test)))
    print("Original Test F1 Score: ",get_f1_score(y_test,predict(w,X_test)))


#%%
data = {
    "X_train":X_train,
    "X_test":X_test,
    "y_train":y_train,
    "y_test":y_test,
    }
remove_ratio = training_config["remove_ratios"][2]
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
    X_remove,y_remove,X_prime,y_prime = sample(data,remove_ratio,remove_class,sampler_seed=0,sampling_type=sampling_type)
    data["X_remove"] =X_remove
    data["X_prime"] =X_prime
    data["y_remove"] =y_remove
    data["y_prime"] =y_prime

X_remove = data["X_remove"]
y_remove = data["y_remove"]
X_prime = data["X_prime"]
y_prime = data["y_prime"]
#%%


def pipeline_guo(w,unlearning_bz,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime):
    w_approx = w.clone()
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    num_removes = X_remove.shape[0]
    guo_rows =[]
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
            "unlearning_bz":unlearning_bz,
            "num_removes":num_removes,
            "remove_ratio":remove_ratio,
            "sampling_type":sampling_type,
            "batch":min(batch+unlearning_bz,num_removes),
            "time":unlearn_time,
            "test_accuracy":accuracy_score(y_test,predict(w_approx,X_test)),
            "cum_remove_accuracy":accuracy_score(y_remove[:batch+unlearning_bz],predict(w_approx,X_remove[:batch+unlearning_bz])),
            "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_approx,X_batch_remove)),
            }
        )

    guo_df = pd.DataFrame(guo_rows)
    guo_df.to_csv(f"guo_{unlearning_bz}_{args.dataset}_{remove_ratio}.csv")

def pipeline_gol(w,unlearning_bz,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime):
    w_approx = w.clone()
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    num_removes = X_remove.shape[0]
    guo_rows =[]
    args.verbose = False
    test_acc_init = accuracy_score(y_test,predict(w,X_test))
    for batch in trange(0,num_removes,unlearning_bz):
        X_batch_remove = X_remove[batch:batch+unlearning_bz]
        y_batch_remove = y_remove[batch:batch+unlearning_bz]
        X_batch_prime = X_train_temp[batch+unlearning_bz:]
        y_batch_prime = y_train_temp[batch+unlearning_bz:]
        start = time()
        w_approx,added_noise = scrub(w_approx,X_batch_prime,y_batch_prime,args.lam,noise=0,noise_seed=0)
        unlearn_time = time()-start
        test_accuracy = accuracy_score(y_test,predict(w_approx,X_test))
        acc_err_init = SAPE(test_accuracy,test_acc_init)[0]
        guo_rows.append({
            "method":"Golatkar",
            "unlearning_bz":unlearning_bz,
            "num_removes":num_removes,
            "remove_ratio":remove_ratio,
            "sampling_type":sampling_type,
            "batch":min(batch+unlearning_bz,num_removes),
            "time":unlearn_time,
            "test_accuracy":accuracy_score(y_test,predict(w_approx,X_test)),
            "cum_remove_accuracy":accuracy_score(y_remove[:batch+unlearning_bz],predict(w_approx,X_remove[:batch+unlearning_bz])),
            "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_approx,X_batch_remove)),
            "pipeline_acc_err":acc_err_init
            }
        )

    guo_df = pd.DataFrame(guo_rows)
    guo_df.to_csv(f"gol_{unlearning_bz}_{args.dataset}_{remove_ratio}.csv")


def retrain(batch_size,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime):
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    retrain_rows =[]
    args.verbose = False
    for batch in trange(0,num_removes,batch_size):
        X_batch_remove = X_remove[batch:batch+batch_size]
        y_batch_remove = y_remove[batch:batch+batch_size]
        X_batch_prime = X_train_temp[batch+batch_size:]
        y_batch_prime = y_train_temp[batch+batch_size:]
        start = time()
        w_prime =  lr_optimize_sgd_batch(X_batch_prime,y_batch_prime,params,args)
        retrain_time = time() - start
        retrain_rows.append({
            "method":"retrain",
            "time":retrain_time,
            "num_removes":num_removes,
            "remove_ratio":remove_ratio,
            "sampling_type":sampling_type,
            "batch":min(batch+batch_size,num_removes),
            "test_accuracy":accuracy_score(y_test,predict(w_prime,X_test)),
            "cum_remove_accuracy":accuracy_score(y_remove[:batch+batch_size],predict(w_prime,X_remove[:batch+batch_size])),
            "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_prime,X_batch_remove)),
            }
        )
    retrain_df = pd.DataFrame(retrain_rows)
    print("Saving")
    retrain_df.to_csv(f"retrain_{batch_size}_{args.dataset}_{remove_ratio}.csv")

def do_nothing(w,batch_size,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime):
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    nothing_rows =[]
    args.verbose = False
    for batch in trange(0,num_removes,batch_size):
        X_batch_remove = X_remove[batch:batch+batch_size]
        y_batch_remove = y_remove[batch:batch+batch_size]
        X_batch_prime = X_train_temp[batch+batch_size:]
        y_batch_prime = y_train_temp[batch+batch_size:]
        nothing_rows.append({
            "method":"nothing",
            "time":0,
            "num_removes":num_removes,
            "remove_ratio":remove_ratio,
            "sampling_type":sampling_type,
            "batch":min(batch+batch_size,num_removes),
            "test_accuracy":accuracy_score(y_test,predict(w,X_test)),
            "cum_remove_accuracy":accuracy_score(y_remove[:batch+batch_size],predict(w,X_remove[:batch+batch_size])),
            "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w,X_batch_remove)),
            }
        )
    nothing_df = pd.DataFrame(nothing_rows)
    print("Saving")
    nothing_df.to_csv(f"nothing_{batch_size}_{args.dataset}_{remove_ratio}.csv")

def gol_acc_test_retrain(w,unlearning_bz,threshold,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime):
    w_approx = w.clone()
    # concatenate the remove and remaining 
    X_train_temp = torch.cat((X_remove,X_prime))
    y_train_temp = torch.cat((y_remove,y_prime))
    num_removes = X_remove.shape[0]
    gol_rows =[]
    args.verbose = False
    test_acc_init = accuracy_score(y_test,predict(w,X_test))
    for batch in trange(0,num_removes,unlearning_bz):
        retrain = False
        X_batch_remove = X_remove[batch:batch+unlearning_bz]
        y_batch_remove = y_remove[batch:batch+unlearning_bz]
        X_batch_prime = X_train_temp[batch+unlearning_bz:]
        y_batch_prime = y_train_temp[batch+unlearning_bz:]
        start = time()
        w_approx,added_noise = scrub(w_approx,X_batch_prime,y_batch_prime,args.lam,noise=0,noise_seed=0)
        running_time = time() - start
        # compute test accuracy
        test_accuracy = accuracy_score(y_test,predict(w_approx,X_test))
        # find the SAPE wrt to test accuracy of last checkpoint
        acc_err_init = SAPE(test_accuracy,test_acc_init)[0]
        # if the unlearned model exceeds acc_test threshold then retrain 
        if acc_err_init > threshold:
            retrain = True
            start = time()
            w_approx =  lr_optimize_sgd_batch(X_batch_prime,y_batch_prime,params,args)
            # add retraining time to run time
            running_time += (time()-start)
            # update checkpoint test accuracy
            test_acc_init = accuracy_score(y_test,predict(w_approx,X_test))
            
            test_accuracy = accuracy_score(y_test,predict(w_approx,X_test)) 
            acc_err_init = SAPE(test_accuracy,test_acc_init)[0]
            
        
        gol_rows.append({
            "method":f"Golatkar threshold: {threshold}%",
            "unlearning_bz":unlearning_bz,
            "num_removes":num_removes,
            "remove_ratio":remove_ratio,
            "sampling_type":sampling_type,
            "batch":min(batch+unlearning_bz,num_removes),
            "time":running_time,
            "test_accuracy":test_accuracy,
            "retrained":retrain,
            "cum_remove_accuracy":accuracy_score(y_remove[:batch+unlearning_bz],predict(w_approx,X_remove[:batch+unlearning_bz])),
            "batch_remove_accuracy":accuracy_score(y_batch_remove,predict(w_approx,X_batch_remove)),
            "pipeline_acc_err":acc_err_init
            }
        )

    guo_df = pd.DataFrame(gol_rows)
    guo_df.to_csv(f"gol_threshold_{threshold}_{unlearning_bz}_{args.dataset}_{remove_ratio}.csv")
#%%

deletion_batch_size = training_config["deletion_batch_size"]
do_nothing(w,deletion_batch_size,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
pipeline_gol(w,deletion_batch_size,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
gol_acc_test_retrain(w,deletion_batch_size,1,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
gol_acc_test_retrain(w,deletion_batch_size,0.5,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
gol_acc_test_retrain(w,deletion_batch_size,0.1,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)

# retrain(10,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
# bzs = [10,50,100,200,500,1000,2000]
# bzs = [1000,2000,3000,4000,5000,math.ceil(78463//2000)*1000,num_removes]
# for unlearning_bz in bzs:
    # pipeline_gol(w,unlearning_bz,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
    # pipeline_guo(w,unlearning_bz,args,sampling_type,remove_ratio,X_remove,X_prime,y_remove,y_prime)
