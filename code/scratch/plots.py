#%%
from IPython import get_ipython
from scipy.sparse import data

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
import seaborn as sns
from time import perf_counter as time
import json
import re
#%%

current_dir = project_dir/"code"/"scratch"
with open(current_dir/"training_config.json") as fp:
    training_configs = json.load(fp)
dataset = "MNIST"
ovr_str= "binary"
training_config = training_configs[f"{dataset}_{ovr_str}"]
remove_ratio=training_config["remove_ratios"][2]
deletion_batch_size = training_config["deletion_batch_size"]
gol_frames = []
for file in current_dir.glob("gol_*"):
    if not re.search(f"{deletion_batch_size}_{dataset}_{remove_ratio}",file.stem):
        continue
    df = pd.read_csv(file)
    gol_frames.append(df)
    
retrain_file = [file for file in current_dir.glob(f"retrain_*") if re.search(f"{dataset}_{remove_ratio}",file.stem)][0]
rettrain_bz = int(retrain_file.stem.split("_")[1])
retrain_df = pd.read_csv(retrain_file)
retrain_df = retrain_df.set_index("batch")

nothing_file = [file for file in current_dir.glob(f"nothing_*") if re.search(f"{dataset}_{remove_ratio}",file.stem)][0]
rettrain_bz = int(nothing_file.stem.split("_")[1])
nothing_df = pd.read_csv(nothing_file)

#%%

def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = np.array((numerator==0)&(denominator==0),ndmin=1)
    sae = np.array(numerator/denominator,ndmin=1)
    sae[both_zero] = 1 
    return sae*100
#%%`Ω
window_size = 15
for df in gol_frames+[nothing_df]:
    df["acc_dis"] = SAPE(df["cum_remove_accuracy"].values,retrain_df.loc[df.batch,"cum_remove_accuracy"].values)
    df["acc_dis_rolling_avg"] = df.acc_dis.rolling(window=window_size).mean()
    df["acc_dis_batch"] = SAPE(df["batch_remove_accuracy"].values,retrain_df.loc[df.batch,"batch_remove_accuracy"].values)
    df["acc_dis_batch_rolling_avg"] = df.acc_dis_batch.rolling(window=window_size).mean()
    df["acc_err"] = SAPE(df["test_accuracy"].values,retrain_df.loc[df.batch,"test_accuracy"].values)
    df["acc_err_rolling_avg"] = df.acc_err.rolling(window=window_size).mean()
    df["acc_err_init"] = SAPE(df["test_accuracy"].values,nothing_df.test_accuracy.values[0])
    df["cum_time"] = df.time.cumsum()
    df["acc_dis_cumsum"] = df.acc_dis.cumsum()
    df["acc_dis_cumsum_avg"] = df.acc_dis_cumsum.values/df.batch
    df["acc_err_cumsum"] = df.acc_err.cumsum()
    df["acc_err_cumsum_avg"] = df.acc_err_cumsum.values/df.batch

retrain_df["cum_time"] = retrain_df.time.cumsum()
retrain_df["acc_err_init"] = (nothing_df.test_accuracy.values[0]-retrain_df["test_accuracy"].values)/nothing_df.test_accuracy.values[0] *100
# %%

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,10))


df = pd.concat(gol_frames)
# df = pd.concat(gol_frames+[nothing_df])
df_full = pd.concat(gol_frames+[retrain_df.loc[gol_frames[0].batch].reset_index()])
# df = df[df.unlearning_bz.isin([10,50,100])]

ax1 = sns.lineplot(data=df_full,x="batch",y="cum_time",hue="method",ax=ax1)
ax1.legend(bbox_to_anchor=(1.05,0.5),loc="upper left")
ax1.set_xlabel("# deletions")
ax1.set_ylabel("Cumulative Running Time")
ax1.set_yscale("symlog")

ax2 = sns.lineplot(data=df,x="batch",y="acc_dis_rolling_avg",hue="method",ax=ax2,legend=False)
ax2.set_xlabel("# deletions")
ax2.set_ylabel("AccDis % (rolling avg)")

ax3 = sns.lineplot(data=df,x="batch",y="acc_err_rolling_avg",hue="method",ax=ax3,legend=False)
ax3.set_xlabel("# deletions")
ax3.set_ylabel("AccErr % (rolling avg)")
plt.suptitle(dataset)
#%%

df = pd.concat(gol_frames)
# df = pd.concat(gol_frames+[nothing_df])
# df = pd.concat(gol_frames+[retrain_df.loc[gol_frames[0].batch].reset_index()])
# df.pipeline_acc_err.fillna(0,inplace=True)

sns.lineplot(data=df,x="batch",y="acc_err_cumsum_avg",hue="method")
plt.legend(bbox_to_anchor=(1.05,0.5),loc="upper left")
# plt.axhline(y=0.1,linestyle="--",alpha=0.5,color="black")
# plt.axhline(y=0.5,linestyle="--",alpha=0.5,color="black")
# plt.axhline(y=1,linestyle="--",alpha=0.5,color="black")
plt.xlabel("# deletions")

# %%
