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
import sys
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str((project_dir/"code").resolve()))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from parse import get_dg_remove_ratio_frames
from plotting.ratio_remove_plots import load_dfs as ratio_load_dfs
from scipy.optimize import curve_fit
from scipy.stats import pearsonr,spearmanr
import json
#%%
mpl.rcParams["figure.dpi"]=100
mpl.rcParams["font.size"]=10
save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
# %%
def SMAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = (numerator==0)&(denominator==0)
    sae = numerator/denominator
    if sae.size > 1:
        sae[both_zero] = 0 
    elif both_zero:
        sae = np.array(0)
    return sae*100

# %%
def gather_dfs(results_dir:Path):
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    df_list = {dataset:load_dfs(results_dir,dataset,ovr_str) for dataset,ovr_str in zip(datasets,ovr_strs) if ovr_str == "binary"}
    return df_list

def combine_sampling_dfs(df_dict:dict,sampling_type):
    l = []
    for dataset,df in df_dict.items():
        temp = df.reset_index().groupby(["remove_ratio","sampling_type","minibatch_fraction"]).mean().xs(1,level=2)
        temp = temp.xs(sampling_type,level=-1)
        temp["dataset"] = dataset
        l.append(temp)
    return pd.concat(l)

def prop_predict(x,c):
    return c*x

def mse(y,y_pred):
    return ((y-y_pred)**2).mean()

def scatter_metric_eps(df,sampling_type:str,ax=None):
    if ax is None:
        fig,ax = plt.subplots()
    # select minibatch fraction 
    fraction = 1
    data_sampling_type=df.reset_index().groupby(["remove_ratio","sampling_type","minibatch_fraction"]).mean().xs(fraction,level=2)
    corr = data_sampling_type.groupby(level=1).apply(lambda x: pd.DataFrame({"Pearson Corr":pearsonr(x.baseline_eps,x.origin_eps),"Spearman Corr":spearmanr(x.baseline_eps,x.origin_eps)})).xs(0,level=-1)
    if sampling_type == "all":
        sns.scatterplot(data=data_sampling_type.reset_index(),x="origin_eps",y="baseline_eps",hue="sampling_type",ax=ax)
    else:
        data_sampling_type = data_sampling_type.xs(sampling_type,level=-1)
        # sns.regplot(data=data_sampling_type.reset_index(),x="origin_eps",y="baseline_eps",ax=ax,order=1)
        # sns.residplot(data=data_sampling_type.reset_index(),x="origin_eps",y="baseline_eps",ax=ax,order=1)
        sns.scatterplot(data=data_sampling_type.reset_index(),x="origin_eps",y="baseline_eps",ax=ax)
        
        
        corr = corr.loc[sampling_type].values

        print(data_sampling_type["baseline_eps"])
        print(data_sampling_type["origin_eps"])

        # calculate the slope as y/x at largest deletion ratio. c = [Acc Dis/ Acc Err]
        slope = data_sampling_type["baseline_eps"].values[-1]/data_sampling_type["origin_eps"].values[-1]
        # slope = 2.10309079
        print(data_sampling_type["baseline_eps"].values,prop_predict(data_sampling_type["origin_eps"],slope).values)
        ax.plot(data_sampling_type["origin_eps"],prop_predict(data_sampling_type["origin_eps"],slope))
        print(f'MSE: {mse(data_sampling_type["baseline_eps"].values,prop_predict(data_sampling_type["origin_eps"].values,slope)):.4f}')
        # print(np.polyfit(data_sampling_type["origin_eps"],data_sampling_type["baseline_eps"],deg=1))
        # slope=0
        ax.set_xscale("log")
        ax.set_yscale("log")
        corr = np.r_[corr,slope]
    
    # ax.set_xlabel("Acc Err wrt initial model")
    # ax.set_ylabel("Acc Dis wrt fully-retrained model")
    
    ax.set_xlabel("Abs Err wrt initial model")
    ax.set_ylabel("Abs Dis ")

    return ax,corr

def load_dfs(results_dir:Path,dataset:str,ovr_str:str,v2=False):
    baseline_df,guo_df,gol_df,_,_ = ratio_load_dfs(results_dir,dataset,ovr_str,suffix="_when_to_retrain",plot_deltagrad=False)
    if ovr_str == "multi":
        threshold = 0.1
    else:
        threshold = 0.45
    baseline_df = baseline_df[baseline_df.remove_ratio<=threshold]
    guo_df = guo_df[guo_df.remove_ratio<=threshold]
    gol_df = gol_df[gol_df.remove_ratio<=threshold]

    with open(results_dir/"true_results.json","r") as fp:
        true_results = json.load(fp) 

    # get the true test accuracy of the initial model
    true_test_accuracy = float(true_results[f"{dataset}{ovr_str.title()}"]["test_accuracy"])
    # set index based on the variables that have been varied 
    baseline_df = baseline_df.set_index(["remove_ratio","sampling_type","sampler_seed"])
    guo_df = guo_df.set_index(["remove_ratio","sampling_type","sampler_seed"])
    gol_df = gol_df.set_index(["remove_ratio","sampling_type","sampler_seed"])
    if not v2:
        # calculate disparity wrt to baseline fully-trained model using remove_accuracy 
        guo_df["baseline_eps"] = SMAPE(guo_df.remove_accuracy.values,baseline_df.loc[guo_df.index].remove_accuracy.values)
        gol_df["baseline_eps"] = SMAPE(gol_df.remove_accuracy.values,baseline_df.loc[gol_df.index].remove_accuracy.values)
    else:
        guo_df["baseline_eps"] = np.abs(guo_df.remove_accuracy.values-baseline_df.loc[guo_df.index].remove_accuracy.values)
        gol_df["baseline_eps"] = np.abs(gol_df.remove_accuracy.values-baseline_df.loc[gol_df.index].remove_accuracy.values)
    if not v2:
        # calculate disparity wrt initial model trained on all the data using test_accuracy
        guo_df["origin_eps"] = SMAPE(true_test_accuracy,guo_df.test_accuracy.values)
        gol_df["origin_eps"] = SMAPE(true_test_accuracy,gol_df.test_accuracy.values)
    else:
        guo_df["origin_eps"] = np.abs(true_test_accuracy-guo_df.test_accuracy.values)
        gol_df["origin_eps"] = np.abs(true_test_accuracy-gol_df.test_accuracy.values)
    return baseline_df,guo_df,gol_df

def plot_when_to_retrain(results_dir:Path,dataset:str,ovr_str:str,v2=False):
    baseline,guo,gol = load_dfs(results_dir,dataset,ovr_str,v2=v2)
    _,corr = scatter_metric_eps(guo,sampling_type="targeted_informed")
    # _,corr = scatter_metric_eps(gol,sampling_type="targeted_random")
    print(corr)
    # scatter_metric_eps(gol,sampling_type="all")

#%%
if __name__ == "__main__":
    dataset = "MNIST"; ovr_str="multi"
    baseline,guo,gol = load_dfs(results_dir,dataset,ovr_str)
    plot_when_to_retrain(results_dir,dataset,ovr_str,v2=False)
    # %%

# %%
a = np.array([0.,0.10172138,0.08085259,0.1135534,0.13438935,0.12971974
,0.09419162,0.06655178,0.56426943,4.19849616] )
b= np.array([0.01110048,0.04073321,0.02591202,0.08890981,0.21146299,0.28218696
,0.39037759,0.73906906,1.52551333,4.19849616])
# %%
