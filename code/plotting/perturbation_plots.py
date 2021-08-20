#%%
from IPython import get_ipython

if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False

from pathlib import Path
import numpy as np
import sys
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str((project_dir/"code").resolve()))
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from parse import get_deltagrad_perturb_dataframes
#%%

save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
#%%

def load_dfs(results_dir:Path,dataset:str,ovr_str:str,plot_deltagrad:bool,suffix:str=""):
    perturb_df = pd.read_csv(results_dir/dataset/f"Perturbation_{ovr_str}{suffix}.csv")
    perturb_df_guo = perturb_df[perturb_df.method=="Guo"]
    perturb_df_gol = perturb_df[perturb_df.method=="Golatkar"]
    perturb_df_baseline = perturb_df[perturb_df.method=="baseline"]
    if plot_deltagrad:
        perturb_df_deltagrad = get_deltagrad_perturb_dataframes(results_dir/dataset/f"Deltagrad_perturb_{ovr_str}.xml")
    else:
        perturb_df_deltagrad = None
    return perturb_df_baseline,perturb_df_guo,perturb_df_gol,perturb_df_deltagrad

def plot_metric(y:str,perturb_df_baseline,perturb_df_guo,perturb_df_gol,perturb_df_deltagrad=None,ax=None):
    x = "noise"; x_label = "$\sigma$"
    y_label=" ".join(y.split("_")).title()
    if ax is None:
        plt,ax = plt.subplots()
    sns.lineplot(data=perturb_df_guo,x=x,y=y,label="$P^{NL}$",color="tab:blue",ax=ax)
    sns.lineplot(data=perturb_df_gol,x=x,y=y,label="$P^{FISH}$",color="tab:orange",ax=ax)
    if perturb_df_deltagrad is not None:
        sns.lineplot(data=perturb_df_deltagrad,x=x,y=y,label="$P^{GAU}$",color="tab:green",ax=ax)
    sns.lineplot(data=perturb_df_baseline,x=x,y=y,label="Baseline",color="tab:red",ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")

    return ax

def smape(forecast,actual):
    smape = np.abs(forecast-actual)/(np.abs(forecast)+np.abs(actual))
    return smape*100

def plot_metric_smape(y:str,perturb_df_baseline,perturb_df_guo,perturb_df_gol,perturb_df_deltagrad=None,ax=None,**kwargs):
    x = "noise"; x_label = "$\sigma$"
    if ax is None:
        plt,ax = plt.subplots()
    baseline_y = perturb_df_baseline[y].iloc[0]
    perturb_df_guo[f"SMAPE_{y}"] = smape(perturb_df_guo[y].values,baseline_y)
    perturb_df_gol[f"SMAPE_{y}"] = smape(perturb_df_gol[y].values,baseline_y)
    sns.lineplot(data=perturb_df_guo,x=x,y=f"SMAPE_{y}",label="${INF}$",color="tab:blue",ax=ax,marker="o",**kwargs)
    sns.lineplot(data=perturb_df_gol,x=x,y=f"SMAPE_{y}",label="${FISH}$",color="tab:orange",ax=ax,marker="o",**kwargs)
    if perturb_df_deltagrad is not None:
        perturb_df_deltagrad[f"SMAPE_{y}"] = smape(perturb_df_deltagrad[y].values,baseline_y)
        sns.lineplot(data=perturb_df_deltagrad,x=x,y=f"SMAPE_{y}",label="${DG}$",color="tab:green",ax=ax,marker="o",**kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test Accuracy SMAPE %")
    ax.set_xscale("log")

    return ax

def plot_model_diff(perturb_df_guo,perturb_df_gol,perturb_df_deltagrad=None,ax=None,**kwargs):
    if ax is None:
        plt,ax = plt.subplots()
    x = "noise"; x_label = "$\sigma$"
    y = "model_diff";y_label="$L_2$ Model Difference"
    sns.lineplot(data=perturb_df_guo,x=x,y=y,ax=ax,label="$INFL$",color="tab:blue",marker="o",**kwargs)
    sns.lineplot(data=perturb_df_gol,x=x,y=y,ax=ax,label="$FISH$",color="tab:orange",marker="o",**kwargs)
    if perturb_df_deltagrad is not None:
        sns.lineplot(data=perturb_df_deltagrad,x=x,y=y,ax=ax,label="$DG$",color="tab:green",marker="o",**kwargs)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax

def plot_perturbation(results_dir:Path,dataset:str,ovr:bool,plot_deltagrad:bool,save_fig=False,suffix:str=""):
    if ovr:
        metric = "accuracy"
        ovr_str = "multi"
    else:
        metric = "accuracy"
        ovr_str = "binary"
    figure_dir = results_dir/"images"/dataset

    if not figure_dir.exists():
        figure_dir.mkdir(exist_ok=True,parents=True)
    
    baseline,guo,gol,deltagrad = load_dfs(results_dir,dataset,ovr_str,plot_deltagrad,suffix)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1 = plot_metric_smape(f"test_{metric}",baseline,guo,gol,deltagrad,ax=ax1)
    ax1.annotate("(a)", xy=(0.45, -0.2), xycoords="axes fraction") 
    ax2 = plot_model_diff(guo,gol,deltagrad,ax=ax2)
    ax2.annotate("(b)", xy=(0.45, -0.2), xycoords="axes fraction") 
    plt.suptitle(f"Effect of Perturbation Mechanism in {ovr_str.title()} {dataset}")
    # plt.tight_layout()
    if save_fig:
        plt.savefig(figure_dir/f"Perturbation_{dataset}_{ovr_str}.pdf",bbox_inches="tight",dpi=200)
    else:
        plt.show()

#%%
if __name__ == "__main__":
    #%%
    mpl.rcParams["figure.dpi"]=100
    #%%
    plot_perturbation(results_dir,"MNIST",ovr=False,plot_deltagrad=True,save_fig=save_fig)
    #%%
    plot_perturbation(results_dir,"MNIST",ovr=True,plot_deltagrad=True,save_fig=save_fig)
    #%%
    plot_perturbation(results_dir,"COVTYPE",ovr=False,plot_deltagrad=True,save_fig=save_fig)
    #%%
    plot_perturbation(results_dir,"CIFAR",ovr=False,plot_deltagrad=True,save_fig=save_fig)
    # %%
    plot_perturbation(results_dir,"HIGGS",ovr=False,plot_deltagrad=True,save_fig=save_fig)
    # %%
    plot_perturbation(results_dir,"EPSILON",ovr=False,plot_deltagrad=True,save_fig=save_fig)
    # %%
