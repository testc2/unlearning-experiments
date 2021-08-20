#%%
from IPython import get_ipython
from numpy.lib.npyio import save
from pandas.core import base
from traitlets.traitlets import BaseDescriptor

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
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from parse import get_deltagrad_dist_dataframes
#%%
mpl.rcParams["figure.dpi"]=100

save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
# %%
def load_dfs(results_dir:Path,dataset,ovr_str:str,dist_type,suffix="",plot_deltagrad=False):
    
    file_name = f"Distribution_{ovr_str}_{dist_type}{suffix}.csv"
    dist = pd.read_csv(results_dir/dataset/file_name)
    dist_guo = dist[dist.method=="Guo"].apply(pd.to_numeric,errors="ignore")
    dist_gol = dist[dist.method=="Golatkar"].apply(pd.to_numeric,errors="ignore")
    dist_baseline = dist[dist.method=="baseline"]
    if "minibatch_fraction" not in dist.columns:
        if dataset == "MNIST":
            num_removals = 4800
            dist_guo["minibatch_fraction"] = num_removals//dist_guo.minibatch_size
            dist_gol["minibatch_fraction"] = num_removals//dist_gol.minibatch_size
            dist_guo["num_removes"] = num_removals
            dist_gol["num_removes"] = num_removals
    
    if plot_deltagrad:
        deltagrad_file = f"Deltagrad_dist_{dist_type}_{ovr_str}.xml"
        dist_deltagrad,_ = get_deltagrad_dist_dataframes(results_dir/dataset/deltagrad_file)
        dist_deltagrad = dist_deltagrad[dist_deltagrad.period.isin([2,5,10,15])]
    else:
        dist_deltagrad = None
    
    return dist_baseline,dist_guo,dist_gol,dist_deltagrad

def plot_dist(metric:str,dist_baseline,dist_guo,dist_gol,dist_deltagrad=None,ovr:bool=False,dist_type="targeted"):
    
    if dist_type == "targeted":
        dist_string = "Uniform $\longrightarrow$ Targeted"
    else:
        dist_string = "Random $\longrightarrow$ Informed"
    
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ratio = int(dist_guo.remove_ratio.iloc[0] * 100)
    minibatch_fraction = 1
    selected_period = 15
    x="sample_prob"
    y = f"remove_{metric}"
    ylabel=" ".join(y.split("_")).title()
    transform = lambda df,column,val: df[df[column]==val]
    guo = transform(dist_guo,"minibatch_fraction",minibatch_fraction)
    gol = transform(dist_gol,"minibatch_fraction",minibatch_fraction)
    if dist_deltagrad is not None:
        deltagrad = transform(dist_deltagrad,"period",selected_period)
    sns.lineplot(data=guo,x=x,y=y,color="tab:blue",label="$R^{INF}$")
    sns.lineplot(data=gol,x=x,y=y,color="tab:orange",label="$R^{FISH}$")
    if dist_deltagrad is not None:
        sns.lineplot(data=deltagrad,x=x,y=y,color="tab:green",label="$R^{DG}$")
    sns.lineplot(data=dist_baseline,x=x,y=y,color="tab:red",label="Baseline")
    plt.xlabel(f"Sample Probability\n{dist_string}")
    plt.ylabel(ylabel)


qerror = lambda x,y: max(x/y,y/x)
def plot_tradeoffs(y,dist_baseline,dist_guo,dist_gol,dist_deltagrad=None,ax=None,legend=False):
    
    sample_prob = dist_guo.sample_prob.unique()[-1]
    transform_tradeoff = lambda x: x[x.sample_prob==sample_prob]
    baseline = transform_tradeoff(dist_baseline)
    guo = transform_tradeoff(dist_guo)
    gol = transform_tradeoff(dist_gol)
    if dist_deltagrad is not None:
        deltagrad = transform_tradeoff(dist_deltagrad)
    markers = ["o","s","v","D"]
    y_label=" ".join(y.split("_")).title()
    max_speedup = 1
    if ax is None:
        fig,ax = plt.subplots()
    xmeans = []
    ymeans = []
    for (minibatch_fraction,df),marker in zip(guo.groupby("minibatch_fraction"),markers):
        x_values = baseline.removal_time.divide(df.removal_time.values)
        a = baseline[y].values
        b = df[y].values
        qerror = np.maximum(a/b,b/a)
        y_values = qerror
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:blue"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
        max_speedup = max(x_values.max(),max_speedup)
    ax.plot(xmeans,ymeans,color="tab:blue",label="$R^{INF}$",ls="--")
    xmeans = []
    ymeans = []
    for (minibatch_fraction,df),marker in zip(gol.groupby("minibatch_fraction"),markers):
        x_values = baseline.removal_time.divide(df.removal_time.values)
        a = baseline[y].values
        b = df[y].values
        qerror = np.maximum(a/b,b/a)
        y_values = qerror
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:orange"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
        max_speedup = max(x_values.max(),max_speedup)
    ax.plot(xmeans,ymeans,color="tab:orange",label="$R^{FISH}$",ls="--")
    if dist_deltagrad is not None:
        xmeans = []
        ymeans = []
        for (period,df),marker in zip(deltagrad.groupby("period"),markers):
            num = min(len(baseline.removal_time),len(df.removal_time))
            x_values = baseline.removal_time[:num].divide(df.removal_time.values[:num])
            a = baseline[y][:num].values
            b = df[y][:num].values
            qerror = np.maximum(a/b,b/a)
            y_values = qerror
            ax.errorbar(
                x=x_values.mean(),
                y=y_values.mean(),
                xerr=x_values.std(),
                yerr=y_values.std(),
                marker=marker,
                label=f"$T_0={period}$", color="tab:green"
            )
            xmeans.append(x_values.mean())
            ymeans.append(y_values.mean())
            max_speedup = max(x_values.max(),max_speedup)
        ax.plot(xmeans,ymeans,color="tab:green",label="$R^{DG}$",ls="--")

    ax.axhline(y=1, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    if legend:
        ax.legend(bbox_to_anchor=(1.05,0.5),loc="center left")
    ax.set_xscale("log")
    # round to next 100
    max_speedup = max_speedup-(max_speedup%100)+100*bool(max_speedup%100)
    ticks = [0.25,1,2,5,10,20,50,100,300,800]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}x" for t in ticks])
    ax.set_xlabel("Speedup (Removal Time Ratio)")
    ax.set_ylabel(f"{y_label} Ratio")
    return ax

def make_plots(results_dir:Path,dataset,ovr:bool,dist_type,suffix="",plot_deltagrad=False,save_fig=False):
    if not ovr:
        ovr_str = "binary"
        metric= "accuracy"
    else:
        ovr_str = "multi"
        metric="f1_score"

    figure_dir = results_dir/"images"/dataset
    
    if not figure_dir.exists():
        figure_dir.mkdir(exist_ok=True,parents=True)
    
    baseline,guo,gol,deltagrad = load_dfs(results_dir,dataset,ovr_str,dist_type,suffix,plot_deltagrad)
    ratio = int(guo.remove_ratio.iloc[0] * 100)
    plot_dist(metric,baseline,guo,gol,deltagrad,ovr,dist_type)
    plt.title(f"{ratio}% {dist_type.title()} Removals {ovr_str.title()} {dataset}")
    if save_fig:
        plt.savefig(figure_dir/f"Dist_{ovr_str}_{dist_type}.pdf",dpi=200,bbox_inches="tight")
    else:
        plt.show()
    # fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    fig,ax2 = plt.subplots()
    # ax1 = plot_tradeoffs(f"test_{metric}",dist_baseline,dist_guo,dist_gol,dist_deltagrad,ax=ax1)    
    # plt.suptitle(f"{dist_type.title()} Removals {ratio}% {ovr_str.title()} {dataset}")
    ax2 = plot_tradeoffs(f"remove_{metric}",baseline,guo,gol,deltagrad,ax=ax2,legend=True)    
    plt.title(f"{dist_type.title()} Removals {ratio}% {ovr_str.title()} {dataset}")
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_dir/f"Dist_{dist_type}_{ovr_str}_tradeoff.pdf",dpi=200,bbox_inches="tight")
    else:
        plt.show()
#%%
make_plots(results_dir,"MNIST",ovr=False,dist_type="targeted",suffix="_batch",plot_deltagrad=True,save_fig=save_fig)
# %%
make_plots(results_dir,"MNIST",ovr=False,dist_type="informed",suffix="_batch",plot_deltagrad=True,save_fig=save_fig)
#%%
make_plots(results_dir,"MNIST",ovr=True,dist_type="targeted",suffix="_batch",plot_deltagrad=True,save_fig=save_fig)
# %%
make_plots(results_dir,"MNIST",ovr=True,dist_type="informed",suffix="_batch",plot_deltagrad=True,save_fig=save_fig)
# %%
make_plots(results_dir,"COVTYPE",ovr=False,dist_type="targeted",plot_deltagrad=True,save_fig=save_fig)
# %%
make_plots(results_dir,"COVTYPE",ovr=False,dist_type="informed",plot_deltagrad=True,save_fig=save_fig)
# %%
