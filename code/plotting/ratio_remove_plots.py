#%%
from IPython import get_ipython

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
import matplotlib.ticker as mticker
from methods.common_utils import SAPE

#%%

save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
# %%
def load_dfs(results_dir:Path,dataset,ovr_str:str,suffix="",plot_deltagrad=False):
    
    file_name = f"Remove_Dist_{ovr_str}{suffix}.csv"
    dist = pd.read_csv(results_dir/dataset/file_name)
    dist_guo = dist[dist.method=="Guo"].apply(pd.to_numeric,errors="ignore")
    dist_gol = dist[dist.method=="Golatkar"].apply(pd.to_numeric,errors="ignore")
    dist_baseline = dist[dist.method=="baseline"]    
    if plot_deltagrad:
        deltagrad_file_prefix = f"Deltagrad_remove_ratio_{ovr_str+suffix}"
        dist_deltagrad,deltagrad_baseline = get_dg_remove_ratio_frames(results_dir,dataset,deltagrad_file_prefix)
        dist_deltagrad = dist_deltagrad[dist_deltagrad.period.isin([2,5,50,100])]
        # dist_deltagrad = dist_deltagrad[dist_deltagrad.period<200]
        deltagrad_baseline.drop("removal_time",axis="columns",inplace=True)
        deltagrad_baseline["sampling_type"]="targeted_informed"
        deltagrad_baseline = pd.concat([dist_baseline,deltagrad_baseline])
    else:
        dist_deltagrad = None
        deltagrad_baseline = None
    
    return dist_baseline,dist_guo,dist_gol,dist_deltagrad,deltagrad_baseline


def plot_tradeoffs(y,ratio,sampling_type,dist_baseline,dist_guo,dist_gol,dist_deltagrad=None,deltagrad_baseline=None,ax=None,legend=False,verbose:bool=False,**kwargs):
    
    transform = lambda df,ratio,sampling: df[(df.remove_ratio==ratio)&(df.sampling_type==sampling)]
    baseline = transform(dist_baseline,ratio,sampling_type)
    guo = transform(dist_guo,ratio,sampling_type)
    gol = transform(dist_gol,ratio,sampling_type)
    if dist_deltagrad is not None:
        deltagrad = transform(dist_deltagrad,ratio,sampling_type)
        baseline_deltagrad  = transform(deltagrad_baseline,ratio,sampling_type)
    markers = ["o","s","v","D","X","P","*",">"]
    y_label=" ".join(y.split("_")).title()
    max_speedup = 1
    min_speedup = np.inf
    if ax is None:
        fig,ax = plt.subplots()
    xmeans = []
    ymeans = []
    for (minibatch_fraction,df),marker in zip(guo.groupby("minibatch_fraction"),markers):
        x_values = baseline.removal_time.dropna().values/df.removal_time.values
        a = np.array(baseline[y].values.mean()) # add eps to avoid dividing by 0
        b = df[y].values 
        sape = SAPE(a,b)
        if verbose:
            print(f"Guo: {minibatch_fraction} X SAPE {x_values}")
        y_values = sape
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:blue",
            **kwargs
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
        max_speedup = max(x_values.max(),max_speedup)
        min_speedup = min(x_values.min(),min_speedup)  
    ax.plot(xmeans,ymeans,color="tab:blue",label="${INF}$",ls="--")
    xmeans = []
    ymeans = []
    for (minibatch_fraction,df),marker in zip(gol.groupby("minibatch_fraction"),markers):
        x_values = baseline.removal_time.dropna().values/(df.removal_time.values)
        a = np.array(baseline[y].values.mean())
        b = df[y].values
        sape = SAPE(a,b)
        y_values = sape
        if verbose:
            print(f"Gol: {minibatch_fraction} X SAPE {x_values}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:orange",
            **kwargs
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
        max_speedup = max(x_values.max(),max_speedup)
        min_speedup = min(x_values.min(),min_speedup)
    ax.plot(xmeans,ymeans,color="tab:orange",label="${FISH}$",ls="--")
    if dist_deltagrad is not None:
        xmeans = []
        ymeans = []
        for (period,df),marker in zip(deltagrad.groupby("period"),markers):
            num = min(len(baseline_deltagrad.removal_time),len(df.removal_time))
            x_values = baseline_deltagrad.removal_time.dropna()[:num].values/(df.removal_time.values[:num])
            a = np.array(baseline_deltagrad[y].values.mean())
            b = df[y].values
            sape = SAPE(a,b)
            y_values = sape
            if verbose:
                print(f"Deltagrad: {period} X SAPE {x_values}")
            ax.errorbar(
                x=x_values.mean(),
                y=y_values.mean(),
                xerr=x_values.std(),
                yerr=y_values.std(),
                marker=marker,
                label=f"$T_0={period}$", color="tab:green",
                **kwargs
            )
            xmeans.append(x_values.mean())
            ymeans.append(y_values.mean())
            max_speedup = max(x_values.max(),max_speedup)
            min_speedup = min(x_values.min(),min_speedup)
        ax.plot(xmeans,ymeans,color="tab:green",label="${DG}$",ls="--")

    ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    if legend:
        ax.legend(bbox_to_anchor=(1.05,0.5),loc="center left")
    ax.set_xscale("log",base=3)
    # ax.set_yscale("symlog",linthresh=1)
    # ax.set_ylim(bottom=-)
    # round to next 100
    # max_speedup = max_speedup-(max_speedup%100)+100*bool(max_speedup%100)
    # ticks = [0.25,1,2,5,10,20,50,100,200,300,800]
    # ax.xaxis.set_major_locator(mticker.LogLocator(3))
    # ticks_loc = ax.get_xticks().tolist()+[1]
    # print(ticks_loc)
    # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    # ax.set_xticklabels([f"{int(t)}x" if t>=1 else f"{t:.2f}x" for t in ticks_loc ] )
    # ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    # ticks = [0.25]+list(np.logspace(np.log10(1),np.log10(max_speedup),5,dtype=int))
    # print(max_speedup)
    # print(ticks)
    # max_index = (np.array(ticks)>=max_speedup).nonzero()[0][0]
    # ticks = ticks[:max_index]
    # ax.set_xticks(ticks)
    # ax.set_yticklabels(fontsize=15)
    # ax.set_xlabel("Speedup (Removal Time Ratio)")
    # ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("Speedup")
    ax.set_ylabel("sAPE $Acc_{del}$")
    return ax

def make_plots(results_dir:Path,dataset,ovr:bool,suffix="",plot_deltagrad=False,save_fig:bool=False):
    if not ovr:
        ovr_str = "binary"
        metric= "accuracy"
    else:
        ovr_str = "multi"
        metric="f1_score"

    figure_dir = results_dir/"images"/dataset
    
    if not figure_dir.exists():
        figure_dir.mkdir(exist_ok=True,parents=True)

    baseline,guo,gol,deltagrad,baseline_deltagrad = load_dfs(results_dir,dataset,ovr_str,suffix,plot_deltagrad)
    remove_ratios = sorted(baseline.remove_ratio.unique())
    sampling_types = baseline.sampling_type.unique()
    fig,ax = plt.subplots(len(remove_ratios),len(sampling_types),figsize=(4*len(sampling_types),4*len(remove_ratios)))
    ax = np.array(ax).reshape(len(remove_ratios),len(sampling_types))
    for i,ratio in enumerate(remove_ratios):
        for j,sampling_type in enumerate(sampling_types):
            axis = ax[i][j]
            axis = plot_tradeoffs(f"remove_{metric}",ratio,sampling_type,baseline,guo,gol,deltagrad,baseline_deltagrad,ax=axis,legend=True) 
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title(f"{sampling_type.title()} Removals {ratio*100:.0f}%")
            if i==0 and j ==0:
                axis.legend(bbox_to_anchor=(0.5,-0.01),loc="upper center",ncol=6,bbox_transform=fig.transFigure)
            else:
                axis.get_legend().remove()
    fig.subplots_adjust(bottom=0.01,top=0.95,wspace=0.25)
    plt.suptitle(f"{ovr_str.title()} {dataset}")
    # plt.tight_layout()
    plt.show()
#%%
if __name__ == "__main__":
    mpl.rcParams["figure.dpi"]=100
    #%%
    make_plots(results_dir,"MNIST",ovr=False,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
 # %%
    make_plots(results_dir,"COVTYPE",ovr=False,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
#%%
    make_plots(results_dir,"HIGGS",ovr=False,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
# %%
    make_plots(results_dir,"CIFAR",ovr=False,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
    # %%
    make_plots(results_dir,"EPSILON",ovr=False,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
# %%
    make_plots(results_dir,"MNIST",ovr=True,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
    # %%
    dfs = load_dfs(results_dir,"EPSILON","binary",suffix="_selected",plot_deltagrad=True)
    ratio = sorted(dfs[1].remove_ratio.unique())[2]
    plot_tradeoffs("remove_accuracy",ratio,"targeted_informed",*dfs,legend=True,verbose=True)
# %%
