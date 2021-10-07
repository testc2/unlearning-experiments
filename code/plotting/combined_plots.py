#%%
from os import replace
from re import escape
from IPython import get_ipython
from numpy.lib.npyio import save
from traitlets.traitlets import default

from plotting.pipeline_plots import compute_all_metrics, compute_error_metrics, load_dfs, plot_metric
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
import plotting.ratio_plots as ratio_plots
import plotting.ratio_remove_plots as dist_plots
import plotting.perturbation_plots as perturb_plots
import plotting.when_to_retrain_plots as retrain_plots
import plotting.unlearn_plots as unlearn_plots
from typing import List
import json
from matplotlib.gridspec import SubplotSpec

# mpl.rcParams["text.usetex"]=True
save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
# %%

def plot_combined(results_dir:Path,num_ratios:int,plot_deltagrad:bool=False,save_fig:bool=False,suffix:str=""):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"f1_score"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    fig,ax = plt.subplots(1+num_ratios,len(datasets),figsize=(4*len(datasets),2*(1+num_ratios)))
    ax = np.array(ax).reshape(1+num_ratios,len(datasets))
    
    relative_test_accuracy_drops=[1,5,10]
    for j in range(len(datasets)):
        ratio_dfs = ratio_plots.load_dfs(results_dir,datasets[j],ovr_strs[j])
        dist_dfs = dist_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,suffix=suffix)
        axis = ax[:,j]

        axis[0] = ratio_plots.plot_metric("remove_accuracy",ratio_dfs,ax=axis[0])
        if j == len(datasets)-1:
            axis[0].legend(bbox_to_anchor=(1.15,0.65),title="Sampling Type",loc="upper left")
        else:
            axis[0].get_legend().remove()
        if j !=0 :
            axis[0].set_ylabel("")
        axis[0].set_title(dataset_names[j])
        ratios = sorted(dist_dfs[1].remove_ratio.unique())
        for ii,ratio in enumerate(ratios):
            i = ii+1
            axis[i] = dist_plots.plot_tradeoffs(f"remove_{metric_map[ovr_strs[j]]}",ratio,"targeted_informed",*dist_dfs,ax=axis[i],legend=True) 
            if j !=0 :
                axis[i].set_ylabel("")
            if ii == len(ratios)-1 and j ==0:
                axis[i].legend(bbox_to_anchor=(0.5,-0.04),loc="upper center",ncol=12,bbox_transform=fig.transFigure)
            else:
                axis[i].get_legend().remove()
            if j == len(datasets)-1:
                axis[i].annotate(f'Targeted Informed\n$Test Accuracy Drop={relative_test_accuracy_drops[ii]:.0f}\%$', xy=(1.4,0.5), rotation=0,ha='center',va='center',xycoords='axes fraction')

    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.01,top=0.95,wspace=0.4,hspace=0.3)
    if save_fig:
        # plt.savefig(figure_dir/"Grid_Plot.pdf",bbox_inches="tight",dpi=300)
        plt.savefig(Path.home()/"Downloads"/"grid.pdf",bbox_inches="tight",dpi=300)
    else:
        plt.show()

def plot_extended_ratios(results_dir:Path,save_fig:bool=False):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"f1_score"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    fig,ax = plt.subplots(2,len(datasets),figsize=(4*len(datasets),2*2))
    ax = np.array(ax).reshape(2,len(datasets))
    for j in range(len(datasets)):
        ratio_dfs = ratio_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],suffix="_extended")
        axis = ax[:,j]
        axis[0] = ratio_plots.plot_metric("test_accuracy",ratio_dfs,ax=axis[0])
        axis[1] = ratio_plots.plot_metric("remove_accuracy",ratio_dfs,ax=axis[1])
        axis[1].get_legend().remove()
        if j == len(datasets)-1:
            axis[0].legend(bbox_to_anchor=(0.9,0.65),title="Sampling Type",loc="upper left",bbox_transform=fig.transFigure)
        else:
            axis[0].get_legend().remove()
        if j !=0 :
            axis[0].set_ylabel("")
            axis[1].set_ylabel("")
        axis[0].set_title(dataset_names[j])
    fig.subplots_adjust(bottom=0.01,top=0.95,wspace=0.4,hspace=0.5)

def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = (numerator==0)&(denominator==0)
    sae = numerator/denominator
    sae[both_zero] = 1 
    return sae*100

def find_selected_ratios(save_fig:bool=False):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"f1_score"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    with open(results_dir/"true_results.json","r") as fp:
        true_results = json.load(fp) 
        
    ratio_dfs_list = []
    for j in range(len(datasets)):
            ratio_dfs = ratio_plots.load_dfs(results_dir,datasets[j],ovr_strs[j])

            true_test_accuracy = float(true_results[f"{datasets[j]}{ovr_strs[j].title()}"]["test_accuracy"])
            # ratio_dfs["accuracy_drop_percentage"] = ((true_test_accuracy-ratio_dfs.test_accuracy)/true_test_accuracy)*100
            ratio_dfs["accuracy_drop_percentage"] = SAPE(true_test_accuracy,ratio_dfs.test_accuracy.values)
            ratio_dfs_list.append(ratio_dfs)
    
    fig,ax = plt.subplots(1,len(datasets),figsize=(4*len(datasets),4),sharex=True,sharey=True)
    ax = np.array(ax).reshape(1,len(datasets))
    for j in range(len(datasets)):
        axis = ax[:,j]
        df = ratio_dfs_list[j]
        if ovr_strs[j]=="binary":
            chosen_class = 1
        else:
            chosen_class = 3
        df = df[(df.sampling_type=="targeted_informed")&(df.remove_class==chosen_class)]
        ratio_plots.plot_class_ratio("targeted_informed","accuracy_drop_percentage",df,ax=axis[0])
        # axis[0].legend().get_texts()[0].set_text("_removed_class")
        axis[0].set_ylabel("Test Accuracy Drop %")
        axis[0].set_title(dataset_names[j])
        l1=axis[0].axhline(1,linestyle="--",label="1%",color="tab:blue")
        l2=axis[0].axhline(5,linestyle="--",label="5%",color="tab:green")
        l3=axis[0].axhline(10,linestyle="--",label="10%",color="tab:orange")
        if j== len(datasets)-1:
            axis[0].legend(handles=[l1,l2,l3],bbox_to_anchor=(0.9,0.5),loc="center left",bbox_transform=fig.transFigure)
        else:
            axis[0].get_legend().remove()
        axis[0].set_ylim(top=20)
        axis[0].set_xlim(right=0.4)
    fig.subplots_adjust(top=0.7)
    plt.suptitle("Selecting Remove Ratios based on drop in test accuracy",fontsize=30)
    if save_fig:
        plt.savefig(figure_dir/"Selecting Ratios Grid.pdf",bbox_inches="tight",dpi=300)

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
#%%
if __name__ == "__main__":
    mpl.rcParams["figure.dpi"]=100
    mpl.rcParams["font.size"]=5
    #%%
    plot_combined(results_dir,[0.01,0.05,0.10],save_fig=save_fig)
    # %%
    plot_combined(results_dir,[],save_fig=save_fig)
    # %%
    plot_extended_ratios(results_dir)
    # %%
    find_selected_ratios(save_fig)
    # %%
    plot_combined(results_dir,3,save_fig=save_fig,plot_deltagrad=True,suffix="_selected")
    # %%

def plot_ratios_grid(results_dir:Path,save_fig:bool=False,latex:bool=False,extended:str=False,fig_width_pt:float=246.0,scale:float=1):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabel = r"$\texttt{Acc}_\text{test}$"
        del_ylabel = r"$\texttt{Acc}_\text{del}$"
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"
    
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    subplots = (2,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0]+1,subplots[1]+2))
    figsize = np.array(figsize)*scale
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)

    if extended :
        fig.subplots_adjust(bottom=0.01,top=0.85,wspace=0.3,hspace=0.3)
    else:
        fig.subplots_adjust(bottom=0.15,top=0.85,wspace=0.3,hspace=0.3)

    for j in range(len(datasets)):
        ratio_dfs = ratio_plots.load_dfs(results_dir,datasets[j],ovr_strs[j])
        extended_ratio_dfs = ratio_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],suffix="_extended")
        axis = ax[:,j]
        
        if extended:
            axis[0] = ratio_plots.plot_metric("test_accuracy",extended_ratio_dfs,ax=axis[0],**dict(palette=["tab:blue","tab:green"],markevery=3,markers=True,markersize=12))
            axis[1] = ratio_plots.plot_metric("remove_accuracy",extended_ratio_dfs,ax=axis[1],**dict(palette=["tab:blue","tab:green"],markevery=3,markers=True,markersize=12))
        else:
            axis[0] = ratio_plots.plot_metric("test_accuracy",ratio_dfs,ax=axis[0],**dict(palette=["tab:blue","tab:orange","tab:green","tab:red"],markersize=12,markers=True))
            axis[1] = ratio_plots.plot_metric("remove_accuracy",ratio_dfs,ax=axis[1],**dict(palette=["tab:blue","tab:orange","tab:green","tab:red"],markersize=12,markers=True))
        
        if j == 0:
            handles, labels = axis[0].get_legend_handles_labels()
            labels = ["-".join(s.split("_")) for s in labels]
            if latex:
                labels = [fr"$\texttt{{{s}}}$" for s in labels]
            leg_loc = (0.5,1.05)
            xlabel_loc = (0.5,0.05)
            axis[0].legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=leg_loc,
                # title="Deletion Distribution",
                loc="upper center",
                bbox_transform=fig.transFigure,
                ncol=4,
                # fontsize=30,
                # markerscale=2,
                )
            axis[0].annotate(
                "Deletion Fraction",
                # fontsize=30,
                xy=xlabel_loc,
                ha='center',
                va='center',
                xycoords='figure fraction'
            )
        else:
            axis[0].get_legend().remove()
        axis[1].get_legend().remove()
        axis[0].set_xlabel("")
        axis[1].set_xlabel("")
        
        if j !=0 :
            axis[0].set_ylabel("")
            axis[1].set_ylabel("")
        else:
            axis[0].set_ylabel(test_ylabel)#,fontsize=30)
            axis[1].set_ylabel(del_ylabel)#,fontsize=30)
        axis[0].set_xlabel("")
        axis[1].set_xlabel("")
        if dataset_base_names[j] == "mnist_multi" and not extended:
            axis[0].set_xticks([0,0.1,0.5])
            axis[1].set_xticks([0,0.1,0.5])
        axis[0].set_title(dataset_names[j])#fontsize=30)
        axis[0].tick_params(axis="both",which="major")#,labelsize=30)
        axis[0].tick_params(axis="both",which="major")#,labelsize=30)
        axis[1].tick_params(axis="both",which="major")#,labelsize=30)
        axis[1].tick_params(axis="both",which="major")#,labelsize=30)
        
    fig.align_ylabels(ax[:,0])
    # plt.suptitle("Effect of Sampling Distributions",fontsize=30)
    if save_fig:
        if not extended:
            plt.savefig(figure_dir/"Effect_of_Sampling_Grid.pdf",bbox_inches="tight")
        else:
            plt.savefig(figure_dir/"Effect_of_Sampling_Grid_extended.pdf",bbox_inches="tight")

# %%
if __name__ == "__main__":
    plot_ratios_grid(results_dir,save_fig=save_fig)

# %%
@mpl.ticker.FuncFormatter
def speed_up(x, pos):
    return f"{int(x)}x" if x>=1 else f"{str(round(x,2)).rstrip('0')}x"

def plot_remove_dist(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",latex:str=False,fig_width_pt:float=246.0,scale:float=3):
    num_ratios=3
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabel = r"$\text{Acc}_\text{test}$"
        del_ylabel = r"Accuracy Disparity (\texttt{AccDis}) \%"
        sape = r"$\texttt{sAPE}$"
        label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"
    
    subplots = (num_ratios,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=1
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(left=0.04,bottom=0.15,top=0.99,wspace=0.3,hspace=0.3) 
    relative_test_accuracy_drops=[1,5,10]
    relative_test_accuracy_drops_names=["small","medium","large"]
    kwargs=dict(markersize=10,markeredgecolor="black")
    for j in range(len(datasets)):
        dist_dfs = dist_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,suffix=suffix)
        axis = ax[:,j]
        ratios = sorted(dist_dfs[1].remove_ratio.unique())
        for i,ratio in enumerate(ratios):
            axis[i] = dist_plots.plot_tradeoffs(f"remove_{metric_map[ovr_strs[j]]}",ratio,"targeted_informed",*dist_dfs,ax=axis[i],legend=True,**kwargs) 
            axis[i].set_ylabel("")
            axis[i].set_xlabel("")
            axis[i].xaxis.set_major_formatter(speed_up)
            if datasets[j] == "HIGGS":
                axis[i].set_yscale("symlog",linthresh=1)
                axis[i].set_ylim(bottom=-0.1,top=1e2+0.1)
                axis[i].set_xticks([1,9,27,81])
            if datasets[j] == "COVTYPE":
                axis[i].set_yscale("symlog",linthresh=1)
                axis[i].set_ylim(bottom=-0.1,top=1e2+0.1)
                axis[i].set_xticks([1,3,27,243])
            if datasets[j] == "CIFAR":
                if i==0:
                    axis[i].set_yscale("symlog",linthresh=1)
                    axis[i].set_ylim(bottom=-0.1,top=1e2+50)
                axis[i].set_xticks([1,3,9])
            if dataset_base_names[j]=="mnist_binary":
                axis[i].set_xticks([1,3,9,27])
            if dataset_base_names[j]=="mnist_multi":
                axis[i].set_xticks([1,3,9,27])
            if datasets[j] == "EPSILON":
                axis[i].set_xticks([0.3,1,3])
            if i==0 and j==0:
                axis[i].annotate(f'Speed-up',#fontsize=30,
                 xy=(0.5,0.05), rotation=0,ha='center',va='center',xycoords='figure fraction')
                axis[i].annotate(f'{del_ylabel}',#fontsize=30,
                 xy=(0.01,0.5), rotation=90,ha='center',va='center',xycoords='figure fraction')
            
            axis[i].get_legend().remove()
            if j == len(datasets)-1:
                axis[i].annotate(f'{relative_test_accuracy_drops_names[i]}', #fontsize=30,
                 xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')
            if i==0:
                axis[i].set_title(dataset_names[j])#,fontsize=30)
    
    fig.align_ylabels(ax[:,0])
    if save_fig:
        plt.savefig(figure_dir/"Certifiability_Efficiency_Grid.pdf",bbox_inches="tight")
    else:
        plt.show()

def plot_remove_dist_single(results_dir:Path,remove_ratio_size:str,plot_deltagrad=False,save_fig:bool=False,suffix:str="",latex:str=False,fig_width_pt:float=246.0,scale:float=3):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        del_ylabel = r"\texttt{AccDis} \%"
        sape = r"$\texttt{sAPE}$"
        label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${AccDis}$"
    

    relative_test_accuracy_drops_names=["small","medium","large"]
    relative_test_accuracy_drops=[1,5,10]
    assert remove_ratio_size in relative_test_accuracy_drops_names
    remove_ratio_idx = relative_test_accuracy_drops_names.index(remove_ratio_size)
    subplots = (1,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=0.5
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(
        left=0.05,
        bottom=0.3,
        top=0.99,
        wspace=0.3,
        hspace=0.3
        )
    kwargs=dict(markersize=10,markeredgecolor="black")
    for j in range(len(datasets)):
        dist_dfs = dist_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,suffix=suffix)
        axis = ax[:,j]
        ratios = sorted(dist_dfs[1].remove_ratio.unique())
        ratio = ratios[remove_ratio_idx]
        i = 0
        axis[i] = dist_plots.plot_tradeoffs(f"remove_{metric_map[ovr_strs[j]]}",ratio,"targeted_informed",*dist_dfs,ax=axis[i],legend=True,**kwargs) 
        axis[i].set_ylabel("")
        axis[i].set_xlabel("")
        axis[i].xaxis.set_major_formatter(speed_up)
        if datasets[j] == "HIGGS":
            axis[i].set_yscale("symlog",linthresh=1)
            axis[i].set_ylim(bottom=-0.1,top=1e2+0.1)
            axis[i].set_xticks([1,9,27,81])
        if datasets[j] == "COVTYPE":
            axis[i].set_yscale("symlog",linthresh=1)
            axis[i].set_ylim(bottom=-0.1,top=1e2+0.1)
            axis[i].set_xticks([1,3,27,243])
        if datasets[j] == "CIFAR":
            if i==0:
                axis[i].set_yscale("symlog",linthresh=1)
                axis[i].set_ylim(bottom=-0.1,top=1e2+50)
            axis[i].set_xticks([1,3,9])
        if dataset_base_names[j]=="mnist_binary":
            axis[i].set_xticks([1,3,9,27])
        if dataset_base_names[j]=="mnist_multi":
            axis[i].set_xticks([1,3,9,27])
        if datasets[j] == "EPSILON":
            axis[i].set_xticks([0.3,1,3])
        if i==0 and j==0:
            axis[i].annotate(f'Speed-up',#fontsize=30,
                xy=(0.5,0.05), rotation=0,ha='center',va='center',xycoords='figure fraction')
            axis[i].annotate(f'{del_ylabel}',#size=30,
                xy=(0.02,0.55), rotation=90,ha='center',va='center',xycoords='figure fraction')
        
        axis[i].get_legend().remove()
        if j == len(datasets)-1:
            axis[i].annotate(f'{relative_test_accuracy_drops_names[remove_ratio_idx]}', #fontsize=30,
                xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')
        if i==0:
            axis[i].set_title(dataset_names[j])#,fontsize=30)
    
    fig.align_ylabels(ax[:,0])
    if save_fig:
        plt.savefig(figure_dir/f"Certifiability_Efficiency_Trade_Off_Grid_{remove_ratio_size}_sigma_0.pdf",bbox_inches="tight")
    else:
        plt.show()


def get_legend_QoA(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",latex:str=False,fig_width_pt:float=246.0,scale:float=3):
    num_ratios=1
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}

    subplots = (num_ratios,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=1
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(left=0.04,bottom=0.15,top=0.99,wspace=0.3,hspace=0.3) 

    dist_dfs = dist_plots.load_dfs(results_dir,datasets[0],ovr_strs[0],plot_deltagrad=plot_deltagrad,suffix=suffix)
    ratio = sorted(dist_dfs[1].remove_ratio.unique())[0]
    kwargs=dict(markersize=10,markeredgecolor="black")
    ax[0][0] = dist_plots.plot_tradeoffs(f"remove_{metric_map[ovr_strs[0]]}",ratio,"targeted_informed",*dist_dfs,ax=ax[0][0],legend=True,**kwargs) 
    # get legend handles
    handles,labels = ax[0][0].get_legend_handles_labels() 
    # next each QoA 
    if plot_deltagrad:
        args_sort_index = [
                            0,1,2, #first the 3 removal steps
                            3,7,11, # first QoA
                            4,8,12, # second QoA
                            5,9,13, # third QoA
                            6,10,14 # fourth QoA
        ]
    else:
        args_sort_index = [
                            0,1, #first the 2 removal steps
                            2,6, # first QoA
                            3,7, # second QoA
                            4,8, # third QoA
                            5,9, # fourth QoA
        ]

    handles = list(np.array(handles)[args_sort_index])
    labels = list(np.array(labels)[args_sort_index])
    if latex:
        labels = [label_map[l] if l in label_map else l  for l in labels ]
    
    legend = ax[0][0].legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5,-0.5    ),
        loc="upper center",
        ncol=5,
        bbox_transform=fig.transFigure,
        # fontsize=30,
    )
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if save_fig:
        fig.savefig(figure_dir/"legend.pdf", bbox_inches=bbox)
    
# %%
if __name__ == "__main__":
    plot_remove_dist(results_dir,plot_deltagrad=True,save_fig=save_fig,suffix="_selected")
# %%

def plot_perturbation(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",latex:bool=False,fig_width_pt:float=246.0,scale:float=1):
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
            dataset_names = [
                r"$\textsc{mnist}^{\text{b}}$",
                r"$\textsc{mnist}$",
                r"$\textsc{covtype}$",
                r"$\textsc{higgs}$",
                r"$\textsc{cifar2}$",
                r"$\textsc{epsilon}$"
            ]
            test_ylabel = r"$\texttt{AccErr} \%$"
            del_ylabel = r"$\texttt{Acc}_\text{del}$"
            sape = r"$\texttt{sAPE}$"
            label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}

    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"

    subplots = (2,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0]+1,subplots[1]))
    figsize = np.array(figsize)*scale
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    kwargs = dict(markersize=10)
    for j in range(len(datasets)):
        perturb_dfs = perturb_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,suffix=suffix)
        axis = ax[:,j]
        
        axis[0] = perturb_plots.plot_metric_smape("test_accuracy",*perturb_dfs,ax=axis[0],**kwargs)
        axis[1] = perturb_plots.plot_model_diff(*perturb_dfs[1:],ax=axis[1],**kwargs)

        axis[0].set_xticks([0.01,1,100])
        axis[1].set_xticks([0.01,1,100])
        if j == len(datasets)-1:
            handles, labels = axis[0].get_legend_handles_labels()
            if latex:
                labels = [label_map[l] if l in label_map else l  for l in labels ]
            axis[0].legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(0.5,0.01),
                loc="upper center",
                bbox_transform=fig.transFigure,
                ncol=4,
                # fontsize=30
            )
        else:
            axis[0].get_legend().remove()
        axis[1].get_legend().remove()
        if j !=0 :
            axis[0].set_ylabel("")
            axis[1].set_ylabel("")
        else:
            axis[0].set_ylabel(f"{test_ylabel}")#,fontsize=30)
            axis[1].set_ylabel("$L_2$ Distance")#,fontsize=30)
        if j ==0:
            axis[0].annotate(f"Noise parameter $\sigma$",#fontsize=30,
            xy=(0.5,0.11), rotation=0,ha='center',va='center',xycoords='figure fraction'
            )
        axis[1].tick_params(axis="both",which="major")#,labelsize=30)
        axis[0].tick_params(axis="both",which="major")#,labelsize=30)
        axis[0].set_xlabel("")
        axis[1].set_xlabel("")
        axis[0].set_title(dataset_names[j])#,fontsize=30)

    fig.align_ylabels(ax[:,0])
    fig.subplots_adjust(bottom=0.1,top=0.9,wspace=0.25,hspace=0.2)
    if save_fig:
        plt.savefig(figure_dir/"Effect_of_Perturbation_Mechanism.pdf",bbox_inches="tight")

#%%
if __name__ == "__main__":
    plot_perturbation(results_dir,plot_deltagrad=True,save_fig=save_fig)

# %%
def plot_when_to_retrain_dist(results_dir:Path,method:str,save_fig:bool=False,suffix:str=""):
    num_sampling_types = 4
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    
    fig,ax = plt.subplots(num_sampling_types+1,len(datasets),figsize=(4*len(datasets),2*(num_sampling_types+1)),squeeze=False)
    
    for j in range(len(datasets)):
        baseline,guo,gol = retrain_plots.load_dfs(results_dir,datasets[j],ovr_strs[j])
        axis = ax[:,j]
        if method == "Guo":
            method_df = guo
            method_str = "$R^{INF}$"
        elif method == "Golatkar":
            method_df = gol
            method_str = "$R^{FISH}$"
        
        sampling_types = method_df.reset_index().sampling_type.unique()
        for i in range(num_sampling_types+1):
            if i ==0:
                axis[i],corr = retrain_plots.scatter_metric_eps(method_df,sampling_type="all",ax=axis[i])
                if j== len(datasets)-1:
                    axis[i].legend(bbox_to_anchor=(1,0.5),loc="center left")
                else:
                    axis[i].get_legend().remove()
            else:
                sampling_type = sampling_types[i-1]
                axis[i],corr = retrain_plots.scatter_metric_eps(method_df,sampling_type=sampling_type,ax=axis[i])
                if j == len(datasets)-1 :
                    # print(f"Here {sampling_type}")
                    axis[i].annotate(f'{sampling_type}', xy=(1.4,0.5),rotation=0,ha='center',va='center',xycoords='axes fraction')
                axis[i].set_title(f'Pearson={corr[0]:.3f}\nSpearman={corr[1]:.3f}')
            if j != 0:
                axis[i].set_ylabel("")
            if i != num_sampling_types :
                axis[i].set_xlabel("")
            if i==0:
                axis[i].set_title(f"{dataset_names[j]}")
    fig.subplots_adjust(left=0.01,bottom=0.05,top=0.9,wspace=0.4,hspace=0.6)
    plt.suptitle(f"When to Retrain {method_str} ",fontsize=30)
    if save_fig:
        plt.savefig(figure_dir/f"When to retrain {method}.pdf",bbox_inches="tight",dpi=300)
    else:
        plt.show()

# %%
if __name__ == "__main__":
    mpl.rcParams["font.size"]=10
    plot_when_to_retrain_dist(results_dir,method="Guo",save_fig=save_fig)
# %%
    mpl.rcParams["font.size"]=10
    plot_when_to_retrain_dist(results_dir,method="Golatkar",save_fig=save_fig)
# %%
def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str, **kwargs):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', **kwargs)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

@mpl.ticker.FuncFormatter
def major_formatter(x, pos):
    return "{0}".format(str(round(x, 3) if (x % 1) and (round(x,3) != 0) else int(x)))

def plot_unlearning(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,latex:bool=False,fig_width_pt:float=246.0,scale:float=3):
    if plot_deltagrad:
        num_methods = 3
    else:
        num_methods = 2
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabel = r"Accuracy Error (\texttt{AccErr}) \%"
        del_ylabel = r"Accuracy Disparity (\texttt{AccDis}) \%"
        sape = r"$\texttt{sAPE}$"
        method_labels = [r"$\textsc{Influence}$",r"$\textsc{Fisher}$",r"$\textsc{DeltaGrad}$"]

    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"
        method_labels = ["${INFL}$","${FISH}$","${DG}$"]
    
    method_colors = ["tab:blue","tab:orange","tab:green"]
    subplots = (num_methods,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=0
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(left=0.05,right=0.93,bottom=0.1,top=0.9,wspace=0.5,hspace=0.2)

    methods = ["Guo","Golatkar","deltagrad"]
    method_short = ["guo","gol","deltagrad"]
    QoA_column_name = ["minibatch_fraction","minibatch_fraction","period"]
    # The QoA values for Guo, Golatkar and DeltaGrad
    QoA_values= [1,1,100]
    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=True)
        axis = ax[:,j]
        for i,method in enumerate(methods[:num_methods]):
            kwargs=dict(
                markersize=10,
            )
            twin_ax = axis[i].twinx()
            axis[i],twin_ax = unlearn_plots.plot_unlearn(
                "remove_accuracy",
                method,
                dfs_dict["baseline"],
                dfs_dict[f"baseline_{method_short[i]}"],
                dfs_dict[method_short[i]],
                QoA_column=QoA_column_name[i],
                QoA=QoA_values[i],
                ax1=axis[i],
                ax2=twin_ax,
                **kwargs
            ) 
            axis[i].set_ylabel("")
            twin_ax.set_ylabel("")
            axis[i].set_xlabel("")
            if datasets[j] == "HIGGS" and i==0:
                twin_ax.set_yticks([0.158,0.168])
            if j == len(datasets)-1:
                axis[i].annotate(f"{method_labels[i]}",#fontsize=30,
                 xy=(1.5,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction',color=method_colors[i])
            if i==0 and j==0:
                axis[i].annotate(f"{del_ylabel}",# fontsize=30,
                 xy=(0.01,0.5), rotation=90,ha='center',va='center',xycoords='figure fraction',color="tab:red")
                axis[i].annotate(f"{test_ylabel}",# fontsize=30,
                 xy=(0.97,0.5), rotation=-90,ha='center',va='center',xycoords='figure fraction',color="black")
                axis[i].annotate(r"Noise Paramter $\sigma$",# fontsize=30,
                 xy=(0.48,0.02), rotation=0,ha='center',va='center',xycoords='figure fraction')
            if i!= num_methods-1:
                axis[i].set_xlabel("")
            
            axis[i].get_legend().remove()
            twin_ax.get_legend().remove()
            if i==0:
                axis[0].set_title(dataset_names[j])#,fontsize=30)
            axis[i].tick_params(axis="both",which="major")#,labelsize=30)
            twin_ax.tick_params(axis="both",which="major")#,labelsize=30)
            axis[i].set_xticks([1e-2,1e0,1e2])
            if i != num_methods-1 :
                axis[i].set_xticklabels([])
    

    if save_fig:
        plt.savefig(figure_dir/"Unlearning_Tradeoff_Grid.pdf",bbox_inches="tight",dpi=300)
    else:
        plt.show()
#%%
if __name__ == "__main__":
    mpl.rcParams["font.size"]=20
    plot_unlearning(results_dir,plot_deltagrad=True,save_fig=save_fig)
# %%

def plot_unlearning_certifiability(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",noise:float=1,remove_ratio_idx:int=2,latex:bool=False,fig_width_pt:float=246.0,scale:float=3):
    num_ratios=1
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        del_ylabel = r"\texttt{AccDis} \%"
        sape = r"$\texttt{sAPE}$"
        label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"
        sape = r"$sAPE$"
    subplots = (num_ratios,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=0.5
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(
        left=0.05,
        bottom=0.3,
        top=0.99,
        wspace=0.3,
        hspace=0.3
        ) 
    kwargs = dict(markersize=10,markeredgecolor="black")
    relative_test_accuracy_drops_names=["small","medium","large"]
    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,ratio_index=remove_ratio_idx)
        axis = ax[:,j]
        ratios = dfs_dict["ratio"]
        i=0  
        axis[i] = unlearn_plots.plot_unlearn_certifiability(f"remove_{metric_map[ovr_strs[j]]}",dfs_dict,noise=noise,legend=True,ax=axis[i],**kwargs)
        axis[i].set_ylabel("")
        axis[i].set_xlabel("")
        axis[i].get_legend().remove()
        if i==0 and j==0:
            axis[i].annotate(f'Speed-up',#fontsize=30,
                xy=(0.5,0.05), rotation=0,ha='center',va='center',xycoords='figure fraction')
            y_label = del_ylabel
            axis[i].annotate(f'{y_label}',#size=30,
                xy=(0.01,0.6), rotation=90,ha='center',va='center',xycoords='figure fraction')
        if i==0:
            axis[i].set_title(dataset_names[j])#,fontsize=30)
        if j == len(datasets)-1:
                axis[i].annotate(f'{relative_test_accuracy_drops_names[remove_ratio_idx]}', #fontsize=30,
                 xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')
        axis[i].set_yscale("symlog",linthresh=0.1)
        axis[i].set_ylim(bottom=-0.05,top=1e2+0.1)
        axis[i].xaxis.set_major_formatter(speed_up)
        if dataset_base_names[j]=="mnist_binary":
            axis[i].set_xticks([1,3,27,243])
        if dataset_base_names[j]=="mnist_multi":
            axis[i].set_xticks([0.3,1,3,9,27])
        if datasets[j] == "COVTYPE":
            axis[i].set_xticks([1,3,27,243])
        if datasets[j] == "HIGGS":
            axis[i].set_xticks([1,9,27,81])
        if datasets[j] == "CIFAR":
            axis[i].set_xticks([1,3,9,27])
        if datasets[j] == "EPSILON":
            axis[i].set_xticks([0.3,1,3])

    if save_fig: 
        if noise ==1 and remove_ratio_idx==2:
            plt.savefig(figure_dir/"Certifiability_Efficiency_Trade_Off_Grid.pdf",bbox_inches="tight")
        
        plt.savefig(figure_dir/f"Certifiability_Efficiency_Trade_Off_Grid_{relative_test_accuracy_drops_names[remove_ratio_idx]}_sigma_{'_'.join(str(noise).split('.'))}.pdf",bbox_inches="tight")
    else:
        plt.show()

# %%
if __name__ == "__main__":
    mpl.rcParams["font.size"]=20
    plot_unlearning_certifiability(results_dir,plot_deltagrad=True,save_fig=save_fig,suffix="_selected",noise=1,y_metric="test")
# %%

def plot_unlearning_effectiveness(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",noise:float=1,remove_ratio_idx:int=2,latex:bool=False,fig_width_pt:float=246.0,scale:float=3):
    num_ratios=1
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabel = r"\texttt{AccErr} \%"
        sape = r"$\texttt{sAPE}$"
        label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        sape = r"$sAPE$"
    subplots = (num_ratios,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=0.5
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(
        left=0.05,
        bottom=0.3,
        top=0.99,
        wspace=0.3,
        hspace=0.3
        ) 
    
    kwargs = dict(markersize=10,markeredgecolor="black")
    relative_test_accuracy_drops_names=["small","medium","large"]

    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,ratio_index=remove_ratio_idx)
        axis = ax[:,j]
        ratios = dfs_dict["ratio"]
        i=0  
        axis[i] = unlearn_plots.plot_unlearn_effectiveness(f"test_{metric_map[ovr_strs[j]]}",dfs_dict,noise=noise,legend=True,ax=axis[i],**kwargs)
        axis[i].set_ylabel("")
        axis[i].set_xlabel("")
        axis[i].get_legend().remove()
        if i==0 and j==0:
            axis[i].annotate(f'Speedup',#fontsize=30,
                xy=(0.5,0.05), rotation=0,ha='center',va='center',xycoords='figure fraction')
            y_label = test_ylabel
            axis[i].annotate(f'{y_label}',#size=30,
                xy=(0.01,0.6), rotation=90,ha='center',va='center',xycoords='figure fraction')
        if i==0:
            axis[i].set_title(dataset_names[j])#,fontsize=30)
        if j == len(datasets)-1:
                axis[i].annotate(f'{relative_test_accuracy_drops_names[remove_ratio_idx]}', #fontsize=30,
                 xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')
        axis[i].set_yscale("symlog",linthresh=0.1)
        axis[i].set_ylim(bottom=-0.05,top=1e2+0.1)
        axis[i].xaxis.set_major_formatter(speed_up)
        if dataset_base_names[j]=="mnist_binary":
            axis[i].set_xticks([1,3,27,243])
        if dataset_base_names[j]=="mnist_multi":
            axis[i].set_xticks([0.3,1,3,9,27])
        if datasets[j] == "COVTYPE":
            axis[i].set_xticks([1,3,27,243])
        if datasets[j] == "HIGGS":
            axis[i].set_xticks([1,9,27,81])
        if datasets[j] == "CIFAR":
            axis[i].set_xticks([0.03,0.3,1,3,9,27])
        if datasets[j] == "EPSILON":
            axis[i].set_xticks([0.3,1,3])
    if save_fig:
        if noise == 1 and remove_ratio_idx==2:
            plt.savefig(figure_dir/"Effectiveness_Efficiency_Trade_Off_Grid.pdf",bbox_inches="tight")
        plt.savefig(figure_dir/f"Effectiveness_Efficiency_Trade_Off_Grid_{relative_test_accuracy_drops_names[remove_ratio_idx]}_sigma_{'_'.join(str(noise).split('.'))}.pdf",bbox_inches="tight")

    else:
        plt.show()
#%%
if __name__ == "__main__":
    mpl.rcParams["font.size"]=20
    plot_unlearning_effectiveness(results_dir,plot_deltagrad=True,save_fig=save_fig,suffix="_selected",noise=1)

# %%
def plot_unlearning_efficiency(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",noise:float=1,latex:bool=False):
    num_ratios=1
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabels = [r"$\epsilon$ \%",r"\texttt{AccDrop} \%"]
        sape = r"$\texttt{sAPE}$"
        label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}
    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabels = [r"${Acc}_{del}$",r"${Acc}_{test}$",]
        sape = r"$sAPE$"
    fig,ax = plt.subplots(2*num_ratios,len(datasets),figsize=(4*len(datasets),4*(2*num_ratios)),squeeze=False)
    
    
    plot_fns = [unlearn_plots.plot_unlearn_certifiability,unlearn_plots.plot_unlearn_effectiveness]
    y_metrics = ["remove","test"] # for certifiablility and effectiveness respectively 
    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad)
        axis = ax[:,j]
        ratios = dfs_dict["ratio"]
        for i,plot_fn in enumerate(plot_fns):
            axis[0] = plot_fn(f"{y_metrics[i]}_{metric_map[ovr_strs[j]]}",dfs_dict,noise=noise,legend=True,ax=axis[i])

            axis[i].set_xlabel("")
            if i == 0 and j ==0:
                handles,labels = axis[i].get_legend_handles_labels() 
                # next each QoA 
                if plot_deltagrad:
                    args_sort_index = [
                                        0,1,2, #first the 3 removal steps
                                        3,7,11, # first QoA
                                        4,8,12, # second QoA
                                        5,9,13, # third QoA
                                        6,10,14 # fourth QoA
                    ]
                else:
                    args_sort_index = [
                                        0,1, #first the 2 removal steps
                                        2,6, # first QoA
                                        3,7, # second QoA
                                        4,8, # third QoA
                                        5,9, # fourth QoA
                    ]

                handles = list(np.array(handles)[args_sort_index])
                labels = list(np.array(labels)[args_sort_index])
                if latex:
                    labels = [label_map[l] if l in label_map else l  for l in labels ]
                
                axis[i].legend(
                    handles=handles,
                    labels=labels,
                    bbox_to_anchor=(0.5,-0.05),
                    loc="upper center",
                    ncol=5,
                    bbox_transform=fig.transFigure,
                    fontsize=30
                )
            else:
                axis[i].get_legend().remove()
            if i==0 and j==0:
                axis[i].annotate(f'Speedup',fontsize=30,
                    xy=(0.5,0.25), rotation=0,ha='center',va='center',xycoords='figure fraction')
            if j==0:
                y_label = test_ylabels[i]
                # axis[i].annotate(f'{y_label}',size=30,
                    # xy=(-0.5,0.5), rotation=90,ha='center',va='center',xycoords='axes fraction')
                axis[i].set_ylabel(y_label,fontsize=30)
            else:
                axis[i].set_ylabel("")
            if i==0:
                axis[i].set_title(dataset_names[j],fontsize=30)

            ticks_loc = axis[i].get_xticks().tolist()+[1]        # 
            # print(ticks_loc)
            axis[i].xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
            axis[i].set_xticklabels([f"{int(t)}x" if t>=1 else f"{t:.1f}x" for t in ticks_loc ] )
            axis[i].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            axis[i].minorticks_on()
            axis[i].tick_params(axis="both",which="major",labelsize=28)
            axis[i].set_yscale("symlog",linthresh=0.1)
            axis[i].set_ylim(bottom=-0.05,top=1e2+0.1)
    
    grid = plt.GridSpec(2*num_ratios,len(datasets))
    for i,method_label in enumerate(["Certifiability-Efficiency Trade-Off","Effectiveness-Efficiency Trade-Off"]):
        create_subtitle(fig, grid[i, ::], f"({chr(97+i)}) {method_label}",**dict(fontsize=30))
    
    fig.subplots_adjust(
        left=0.1,
        bottom=0.1,
        top=0.75,
        wspace=0.3,
        hspace=0.75
        ) 
    if save_fig:
        if noise == 1:
            plt.savefig(figure_dir/"Unlearning_Efficiency_Trade_Off_Grid.pdf",bbox_inches="tight",dpi=300)
        plt.savefig(figure_dir/f"Unlearning_Efficiency_Trade_Off_Grid_sigma_{noise}.pdf",bbox_inches="tight",dpi=300)
        

    else:
        plt.show()

#%%
def plot_unlearning_appendix(results_dir:Path,method:str,remove_ratio_idx:int,save_fig:bool=False,latex:bool=False,fig_width_pt:float=246.0,scale:float=3):
    
    assert method in ["guo","gol","deltagrad"]
    assert remove_ratio_idx < 3

    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        test_ylabel = r"Accuracy Error (\texttt{AccErr}) \%"
        del_ylabel = r"Accuracy Disparity (\texttt{AccDis}) \%"
        sape = r"$\texttt{sAPE}$"
        method_labels = [r"$\textsc{Influence}$",r"$\textsc{Fisher}$",r"$\textsc{DeltaGrad}$"]

    else:
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        test_ylabel = r"${Acc}_{test}$"
        del_ylabel = r"${Acc}_{del}$"
        method_labels = ["${INFL}$","${FISH}$","${DG}$"]
    
    method_name_long = {"guo":"Guo","gol":"Golatkar","deltagrad":"deltagrad"}
    QoA_column_name = {"guo":"minibatch_fraction","gol":"minibatch_fraction","deltagrad":"period"}
    QoA_range= {"guo":[1,2,4,8], "gol":[1,2,4,8], "deltagrad":[2,5,50,100]}
    deletion_volumes=["small","medium","large"]

    num_rows = len(QoA_range[method])
    subplots = (num_rows,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=2
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(left=0.05,right=0.93,bottom=0.1,top=0.9,wspace=0.55,hspace=0.3)

    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=True,ratio_index=remove_ratio_idx)
        axis = ax[:,j]
        for i,QoA in enumerate(QoA_range[method]):
            kwargs=dict(
                markersize=10,
            )
            twin_ax = axis[i].twinx()
            axis[i],twin_ax = unlearn_plots.plot_unlearn(
                "remove_accuracy",
                method_name_long[method],
                dfs_dict["baseline"],
                dfs_dict[f"baseline_{method}"],
                dfs_dict[method],
                QoA_column=QoA_column_name[method],
                QoA=QoA,
                ax1=axis[i],
                ax2=twin_ax,
                **kwargs
            ) 
            axis[i].set_ylabel("")
            twin_ax.set_ylabel("")
            axis[i].set_xlabel("")
            axis[i].set_xticks([1e-2,1e0,1e2])
            if j == len(datasets)-1:
                if method == "guo" or method == "gol":
                    axis[i].annotate(f"$m^{{\prime}}=m/{QoA}$",#fontsize=30,
                    xy=(1.3,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction',)
                elif method == "deltagrad":
                    axis[i].annotate(f"$T_0={QoA}$",#fontsize=30,
                    xy=(1.3,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction',)
            if i==0 and j==0:
                axis[i].annotate(f"{del_ylabel}",# fontsize=30,
                 xy=(0.01,0.5), rotation=90,ha='center',va='center',xycoords='figure fraction',color="tab:red")
                axis[i].annotate(f"{test_ylabel}",# fontsize=30,
                 xy=(0.99,0.5), rotation=-90,ha='center',va='center',xycoords='figure fraction',color="black")
                axis[i].annotate(r"Noise Paramter $\sigma$",# fontsize=30,
                 xy=(0.48,0.03), rotation=0,ha='center',va='center',xycoords='figure fraction')
            if i!= num_rows-1:
                axis[i].set_xlabel("")
            
            axis[i].get_legend().remove()
            twin_ax.get_legend().remove()
            if i==0:
                axis[0].set_title(dataset_names[j])#,fontsize=30)
    if save_fig:
        plt.savefig(figure_dir/f"Certifiability_Effectiveness_Tradeoff_Grid_{method}_{deletion_volumes[remove_ratio_idx]}.pdf",bbox_inches="tight")
    else:
        plt.show()

#%%

def plot_speedup(results_dir:Path,strategy:str,noise_level:float=0,save_fig:bool=False,latex:bool=False,fig_width_pt:float=246.0,scale:float=3):
    
    assert strategy in ["gol_test","gol_dis_v1","gol_dis_v2"]
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    if latex:
        dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"$\textsc{mnist}$",
            r"$\textsc{covtype}$",
            r"$\textsc{higgs}$",
            r"$\textsc{cifar2}$",
            r"$\textsc{epsilon}$"
        ]
        y_label="Speed-up"
        x_label="No. Deletions"
        method_labels = [r"$\textsc{Influence}$",r"$\textsc{Fisher}$",r"$\textsc{DeltaGrad}$"]

    else:
        y_label = "Speed-Up"
        x_label="No. Deletions"
        dataset_names = ["$MNIST^b$","MNIST","COVTYPE","HIGGS","CIFAR2","EPSILON"]
        method_labels = ["${INFL}$","${FISH}$","${DG}$"]
    
    
    sampling_types = ["uniform_random","uniform_informed","targeted_random","targeted_informed"]
    # sampling_types = ["uniform_random","targeted_informed"]
    num_rows = len(sampling_types)
    subplots = (num_rows,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)


    if len(sampling_types)==4:
        full_str = "full_"
        hspace = 0.3
        rotation = 0
        figsize[1]+=2
        xy=(1.6,0.5)
    else:
        full_str=""
        rotation = 270
        hspace = 0.12
        xy=(1.3,0.5)
        figsize[1]+=1.2

    fig.subplots_adjust(left=0.05,right=0.93,bottom=0.1,top=0.9,wspace=0.35,hspace=hspace)
    for j,dataset in enumerate(datasets):
        data = load_dfs(results_dir,dataset,ovr_strs[j])
        data = compute_all_metrics(data)
        data = data._asdict()
        df = data[strategy]
        if strategy in ["gol_dis_v1","gol_dis_v2"]:
            if dataset == "HIGGS":
                df = df[df.threshold.isin([10,20,50])]
            else:
                df = df[df.threshold.isin([1,2,5])]
        else:
            df = df[df.threshold.isin([0.25,0.5,1])]
        for i,sampling_type in enumerate(sampling_types):
            print(i,j,dataset,sampling_type)
            axis = ax[i,j]
            axis = plot_metric(df,"speedup",sampling_type,noise_level,ax=axis)
            if dataset not in ["MNIST","CIFAR","EPSILON"]:
                axis.set_yscale("log")
            axis.axhline(1,color="black",linestyle="--",alpha=0.5)
            if i == 0 and j==0 :
                axis.legend(bbox_to_anchor=(0.5,-0.05),
                    labels=[f"$\kappa={t}$" for t in sorted(df.threshold.unique())],
                    loc="upper center",
                    ncol=5,
                    bbox_transform=fig.transFigure)
                axis.annotate("Num Deletions",#fontsize=30,
                xy=(0.5,0.12), rotation=0,ha='center',va='center',xycoords='figure fraction'
                )
                axis.annotate(y_label,#fontsize=30,
                xy=(0.01,0.6), rotation=90,ha='center',va='center',xycoords='figure fraction')
            else:
                axis.get_legend().remove()
            
            if j == len(datasets)-1:
                axis.annotate(fr"$\texttt{{{'-'.join(sampling_type.split('_'))}}}$", #fontsize=30,
                 xy=xy, rotation=rotation,ha='center',va='center',xycoords='axes fraction')
            axis.set_ylabel("")
            axis.set_xlabel("")
            if i != len(sampling_types)-1:
                axis.set_xticks([])

            if i ==0:
                axis.set_title(dataset_names[j])
    
    if save_fig:
        plt.savefig(figure_dir/f"Pipeline_{strategy}_speedup_{full_str}grid_noise_{noise_level}.pdf",bbox_inches="tight")


def print_combined_error_metrics(results_dir:Path,strategy:str,metric:str,noise_level:float=0):
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    dataset_names = [
            r"$\textsc{mnist}^{\text{b}}$",
            r"\textsc{mnist}",
            r"\textsc{covtype}",
            r"\textsc{higgs}",
            r"\textsc{cifar2}",
            r"\textsc{epsilon}"
        ]
    sampling_types = ["uniform_random","uniform_informed","targeted_random","targeted_informed"]

    if strategy == "gol_test":
        valid_thresholds = [0.25,0.5,1]
    elif strategy in ["gol_dis_v1","gol_dis_v2"]:
        valid_thresholds = [1,2,5]
    temp =[ ]
    for j,dataset in enumerate(datasets):
        temp1 = []
        data = load_dfs(results_dir,dataset,ovr_strs[j])
        data = compute_all_metrics(data)
        data = data._asdict()
        df = data[strategy]
        if strategy in ["gol_dis_v1","gol_dis_v2"] and dataset == "HIGGS":
            df = df[df.threshold.isin([1,2,5,10,20,50])]
        else:
            df = df[df.threshold.isin(valid_thresholds)]
        for sampling_type in sampling_types:
            temp1.append(compute_error_metrics(df,metric,sampling_type,noise_level))
        temp1 = pd.concat(temp1)
        temp1 = temp1.reset_index()
        temp1["dataset"] = dataset_names[j]
        temp.append(temp1)
        
    
    temp = pd.concat(temp)
    temp["output_str"]=temp.apply(lambda row: rf"${row['mean']:.2f}\pm{row['std']:.1f}$",axis=1)
    temp["sampling type"] = temp.sampling_type.apply(lambda x : fr"$\texttt{{{'-'.join(x.split('_'))}}}$")
    temp = temp.pivot(index=["dataset","threshold"],columns="sampling type",values="output_str")
    temp = temp.sort_index(ascending=False,axis=1)
    temp = temp.loc[dataset_names]
    print(temp.to_latex(multirow=True,escape=False))
# %%

def plot_custom(results_dir:Path,plot_deltagrad=False,save_fig:bool=False,suffix:str="",noise:float=1,remove_ratio_idx:int=2,fig_width_pt:float=246.0,scale:float=3):
    num_row=2
    figure_dir = results_dir/"images"
    dataset_base_names = ["mnist_binary","mnist_multi","covtype_binary","higgs_binary","cifar_binary","epsilon_binary"]
    ovr_strs = [dataset.split("_")[1] for dataset in dataset_base_names]
    metric_map = {"binary":"accuracy","multi":"accuracy"}
    datasets = [dataset.split("_")[0].upper() for dataset in dataset_base_names]
    dataset_names = [
        r"$\textsc{mnist}^{\text{b}}$",
        r"$\textsc{mnist}$",
        r"$\textsc{covtype}$",
        r"$\textsc{higgs}$",
        r"$\textsc{cifar2}$",
        r"$\textsc{epsilon}$"
    ]
    dis_ylabel = r"\texttt{AccDis} \%"
    err_ylabel = r"\texttt{AccErr} \%"
    sape = r"$\texttt{sAPE}$"
    label_map = {"${INF}$":r"$\textsc{Influence}$","${FISH}$":r"$\textsc{Fisher}$","${DG}$":r"$\textsc{DeltaGrad}$"}

    subplots = (num_row,len(datasets))
    figsize = set_size(fig_width_pt,subplots=(subplots[0],subplots[1]))
    figsize = np.array(figsize)*scale
    figsize[1]+=0.5
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(
        left=0.05,
        bottom=0.15,
        top=0.99,
        wspace=0.2,
        hspace=0.4
        ) 
    kwargs = dict(markersize=10,markeredgecolor="black")
    relative_test_accuracy_drops_names=["small","medium","large"]
    for j in range(len(datasets)):
        dfs_dict = unlearn_plots.load_dfs(results_dir,datasets[j],ovr_strs[j],plot_deltagrad=plot_deltagrad,ratio_index=remove_ratio_idx)
        axis = ax[:,j]
        for i, subcaption in enumerate(["(a)","(b)"]):
            if i ==0 :

                axis[i] = unlearn_plots.plot_unlearn_certifiability(f"remove_{metric_map[ovr_strs[j]]}",dfs_dict,noise=noise,legend=True,ax=axis[i],**kwargs)
            else:
                axis[i] = unlearn_plots.plot_unlearn_effectiveness(f"test_{metric_map[ovr_strs[j]]}",dfs_dict,noise=noise,legend=True,ax=axis[i],**kwargs)
            axis[i].set_ylabel("")
            axis[i].set_xlabel("")
            axis[i].get_legend().remove()
            if j==0:
                axis[i].annotate(f'Speed-up',#fontsize=30,
                    xy=(0.5,0.04), rotation=0,ha='center',va='center',xycoords='figure fraction')
                if i ==0 :
                    y_label = dis_ylabel
                else:
                    y_label = err_ylabel
                axis[i].annotate(f'{y_label}',#size=30,
                    xy=(-0.4,0.5), rotation=90,ha='center',va='center',xycoords='axes fraction')
            
            if i==0:
                axis[i].set_title(dataset_names[j])#,fontsize=30)
            if j == len(datasets)-1:
                    axis[i].annotate(subcaption, #fontsize=30,
                    xy=(1.1,0.5), rotation=0,ha='center',va='center',xycoords='axes fraction')
            if j == 0 :
                axis[i].set_yscale("symlog",linthresh=0.1)
                axis[i].set_ylim(bottom=-0.05,top=1e2+0.1)
            else:
                axis[i].set_yticks([])
            axis[i].xaxis.set_major_formatter(speed_up)
            if dataset_base_names[j]=="mnist_binary":
                axis[i].set_xticks([1,3,27,243])
            if dataset_base_names[j]=="mnist_multi":
                axis[i].set_xticks([0.3,1,3,9,27])
            if datasets[j] == "COVTYPE":
                axis[i].set_xticks([1,3,27,243])
            if datasets[j] == "HIGGS":
                axis[i].set_xticks([1,9,27,81])
            if datasets[j] == "CIFAR":
                if i == 0:
                    axis[i].set_xticks([1,3,9,27])
                else:
                    axis[i].set_xticks([0.03,0.3,1,3,9,27])
            if datasets[j] == "EPSILON":
                axis[i].set_xticks([0.3,1,3])
        
    if save_fig:
        plt.savefig(figure_dir/f"Effectiveness_Custom_Grid_{relative_test_accuracy_drops_names[remove_ratio_idx]}_sigma_{'_'.join(str(noise).split('.'))}.pdf",bbox_inches="tight")
    else:
        plt.show()


