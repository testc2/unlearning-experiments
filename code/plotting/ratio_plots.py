#%%
from IPython import get_ipython
from numpy.lib.npyio import save
from pandas.core import base
from seaborn import palettes
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from parse import get_deltagrad_dist_dataframes
#%%

save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
#%%
get_label = lambda string: " ".join(string.split("_")).title()

def plot_metric(y:str,ratio_df:pd.DataFrame,ax=None,**kwargs):
    if ax is None:
        fig,ax = plt.subplots()
    x = "remove_ratio"
    ax = sns.lineplot(data=ratio_df,x=x,y=y,hue="sampling_type",ax=ax,style="sampling_type",**kwargs,legend="full")
    ax.set_xlabel(get_label(x))
    ax.set_ylabel(get_label(y))
    return ax

def plot_class_ratio(sampling_type:str,y:str,ratio_df:pd.DataFrame,ax=None):
    if ax is None:
        fig,ax = plt.subplots()
    filtered_df = ratio_df[ratio_df.sampling_type==sampling_type]
    x = "remove_ratio"
    n_classes = (ratio_df.remove_class.unique()>=0).sum()
    palette = sns.color_palette("husl",n_colors=n_classes)
    ax = sns.lineplot(data=filtered_df,x=x,y=y,hue="remove_class",ax=ax,palette=palette,marker="o")
    ax.set_xlabel(get_label(x))
    ax.set_ylabel(get_label(y))
    ax.set_title(get_label(sampling_type))
    return ax

def load_dfs(results_dir:Path,dataset:str,ovr_str:str,suffix:str=""):
    file = results_dir/dataset/f"Ratio_{ovr_str+suffix}.csv"
    ratio_df = pd.read_csv(file)
    if dataset == "MNIST" and ovr_str == "multi":
        ratio_df.loc[(ratio_df.sampling_type.isin(["targeted_informed","targeted_random"]))&(ratio_df.remove_ratio>0.11),"test_accuracy"]=np.nan
        ratio_df.loc[(ratio_df.sampling_type.isin(["targeted_informed","targeted_random"]))&(ratio_df.remove_ratio>0.11),"remove_accuracy"]=np.nan
    return ratio_df

def plot_ratio(results_dir:Path,dataset:str,ovr:bool,save_fig:bool=False):
    if ovr:
        ovr_str = "multi"
    else:
        ovr_str = "binary"
    figure_dir = results_dir/"images"/dataset
    if not figure_dir.exists():
        figure_dir.mkdir(exist_ok=True,parents=True)

    ratio_df = load_dfs(results_dir,dataset,ovr_str)
    n_classes = (ratio_df.remove_class.unique()>=0).sum()
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
    ax1 = plot_metric("test_accuracy",ratio_df,ax=ax1)
    ax1.get_legend().remove()
    ax2 = plot_metric("model_diff_orig",ratio_df,ax=ax2)
    ax2.set_yscale("log")
    fig.subplots_adjust(top=0.92,hspace = 0.49,bottom=0.2)
    ax2.legend(bbox_to_anchor=(0.5,0.58),bbox_transform=fig.transFigure,ncol=4,title="Sampling Type",loc='upper center')
    
    ax3 = plot_class_ratio("targeted_random","test_accuracy",ratio_df,ax=ax3)
    ax3.get_legend().remove()   
    ax4 = plot_class_ratio("targeted_informed","test_accuracy",ratio_df,ax=ax4)
    ax4.legend(bbox_to_anchor=(0.5,0.15),bbox_transform=fig.transFigure,ncol=n_classes,title="Removed Class",loc='upper center')
    # fig.subplots_adjust(top = 0.1,bottom=0.1)
    plt.suptitle(f"{ovr_str.title()} {dataset}",fontsize=15)
    if save_fig:
        plt.savefig(figure_dir/f"Ratio_plots_{ovr_str}.pdf",dpi=300,bbox_inches="tight")
    else:
        plt.show()

#%%

if __name__ == "__main__":
    file = results_dir/"MNIST"/"Ratio_binary.csv"
    df = pd.read_csv(file)
    mpl.rcParams["figure.dpi"]=100  
    # %%
    avgs = df.groupby(["remove_ratio","sampling_type"]).mean().reset_index()
    #%%
    px.line(avgs,x="remove_ratio",y="test_accuracy",color="sampling_type")
    #%%
    px.line(avgs,x="remove_ratio",y="model_diff_orig",color="sampling_type",log_x=False,log_y=True)
    # %%
    class_avgs = df.groupby(["remove_ratio","sampling_type","remove_class"]).mean().reset_index()
    px.line(class_avgs[class_avgs.sampling_type=="targeted_informed"],x="remove_ratio",y="test_accuracy",color="remove_class")
    plot_ratio(results_dir,"MNIST",ovr=False,save_fig=save_fig)
    # %%
    plot_ratio(results_dir,"MNIST",ovr=True,save_fig=save_fig)
    # %%
    plot_ratio(results_dir,"COVTYPE",ovr=False,save_fig=save_fig)
    # %%
    plot_ratio(results_dir,"HIGGS",ovr=False,save_fig=save_fig)
    # %%
    plot_ratio(results_dir,"EPSILON",ovr=False,save_fig=save_fig)
    # %%
    plot_ratio(results_dir,"CIFAR",ovr=False,save_fig=save_fig)
#%%
    plot_metric("test_accuracy",df,**dict(markersize=12))
# %%
