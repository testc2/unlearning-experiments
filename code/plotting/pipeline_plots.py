#%%
from collections import namedtuple
from typing import Optional
from IPython import get_ipython
from numpy.lib.npyio import load, save
from torch import threshold_
from torch.nn.functional import threshold

from pathlib import Path
import sys
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str((project_dir/"code").resolve()))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    from plotting.combined_plots import set_size
else:
    notebook = False
from plotting.ratio_remove_plots import load_dfs as ratio_load_dfs
from methods.common_utils import SAPE
from scipy.stats import sem
import json
from ast import literal_eval

#%%

Data = namedtuple('Data', ["dataset","ovr_str",'retrain', 'gol', 'nothing', 'gol_test', 'gol_dis_v1', 'gol_dis_v2', 'guo_dis_v1', 'guo_dis_v2'])

def load_df(exp_dir:Path,ovr_str:str,strategy_file_prefix:str,strategy_name:str):
    temp = []
    for file in exp_dir.glob(f"{strategy_file_prefix}_{ovr_str}*.csv"):
        df = pd.read_csv(file)
        # print(file.stem,len(df))
        temp.append(df)
    if len(temp):
        df = pd.concat(temp)     
        # to make the NaN noise and noise seed 0
        df.noise.fillna(0,inplace=True)
        df.noise_seed.fillna(0,inplace=True)
        df = df.infer_objects()
        df["strategy"] = strategy_name
    else:
        df = pd.DataFrame()
    return df

def load_dfs(results_dir:Path,dataset:str,ovr_str:str):
    exp_dir = results_dir/dataset/"when_to_retrain"
    retrain_df = load_df(exp_dir,ovr_str,"retrain","retrain")
    retrain_df.checkpoint_remove_accuracy.fillna("None",inplace=True)
    retrain_df["checkpoint_remove_accuracy"] = retrain_df.checkpoint_remove_accuracy.apply(literal_eval)
    gol_df = load_df(exp_dir,ovr_str,"golatkar","Golatkar")
    nothing_df = load_df(exp_dir,ovr_str,"nothing","nothing")
    gol_test_df = load_df(exp_dir,ovr_str,"golatkar_test_thresh","Golatkar Test")
    gol_test_df["strategy"] = gol_test_df.threshold.apply(lambda t: f"Gol Test Threshold {t} %")
   
    gol_dis_v1_df = load_df(exp_dir,ovr_str,"golatkar_disparity_thresh_v1","Golatkar Disparity")
    gol_dis_v1_df["strategy"] = gol_dis_v1_df.threshold.apply(lambda t: f"Gol Dis V1 Threshold {t} %")
    gol_dis_v2_df = load_df(exp_dir,ovr_str,"golatkar_disparity_thresh_v2","Golatkar Disparity")
    gol_dis_v2_df["strategy"] = gol_dis_v2_df.threshold.apply(lambda t: f"Gol Dis V2 Threshold {t} %")

    guo_dis_v1_df = load_df(exp_dir,ovr_str,"guo_disparity_thresh_v1","Guo Disparity")
    if len(guo_dis_v1_df):
        guo_dis_v1_df["strategy"] = guo_dis_v1_df.threshold.apply(lambda t: f"Guo V1 Dis Threshold {t} %")
    guo_dis_v2_df = load_df(exp_dir,ovr_str,"guo_disparity_thresh_v2","Guo Disparity")
    if len(guo_dis_v2_df):
        guo_dis_v2_df["strategy"] = guo_dis_v2_df.threshold.apply(lambda t: f"Guo V2 Dis Threshold {t} %")
    
    return Data(dataset,ovr_str,retrain_df,gol_df,nothing_df,gol_test_df,gol_dis_v1_df,gol_dis_v2_df,guo_dis_v1_df,guo_dis_v2_df)
# %%
noise_filter = lambda df,noise: df[df.noise==noise]
threshold_filter = lambda df,threshold : df[df.threshold==threshold]
sampling_type_filter = lambda df,sampling_type : df[df.sampling_type == sampling_type]
#%%

def row_func(row,checkpoint_batches,verbose=False):
    # print(f"Row name: {row.name}")
    # print(f"Checkpoint Batches {checkpoint_batches.values}")
    # print(row.checkpoint_remove_accuracy)
    if np.isnan(checkpoint_batches.values).any():
        return np.nan
    if verbose:
        print(row.checkpoint_remove_accuracy[checkpoint_batches[row.name]],type(row.checkpoint_remove_accuracy[checkpoint_batches[row.name]]))
    return row.checkpoint_remove_accuracy[checkpoint_batches[row.name]]

def compute_metrics(retrain_df,method_df,nothing_df,threshold=None,window_size=20):
    temp = []
    groupby_cols = ["sampling_type","noise"]
    for (sampling_type,noise),df in retrain_df.groupby(groupby_cols):
        method_filter_df = noise_filter(sampling_type_filter(method_df,sampling_type),noise) 
        nothing_filter_df = noise_filter(sampling_type_filter(nothing_df,sampling_type),noise) 
        groupby_cols_1 = ["sampler_seed","noise_seed"]
        if len(method_filter_df):
            # print(sampling_type,noise)
            for ((g1,r_df),(g2,m_df),(g3,n_df)) in zip(df.groupby(groupby_cols_1),method_filter_df.groupby(groupby_cols_1),nothing_filter_df.groupby(groupby_cols_1)):
                # print(g1,len(r_df),len(m_df),len(n_df))
                assert g1 == g2 == g3
                r_df.set_index("num_deletions",inplace=True)
                m_df["acc_dis"] = SAPE(m_df["cum_remove_accuracy"].values,r_df.loc[m_df.num_deletions,"cum_remove_accuracy"].values)
                m_df["abs_dis"] = np.abs(m_df["cum_remove_accuracy"].values-r_df.loc[m_df.num_deletions,"cum_remove_accuracy"].values)
                m_df["acc_dis_rolling_avg"] = m_df.acc_dis.rolling(window=window_size).mean()
                m_df["acc_dis_batch"] = SAPE(m_df["batch_remove_accuracy"].values,r_df.loc[m_df.num_deletions,"batch_remove_accuracy"].values)
                m_df["acc_dis_batch_rolling_avg"] = m_df.acc_dis_batch.rolling(window=window_size).mean()
                m_df["acc_err"] = SAPE(m_df["test_accuracy"].values,r_df.loc[m_df.num_deletions,"test_accuracy"].values)
                m_df["abs_err"] = np.abs(m_df["test_accuracy"].values-r_df.loc[m_df.num_deletions,"test_accuracy"].values)
                m_df["acc_err_rolling_avg"] = m_df.acc_err.rolling(window=window_size).mean()
                m_df["acc_err_init"] = SAPE(m_df["test_accuracy"].values,n_df.test_accuracy.values[0])
                m_df["abs_err_init"] = np.abs(m_df["test_accuracy"].values-n_df.test_accuracy.values[0])
                m_df["cum_running_time"] = m_df.running_time.cumsum()
                m_df["cum_unlearning_time"] = m_df.unlearning_time.cumsum()
                m_df["cum_retraining_time"] = m_df.retraining_time.cumsum()
                m_df["cum_other_time"] = m_df.other_time.cumsum()
                m_df["acc_dis_cumsum"] = m_df.acc_dis.cumsum()
                m_df["acc_dis_cumsum_avg"] = m_df.acc_dis_cumsum.values/m_df.num_deletions
                m_df["acc_dis_batch_cumsum"] = m_df.acc_dis_batch.cumsum()
                m_df["acc_dis_batch_cumsum_avg"] = m_df.acc_dis_batch_cumsum.values/m_df.num_deletions
                m_df["acc_err_cumsum"] = m_df.acc_err.cumsum()
                m_df["acc_err_cumsum_avg"] = m_df.acc_err_cumsum.values/m_df.num_deletions
                m_df["speedup"] = r_df.running_time.cumsum().values/ m_df.cum_running_time.values
                if not m_df.checkpoint_remove_accuracy.isna().any():
                    checkpoint_batches = m_df.set_index("num_deletions").checkpoint_batch
                    retrain_checkpoint_remove_accuracy = r_df.loc[m_df.num_deletions].apply(row_func,args=(checkpoint_batches,),axis=1).values
                    if isinstance(m_df["checkpoint_remove_accuracy"].iloc[0],dict):
                        m_df["checkpoint_remove_accuracy"] = m_df.set_index("num_deletions").apply(row_func,args=(r_df.checkpoint_batch,),axis=1)
                    m_df["checkpoint_acc_dis"] = SAPE(m_df["checkpoint_remove_accuracy"].values,retrain_checkpoint_remove_accuracy)
                    m_df["checkpoint_acc_dis_cumsum"] = m_df.checkpoint_acc_dis.cumsum()
                    m_df["checkpoint_acc_dis_rolling_avg"] = m_df.checkpoint_acc_dis.rolling(window_size).mean()
                if threshold is not None :
                    # find where the acc_err exceeded the threshold
                    # where acc_err was lower than threshold error is considered 0
                    m_df["error_acc_err"] = np.maximum(m_df.acc_err-threshold,0) 
                    m_df["error_acc_err_rel_per"] = (m_df.error_acc_err/threshold)*100
                    m_df["error_acc_err_rel_per_mean"] = m_df.error_acc_err_rel_per.expanding().mean()
                    m_df["error_acc_err_cumsum"] = m_df.error_acc_err.cumsum()
                    m_df["error_acc_err_mean"] = m_df.error_acc_err.expanding().mean()
                    # similarly check for acc_dis
                    m_df["error_acc_dis"] = np.maximum(m_df.acc_dis-threshold,0) 
                    m_df["error_acc_dis_rel_per"] = (m_df.error_acc_dis/threshold)*100
                    m_df["error_acc_dis_rel_per_mean"] = m_df.error_acc_dis_rel_per.expanding().mean()
                    m_df["error_acc_dis_mean"] = m_df.error_acc_dis.expanding().mean()
                    m_df["error_acc_dis_cumsum"] = m_df.error_acc_dis.cumsum()
                    # if checkpoint accuracy is available 
                    if not m_df.pipeline_acc_dis_est.isna().any():
                        # similarly check for checkpoint_acc_dis
                        m_df["error_checkpoint_acc_dis"] = np.maximum(m_df.checkpoint_acc_dis-threshold,0) 
                        m_df["error_checkpoint_acc_dis_rel_per"] = (m_df.error_checkpoint_acc_dis/threshold)*100
                        m_df["error_checkpoint_acc_dis_rel_per_mean"] = m_df.error_checkpoint_acc_dis_rel_per.expanding().mean()
                        m_df["error_checkpoint_acc_dis_mean"] = m_df.error_checkpoint_acc_dis.expanding().mean()
                        m_df["error_checkpoint_acc_dis_cumsum"] = m_df.error_checkpoint_acc_dis.cumsum()
                        # compute the error metric for twice the pipeline threshold 
                        m_df["error_checkpoint_acc_dis_double"] = np.maximum(m_df.checkpoint_acc_dis-(2*threshold),0) 
                        m_df["error_checkpoint_acc_dis_rel_per_double"] = (m_df.error_checkpoint_acc_dis_double/(2*threshold))*100
                        m_df["error_checkpoint_acc_dis_rel_per_mean_double"] = m_df.error_checkpoint_acc_dis_rel_per_double.expanding().mean()
                        m_df["error_checkpoint_acc_dis_mean_double"] = m_df.error_checkpoint_acc_dis_double.expanding().mean()
                        m_df["error_checkpoint_acc_dis_cumsum_double"] = m_df.error_checkpoint_acc_dis_double.cumsum()
                    if not (m_df.pipeline_acc_dis_est.unique()[0] == 'None' or m_df.pipeline_acc_dis_est.isna().any()):
                        m_df["pipeline_acc_dis_est_rolling_avg"] = m_df.pipeline_acc_dis_est.rolling(window_size).mean()
                temp.append(m_df)
    return pd.concat(temp)

def compute_all_metrics(data:Data,window_size=20):
    gol = compute_metrics(data.retrain,data.gol,data.nothing,window_size=window_size)
    gol_test = pd.concat([compute_metrics(data.retrain,df,data.nothing,threshold=t,window_size=window_size) for t,df in data.gol_test.groupby("threshold")])
    gol_dis_v1 = pd.concat([compute_metrics(data.retrain,df,data.nothing,threshold=t,window_size=window_size) for t,df in data.gol_dis_v1.groupby("threshold")])
    gol_dis_v2 = pd.concat([compute_metrics(data.retrain,df,data.nothing,threshold=t,window_size=window_size) for t,df in data.gol_dis_v2.groupby("threshold")])
    if len(data.guo_dis_v1):
        guo_dis_v1 = pd.concat([compute_metrics(data.retrain,df,data.nothing,threshold=t,window_size=window_size) for t,df in data.guo_dis_v1.groupby("threshold")])
    else:
        guo_dis_v1 = data.guo_dis_v1
    if len(data.guo_dis_v2):
        guo_dis_v2 = pd.concat([compute_metrics(data.retrain,df,data.nothing,threshold=t,window_size=window_size) for t,df in data.guo_dis_v2.groupby("threshold")])
    else:
        guo_dis_v2 = data.guo_dis_v2
    nothing = compute_metrics(data.retrain,data.nothing,data.nothing,window_size=window_size)
    retrain = compute_metrics(data.retrain,data.retrain,data.nothing,window_size=window_size)
    
    return Data(data.dataset,data.ovr_str,retrain,gol,nothing,gol_test,gol_dis_v1,gol_dis_v2,guo_dis_v1,guo_dis_v2)
#%%

def plot_grid(data:Data,strategy:str,noise_level:float):
    if strategy == "test":
        df = data.gol_test
    elif strategy == "dis_v1":
        df = data.gol_dis_v1
    elif strategy == "dis_v1":
        df = data.gol_dis_v2
    else:
        raise ValueError(f"stragtegy={strategy} is not valid options")
    

    combined = pd.concat([data.gol,df,data.retrain])
    combined = noise_filter(combined,noise=noise_level)
    fig_width_pt = 234.8775
    scale = 3
    scaled_params = {k:v*scale for k,v in get_default(3).items()}
    # scale["legend.markerscale"]=2
    new_rc_params.update(scaled_params)
    mpl.rcParams.update(new_rc_params)
    subplots = (3,4)
    figsize = set_size(fig_width_pt,subplots=(subplots[0]+1,subplots[1]+2))
    figsize = np.array(figsize)*scale
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(bottom=0.015,top=0.85,wspace=0.3,hspace=0.3)

    for j,sampling_type in enumerate(["uniform_random","uniform_informed","targeted_random","targeted_informed"]):
        temp = sampling_type_filter(combined,sampling_type)
        ax1,ax2,ax3 = ax[:,j]    
        num_strategy = temp.strategy.nunique()
        ax1 = sns.lineplot(data=temp,x="num_deletions",y="speedup",hue="strategy",ax=ax1,style="strategy")
        ax1.set_xlabel("")
        ax1.set_yscale("log")
        ax1.set_title(" ".join(sampling_type.split("_")))

        ax2 = sns.lineplot(data=temp,x="num_deletions",y="acc_dis_cumsum",hue="strategy",ax=ax2,legend=False,style="strategy")
        ax2.set_xlabel("")

        ax3 = sns.lineplot(data=temp,x="num_deletions",y="acc_err_cumsum",hue="strategy",ax=ax3,legend=False,style="strategy")
        
        if j ==0 :
            ax1.legend(
                bbox_to_anchor=(0.5,-0.1),
                loc="upper center",
                bbox_transform=fig.transFigure,
                ncol=4
            )
            ax1.set_ylabel("Speed-Up")
            # ax1.set_ylabel("Cum. Time")
            ax2.set_ylabel("Cum. \n AccDis \%")
            ax3.set_ylabel("Cum. \n AccErr \%")
        else:
            ax1.get_legend().remove()
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax3.set_ylabel("")
        ax3.set_xlabel("\# deletions")
    plt.suptitle(f"{dataset} $\sigma={noise_level}$ strategy={' '.join(strategy.split('_'))}")
    fig.align_ylabels(ax[:,0])
    if save_fig:
        plt.savefig(f"{dataset}_{ovr_str}_pipeline_{strategy}_noise_{noise_level}_grid.pdf",bbox_inches="tight")
    else:
        plt.show()


def compare_noise(data:Data,strategy:"str",sampling_type:str):
    if strategy == "test":
        df = data.gol_test
    elif strategy == "dis_v1":
        df = data.gol_dis_v1
    elif strategy == "dis_v1":
        df = data.gol_dis_v2
    else:
        raise ValueError(f"stragtegy={strategy} is not valid options")
    

    combined = pd.concat([data.gol,df,data.retrain])
    combined = sampling_type_filter(combined,sampling_type)
    fig_width_pt = 234.8775
    scale = 3
    scaled_params = {k:v*scale for k,v in get_default(3).items()}
    # scale["legend.markerscale"]=2
    new_rc_params.update(scaled_params)
    mpl.rcParams.update(new_rc_params)
    subplots = (3,2)
    figsize = set_size(fig_width_pt,subplots=(subplots[0]+1,subplots[1]+2))
    figsize = np.array(figsize)*scale
    fig,ax = plt.subplots(*subplots,figsize=figsize,squeeze=False)
    fig.subplots_adjust(bottom=0.015,top=0.85,wspace=0.3,hspace=0.3)

    for j,noise in enumerate([0,1]):
        temp = noise_filter(combined,noise)
        ax1,ax2,ax3 = ax[:,j]    
        num_strategy = temp.strategy.nunique()
        ax1 = sns.lineplot(data=temp,x="num_deletions",y="speedup",hue="strategy",ax=ax1,style="strategy")
        ax1.set_xlabel("")
        ax1.set_yscale("log")
        ax1.set_title(f"$\sigma= {noise}$")

        ax2 = sns.lineplot(data=temp,x="num_deletions",y="acc_dis_cumsum",hue="strategy",ax=ax2,legend=False,style="strategy")
        ax2.set_xlabel("")

        ax3 = sns.lineplot(data=temp,x="num_deletions",y="acc_err_cumsum",hue="strategy",ax=ax3,legend=False,style="strategy")
        
        if j ==0 :
            ax1.legend(
                bbox_to_anchor=(0.5,-0.1),
                loc="upper center",
                bbox_transform=fig.transFigure,
                ncol=4
            )
            ax1.set_ylabel("Speed-Up")
            # ax1.set_ylabel("Cum. Time")
            ax2.set_ylabel("Cum. \n AccDis \%")
            ax3.set_ylabel("Cum. \n AccErr \%")
        else:
            ax1.get_legend().remove()
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax3.set_ylabel("")
        ax3.set_xlabel("\# deletions")
    plt.suptitle(f"{dataset} {' '.join(sampling_type.split('_'))} strategy={' '.join(strategy.split('_'))}")
    fig.align_ylabels(ax[:,0])
    if save_fig:
        plt.savefig(f"{dataset}_{ovr_str}_pipeline_noise_strategies.pdf",bbox_inches="tight")
    else:
        plt.show()

def plot_acc_dis_estimation(df:pd.DataFrame,sampling_type:str,noise_level:float,threshold:float,ax:Optional[plt.axes]=None):
    if ax is None:
        fig, ax = plt.subplots()
    temp = threshold_filter(noise_filter(sampling_type_filter(df,sampling_type),noise_level),threshold)
    temp = temp.groupby(["num_deletions","noise_seed","sampler_seed"]).mean().reset_index()
    if not len(temp):
        print("Filtered Dataframe seems to be empty, check the filters")
    temp.plot(x="num_deletions",y="acc_dis",label="True AccDis",ax=ax,marker="s")
    temp.plot(x="num_deletions",y="pipeline_acc_dis_est",label="Estimated AccDis",ax=ax,marker="s")
    ax.axhline(threshold,linestyle="--",color="black",alpha=0.5,marker="s")

    return ax

def plot_acc_dis_versions(data,unlearning_method:str,sampling_type:str,noise_level:float,threshold:float):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    if unlearning_method == "Golatkar":
        df1=data.gol_dis_v1
        df2=data.gol_dis_v2
    elif unlearning_method == "Guo":
        df1=data.guo_dis_v1
        df2=data.guo_dis_v2
    else:
        raise ValueError(f"Unlearning method: {unlearning_method} not recognized")

    ax1 = plot_acc_dis_estimation(df1,sampling_type,noise_level=noise_level,threshold=threshold,ax=ax1)
    ax1.set_title("V1 Strategy")
    ax2 = plot_acc_dis_estimation(df2,sampling_type,noise_level=noise_level,threshold=threshold,ax=ax2)
    ax2.set_title("V2 Strategy")

    plt.suptitle(f"{data.dataset}, {' '.join(sampling_type.split('_'))}, $\sigma={noise_level}$ threshold={threshold}")  

def plot_metric(df:pd.DataFrame,metric:str,sampling_type:str,noise_level:float,threshold:Optional[float]=None,ax:Optional[plt.axes]=None):
    if ax is None:
        fig, ax = plt.subplots()
    temp = noise_filter(sampling_type_filter(df,sampling_type),noise_level)
    if threshold is not None:
        temp = threshold_filter(temp,threshold)
    sns.lineplot(data=temp,x="num_deletions",y=metric,hue="strategy",ax=ax)
    return ax

def ci_95(x):
    return sem(x, ddof=1) * 1.96

def compute_error_metrics(df:pd.DataFrame,metric:str,sampling_type:str,noise_level:float):
    temp = noise_filter(sampling_type_filter(df,sampling_type),noise_level)
    # find the mean and std for each run
    temp =temp.groupby(["threshold","sampling_type","sampler_seed","noise_seed"])[metric].mean()
    # then the average mean and std over all runs 
    res = temp.groupby(["threshold","sampling_type"]).agg(["mean",ci_95])
    return res

def plot_acc_dis_helper(data:Data,sampling_type:str,noise_level:float,threshold:float,ax:plt.Axes=None):
    if ax is None:
        fig,ax = plt.subplots()
    fig = plt.gcf()
    temp = threshold_filter(noise_filter(sampling_type_filter(data.gol_dis_v1,sampling_type),noise_level),threshold)
    temp = temp.query("noise_seed==5 and sampler_seed==0")
    # temp = temp.groupby(["num_deletions","threshold","sampling_type","noise"]).mean().reset_index()
    print(len(temp))
    deletion_batch_size = temp.deletion_batch_size.values[0]
    x_ticks = (temp.num_deletions/deletion_batch_size).round()
    ax.bar((temp.num_deletions/deletion_batch_size).round(),temp.checkpoint_acc_dis.values,label="True")
    ax.plot((temp.num_deletions/deletion_batch_size).round(),temp.pipeline_acc_dis_est,label="Estimate",color="red")
    ax.axhline(threshold,color="black",linestyle="--",alpha=0.5,label="Threshold")
    for i,y in enumerate(temp.query("retrained==True").num_deletions):
        if i ==0:
            kwargs = {"label":"Restart"}
        else:
            kwargs = {}
        ax.axvline((y/deletion_batch_size).round(),linestyle=":",alpha=0.9,color="green",**kwargs)
    ax.legend(bbox_to_anchor=(0.9,0.5),
                    loc="center left",
                    ncol=1,
                    bbox_transform=fig.transFigure)
    ax.set_xticks(x_ticks[::10])
    ax.set_xticklabels(labels=[str(int(x)*deletion_batch_size) for x in x_ticks[::10]])
    ax.set_xlabel("Num Deletions")
    ax.set_ylabel("AccDis %")
    plt.savefig(f"{data.dataset}_{data.ovr_str}_{sampling_type}_threshold_{'_'.join(str(threshold).split('.'))}_acc_dis_noise_{noise_level}.pdf",dpi=300,bbox_inches="tight")

    

def plot_acc_err_helper(data:Data,sampling_type:str,noise_level:float,threshold:float,ax:plt.Axes=None):
    if ax is None:
        fig,ax = plt.subplots()
    fig = plt.gcf()
    temp = threshold_filter(noise_filter(sampling_type_filter(data.gol_test,sampling_type),noise_level),threshold)
    temp = temp.query("noise_seed==4 and sampler_seed==0")
    # temp = temp.groupby(["num_deletions","threshold","sampling_type","noise"]).mean().reset_index()
    print(len(temp))
    deletion_batch_size = temp.deletion_batch_size.values[0]
    x_ticks = (temp.num_deletions/deletion_batch_size).round()
    ax.bar(x_ticks,temp.acc_err.values,label="True")
    ax.plot(x_ticks,temp.pipeline_acc_err,label="Estimate",color="red")
    ax.axhline(threshold,color="black",linestyle="--",alpha=0.5,label="Threshold")
    for i,y in enumerate(temp.query("retrained==True").num_deletions):
        if i ==0:
            kwargs = {"label":"Restart"}
        else:
            kwargs = {}
        ax.axvline((y/deletion_batch_size).round(),linestyle=":",alpha=0.5,color="green",**kwargs)

    ax.set_xticks(x_ticks[::10])
    ax.set_xticklabels(labels=[str(int(x)*deletion_batch_size) for x in x_ticks[::10]])
    ax.set_xlabel("Num  Deletions")
    ax.set_ylabel("AccErr %")
    ax.legend(bbox_to_anchor=(0.9,0.5),
                    loc="center left",
                    ncol=1,
                    bbox_transform=fig.transFigure)
    plt.savefig(f"{data.dataset}_{data.ovr_str}_{sampling_type}_threshold_{'_'.join(str(threshold).split('.'))}_acc_err_noise_{noise_level}.pdf",dpi=300,bbox_inches="tight")

#%%
if __name__ == "__main__":
    def get_default(base:float=6,inc:float=0):
        return{
        'font.size' : base+inc,
        'axes.labelsize' : base+1+inc,
        'legend.fontsize': base+inc,
        'legend.title_fontsize': base+inc,
        'xtick.labelsize' : base+inc,
        # 'xtick.major.size':3.5,
        'ytick.labelsize' : base+inc,
        'figure.titlesize':base+inc,
        }
    default = get_default()
    new_rc_params = {
            'text.usetex': True,
            'figure.dpi':200,
            'savefig.dpi':1000,
            'font.family': 'serif',
            }
    new_rc_params.update(default)
    # mpl.rcParams.update(new_rc_params)
    save_fig = False
    if not save_fig:
        new_rc_params["text.usetex"]=False
    data_dir = project_dir/"data"
    results_dir = data_dir/"results"
    
    dataset = "MNIST"
    ovr_str = "binary"
    data = load_dfs(results_dir,dataset,ovr_str)
    data = compute_all_metrics(data,window_size=10)
    
#%%
    plot_acc_dis_versions(data,"Golatkar","targeted_informed",noise_level=0,threshold=1)
#%%
    plot_metric(data.gol_test,"acc_err","targeted_informed",noise_level=0,threshold=0.1)
#%%
    plot_grid(data,"dis_v1",noise_level=0)
# %%
    df = noise_filter(threshold_filter(data.gol_test,0.1),noise= 1)
    sns.lineplot(data=df,x="num_deletions",y="cum_running_time",hue="sampling_type")
    plt.yscale("log")
    # %%
    df = pd.concat([data.gol_dis_v1,data.gol])
    df = noise_filter(sampling_type_filter(df,"targeted_informed"),noise=0)
    # filter = filter[filter.threshold.isin([1,0.5])]
    sns.lineplot(data=df,x="num_deletions",y="error_acc_dis_cumsum",hue="strategy")
    plt.show()

    # %%
    # plots to compare impact of threshold on a metric for particular noise level and sampling type 
    sampling_type = "targeted_informed"
    noise_level = 0
    metric = "error_checkpoint_acc_dis_mean"
    dataframe = data.gol_dis_v1
    df = noise_filter(sampling_type_filter(dataframe,sampling_type),noise_level)
    sns.lineplot(data=df,x="num_deletions",y=metric,hue="threshold",style="threshold")
    plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$")
    if metric in ["speedup","cum_running_time"]:
        plt.yscale("log")
        plt.axhline(1,color="black",linestyle="--",alpha=0.5)
    plt.show()
    print("Per Batch Average Estimation Error of AccDis")
    print(df.groupby(["threshold","sampler_seed"]).error_acc_dis.mean())
    print(df.groupby(["threshold","sampler_seed"]).error_checkpoint_acc_dis.mean())
#%%

    # plots to compare the impact of noise on AccDis and AccErr for a threshold and sampling type 
    threshold = 0.1
    sampling_type = "targeted_informed"
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    df = sampling_type_filter(threshold_filter(data.gol_dis_v1,threshold),sampling_type)
    ax1= sns.lineplot(data=df,x="num_deletions",y="acc_dis_cumsum",hue="noise",ax=ax1)
    ax2= sns.lineplot(data=df,x="num_deletions",y="acc_err_cumsum",hue="noise",ax=ax2)
    ax1.set_xlabel("Num Deletions")
    ax2.set_xlabel("Num Deletions")
    ax1.set_ylabel("Cum. AccDis")
    ax2.set_ylabel("Cum. AccErr")

    plt.suptitle(f"{dataset}, {' '.join(sampling_type.split('_'))}, Golatkar AccErr threshold: {threshold}%")
    # %%

    # plot to explore the linearity between the AccErr_init and AccDis
    noise_level=0
    sampling_type="targeted_informed"
    df = sampling_type_filter(noise_filter(data.gol,noise_level),sampling_type)
    c = 0.61517267784659
    # c = df.abs_dis.values[-1]/df.abs_err_init.values[-1]
    print(c)
    sns.scatterplot(data=df,x="abs_err_init",y="abs_dis")
    sns.lineplot(x=df.abs_err_init,y=c*df.abs_err_init)
    # c = df.acc_dis.values[-1]/df.acc_err_init.values[-1]
    # sns.lineplot(x=df.acc_err_init,y=c*df.acc_err_init)
    # sns.scatterplot(data=df,x="acc_err_init",y="acc_dis")
    plt.xscale("log")
    plt.yscale("log")
    # %%
    # plots to check the true and estimated pipeline AccDis
    sampling_type = "targeted_informed"
    noise_level = 0
    threshold = 1
    df = threshold_filter(noise_filter(sampling_type_filter(data.gol_dis_v1,sampling_type),noise_level),threshold)
    sns.lineplot(data=df,x="num_deletions",y="pipeline_acc_dis_est",label="pipeline estimate",marker="s")
    sns.lineplot(data=df,x="num_deletions",y="checkpoint_acc_dis",label="true",marker="s")
    plt.axhline(y=threshold,linestyle="--",color="black",alpha=0.5)
    # sns.lineplot(data=df,x="num_deletions",y="acc_dis_cumsum",hue="threshold")
    plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$ Threshold:{threshold}")
    # %%

    # Code to rerun the computations of the pipeline estimations
    sampling_type = "targeted_informed"
    noise_level = 0
    threshold = 1
    df = threshold_filter(noise_filter(sampling_type_filter(data.gol_dis_v1,sampling_type),noise_level),threshold)
    r_df = noise_filter(sampling_type_filter(data.retrain,sampling_type),noise_level)
    c = df.prop_const.unique()[0]
    # c = 4.69518422

    acc_dis_pred_v1 = df.pipeline_acc_err * c
    predictions = df.cum_remove_accuracy.values + c * df.pipeline_abs_err
    errors = SAPE(predictions,r_df.cum_remove_accuracy.values)
    acc_dis_pred_v2 = SAPE(df.cum_remove_accuracy,predictions)

    sns.lineplot(x=df.num_deletions.values,y=r_df.cum_remove_accuracy.values,label="Retrained acc_del",marker="s")
    # sns.lineplot(x=df.num_deletions.values,y=predictions,label="Predictions",marker="s")
    sns.lineplot(x=df.num_deletions.values,y=errors,label="Errors",marker="s")

    # sns.lineplot(x=df.num_deletions.values,y=acc_dis_pred_v1,label="Acc Dis Predictions_v1",marker="s")
    # sns.lineplot(x=df.num_deletions.values,y=acc_dis_pred_v2,label="Acc Dis Predictions_v2",marker="s")
    # sns.lineplot(x=df.num_deletions.values,y=df.acc_dis,label="True Acc Dis",marker="s")
    # plt.axhline(y=threshold,linestyle="--",color="black",alpha=0.5)
    # sns.lineplot(x=df.num_deletions.values,y=df.pipeline_acc_dis_est,label="Acc Dis Predictions",marker="s")

    plt.ylabel("")
    plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$ Threshold:{threshold}")
    # %%
    fig,((ax11,ax21),(ax12,ax22),(ax13,ax23)) = plt.subplots(3,2,figsize=(10,15))
    noise_level = 0
    sampling_type = "targeted_random"
    df = noise_filter(sampling_type_filter(data.gol_dis_v1,sampling_type),noise_level)
    # df = df.groupby(["num_deletions","threshold","noise","sampling_type"]).mean().reset_index()

    _rolling_avg = "_rolling_avg"
    # _rolling_avg = ""

    df1 = threshold_filter(df,2)
    sns.lineplot(data=df1,x="num_deletions",y=f"acc_dis{_rolling_avg}",label="AccDis",ax=ax11,marker="s")
    sns.lineplot(data=df1,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax11,marker="s")
    ax11.axhline(2,linestyle="--",color="black",alpha=0.5,marker="s")
    ax11.set_title("Cumulative AccDis")


    sns.lineplot(data=df1,x="num_deletions",y=f"checkpoint_acc_dis{_rolling_avg}",label="Checkpoint AccDis",ax=ax21,marker="s")
    sns.lineplot(data=df1,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax21,marker="s")
    ax21.axhline(2,linestyle="--",color="black",alpha=0.5,marker="s")
    ax21.set_title("Checkpoint AccDis")

    ax21.annotate(f"Threshold=2",xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')

    df2 = threshold_filter(df,1)
    sns.lineplot(data=df2,x="num_deletions",y=f"acc_dis{_rolling_avg}",label="AccDis",ax=ax12,marker="s")
    sns.lineplot(data=df2,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax12,marker="s")
    ax12.axhline(1,linestyle="--",color="black",alpha=0.5,marker="s")

    sns.lineplot(data=df2,x="num_deletions",y=f"checkpoint_acc_dis{_rolling_avg}",label="Checkpoint AccDis",ax=ax22,marker="s")
    sns.lineplot(data=df2,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax22,marker="s")
    ax22.axhline(1,linestyle="--",color="black",alpha=0.5,marker="s")

    ax22.annotate(f"Threshold=1",xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')

    df3 = threshold_filter(df,0.5)
    sns.lineplot(data=df3,x="num_deletions",y=f"acc_dis{_rolling_avg}",label="AccDis",ax=ax13,marker="s")
    sns.lineplot(data=df3,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax13,marker="s")
    ax13.axhline(0.5,linestyle="--",color="black",alpha=0.5,marker="s")

    sns.lineplot(data=df3,x="num_deletions",y=f"checkpoint_acc_dis{_rolling_avg}",label="Checkpoint AccDis",ax=ax23,marker="s")
    sns.lineplot(data=df3,x="num_deletions",y=f"pipeline_acc_dis_est{_rolling_avg}",label="Estimated AccDis",ax=ax23,marker="s")
    ax23.axhline(0.5,linestyle="--",color="black",alpha=0.5,marker="s")
    ax23.annotate(f"Threshold=0.5",xy=(1.1,0.5), rotation=-90,ha='center',va='center',xycoords='axes fraction')

    plt.suptitle(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$")
    # plt.savefig(f"{dataset}_{sampling_type}_checkpoint_metric_comparison.pdf",bbox_inches="tight")
    plt.show()

    # %%
    fig,ax = plt.subplots(figsize=(4,1.5))
    plot_acc_dis_helper(data,"targeted_informed",1,1,ax=ax)
