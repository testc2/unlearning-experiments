#%%
from IPython import get_ipython
from pandas.core.frame import DataFrame
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False

from typing import Optional
from pathlib import Path
import sys
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str((project_dir/"code").resolve()))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from parse import get_dg_unlearn_frames
from plotting.ratio_remove_plots import load_dfs as ratio_load_dfs

#%%
save_fig = False
data_dir = project_dir/"data"
results_dir = data_dir/"results"
# %%
def KL_divergence(mu1,sigma1,mu2,sigma2):
    eps=1e-18
    return np.log(sigma2/(sigma1+eps))+((sigma1**2+(mu1-mu2)**2)/(2*sigma2**2+eps))-0.5

def JS(mu1,sigma1,mu2,sigma2):
    mid_mu = (mu1+mu2)/2
    mid_sigma = np.sqrt((sigma1**2+sigma2**2)/2)
    kl_p_m = KL_divergence(mu1,sigma1,mid_mu,mid_sigma)
    kl_q_m= KL_divergence(mu2,sigma2,mid_mu,mid_sigma)
    print(kl_p_m,kl_q_m)
    return 0.5*kl_p_m + 0.5*kl_q_m


def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = (numerator==0)&(denominator==0)
    sae = numerator/denominator
    if sae.size > 1:
        sae[both_zero] = 1 
    elif both_zero:
        sae = np.array(1)
    return sae*100

def error(baseline,method):
    # return (method-baseline)/np.abs(baseline)*100
    numerator = method-baseline
    denominator = np.abs(method)+np.abs(baseline)
    both_zero = (numerator==0)&(denominator==0)
    se = numerator/denominator
    if se.size > 1:
        se[both_zero] = 1 
    elif both_zero:
        se = np.array(1)
    return se*100
#%%

qoa_transform = lambda df,qoa_column,qoa: df[df[qoa_column]==qoa].copy()

def load_dfs(results_dir:Path,dataset:str,ovr_str:str,plot_deltagrad:bool=None,ratio_index:int=2):

    df = pd.read_csv(results_dir/dataset/f"Unlearn_{ovr_str}.csv")
    df_guo = df[df.method=="Guo"].apply(pd.to_numeric,errors="ignore")
    df_gol = df[df.method=="Golatkar"].apply(pd.to_numeric,errors="ignore")
    df_baseline = df[df.method=="baseline"].apply(pd.to_numeric,errors="ignore")
    df_baseline_guo = df[df.method=="baseline_Guo"].apply(pd.to_numeric,errors="ignore")
    df_baseline_gol = df[df.method=="baseline_Golatkar"].apply(pd.to_numeric,errors="ignore")

    if plot_deltagrad:
        df_deltagrad, df_baseline_deltagrad = get_dg_unlearn_frames(results_dir,dataset,f"Deltagrad_unlearn_{ovr_str}")
        df_baseline_deltagrad["method"] = "baseline_deltagrad"
    
    ratio = sorted(df_guo.remove_ratio.unique())[ratio_index]
    transform = lambda df,ratio: df[df.remove_ratio==ratio]
    guo = transform(df_guo,ratio)
    gol = transform(df_gol,ratio)
    gol = transform(df_gol,ratio)
    baseline_guo = transform(df_baseline_guo,ratio)
    baseline_gol = transform(df_baseline_gol,ratio)
    baseline = transform(df_baseline,ratio)
    if plot_deltagrad:
        deltagrad = transform(df_deltagrad,ratio)
        baseline_deltagrad = transform(df_baseline_deltagrad,ratio)
    else:
        deltagrad = None
        baseline_deltagrad = None

    return dict(
        ratio=ratio,
        baseline_guo=baseline_guo,
        baseline_gol=baseline_gol,
        baseline_deltagrad=baseline_deltagrad,
        baseline = baseline,
        guo=guo,
        gol=gol,
        deltagrad=deltagrad,


    )

def plot_default(
    y:str,
    method_name:str,
    method:pd.DataFrame,
    baseline:pd.DataFrame,
    QoA_column:str,
    QoA:float,
    ax:Optional[plt.Axes]=None
):

    method = qoa_transform(method,QoA_column,QoA)
    if method_name == "Golatkar":
        baseline = qoa_transform(baseline,QoA_column,QoA)
    
    palette = sns.color_palette("husl",n_colors=method[QoA_column].nunique())
    if ax is None:
        fix,ax = plt.subplots()
    
    sns.lineplot(data=method,x="noise",y=y,palette=palette,err_style="band",linestyle="dashed",legend=True,ax=ax,label=f"QoA={QoA}")
    sns.lineplot(data=baseline,x="noise",y=y,label="Baseline Perturbed",err_style="band",linestyle="dashed",ax=ax)
    ax.set_xscale("log")
    ax.legend(bbox_to_anchor=(0.5,-0.01),loc="upper center",bbox_transform=plt.gcf().transFigure)
    return ax

#%%
def plot_noise_minibatch(y:str,method_name:str,df:pd.DataFrame,baseline:pd.DataFrame,metric:str):
    markers = ["o","s","v","D"]
    palette = sns.color_palette("husl",n_colors=guo.noise.nunique())
    fig,ax = plt.subplots()
    df_noise_gb = df.groupby("noise")
    baseline_noise_gb = baseline.groupby("noise")
    for i,((noise,df_noise_df),(noise,baseline_noise_df)) in enumerate(zip(df_noise_gb,baseline_noise_gb)):
        xmeans = []
        ymeans = []
        for j,((minibatch_fraction,m_df),marker) in enumerate(zip(df_noise_df.groupby("minibatch_fraction"),markers)):
            x_values = baseline_noise_df.removal_time.values/m_df.removal_time.values
            a = baseline_noise_df[y].values
            b = m_df[y].values
            # if minibatch_fraction == 1:
                # print(noise,minibatch_fraction,b,b.mean(),a, a.mean())
            # y_values = JS(b.mean(),b.std(),a.mean(),a.std())
            # print(y_values)
            y_values = SAPE(b.mean(),a.mean())
            if i==0:
                label=f"$m^\prime=m/{minibatch_fraction}$"
            else :
                label=""
            ax.errorbar(
                x=x_values.mean(),
                y=y_values.mean(),
                xerr=x_values.std(),
                yerr=y_values.std(),
                marker=marker,
                label=label,color=palette[i]
            )
            xmeans.append(x_values.mean())
            ymeans.append(y_values.mean())
        ax.plot(xmeans,ymeans,color=palette[i],label=f"$\sigma={noise}$",ls="--")
    ax.legend()
    # ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    ax.set_xlabel("Speedup")
    y_label=" ".join(y.split("_")).title()
    # ax.set_ylabel(f"KL of {y_label}")
    ax.set_ylabel(f"sAPE of {y_label}")
    # ax.set_yscale("log")
    ax.set_xscale("log")
#%%
def plot_minibatch_noise(y:str,method_name:str,df:pd.DataFrame,baseline:pd.DataFrame,metric:str):

    markers = ["o","s","v","D"]
    palette = sns.color_palette("husl",n_colors=guo.noise.nunique())
    fig,ax = plt.subplots()
    df_mb_gb = df.groupby("minibatch_fraction")
    for i,(minibatch_fraction,m_df) in enumerate(df_mb_gb):
        xmeans = []
        ymeans = []
        df_noise_gb = m_df.groupby("noise")
        baseline_noise_gb = baseline.groupby("noise")
        for j,((noise,noise_df),(noise,baseline_noise_df),marker) in enumerate(zip(df_noise_gb,baseline_noise_gb,markers)):
            x_values = baseline_noise_df.removal_time.values/noise_df.removal_time.values
            a = baseline_noise_df[y].values
            b = noise_df[y].values
            # if minibatch_fraction == 1:
            #     print(noise,minibatch_fraction,b,b.mean(),a, a.mean())
            
            # print(noise,minibatch_fraction,b.mean(),b.std(),a.mean(),a.std())
            y_values = JS(b.mean(),b.std(),a.mean(),a.std())
            # print(y_values)
            # y_values = SMAPE(b.mean(),a.mean())
            if i==0:
                label=f"$\sigma={noise}$"
            else :
                label=""
            ax.errorbar(
                x=x_values.mean(),
                y=y_values.mean(),
                xerr=x_values.std(),
                yerr=y_values.std(),
                marker=marker,
                label=label,color=palette[i]
            )
            xmeans.append(x_values.mean())
            ymeans.append(y_values.mean())
        ax.plot(xmeans,ymeans,color=palette[i],label=f"$m^\prime=m/{minibatch_fraction}$",ls="--")
    ax.legend()
    # ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    ax.set_xlabel("Speedup")
    y_label=" ".join(y.split("_")).title()
    # ax.set_ylabel(f"KL of {y_label}")
    ax.set_ylabel(f"SMAPE of {y_label}")
    # ax.set_yscale("log")
    ax.set_xscale("log")
#%%

def func(df,method,y):
    a = df[df.method==f"baseline_{method}"][y].values
    a = np.r_[a,a.mean()]
    b = df[df.method==f"{method}"][y].values
    b = np.r_[b,b.mean()]
    return pd.Series({y:SAPE(*np.array(np.meshgrid(a,b)).reshape(2,-1)).min()})


def compute_sape(
    y:str,
    baseline:pd.DataFrame,
    baseline_method:pd.DataFrame,
    method:pd.DataFrame,
    method_name:str,
    ):

    # Alternate SAPE calculation
    # # print(method.groupby("noise")[y].mean(),baseline_method.groupby("noise")[y].mean())
    # y_smape2 = SAPE(baseline_method.groupby("noise")[y].mean(),method.groupby("noise")[y].mean()
    # ).reset_index()

    # compute SAPE as the smallest possible difference 
    concat = pd.concat((method,baseline_method))
    y_smape2 = concat.groupby("noise").apply(func,method=method_name,y=y)
    y_smape2 = y_smape2.explode(y).reset_index().apply(pd.to_numeric,errors="ignore")
    temp = method.copy()
    # for effectiveness compute SAPE between the method and the baseline with no noise
    # print(baseline.test_accuracy.mean(),temp.test_accuracy)
    temp["test_sape"] = SAPE(baseline.test_accuracy.mean(),temp.test_accuracy)
    
    return y_smape2, temp


def plot_unlearn(
    y:str,
    method_name:str,
    baseline:pd.DataFrame,
    baseline_method:pd.DataFrame,
    method:pd.DataFrame,
    QoA_column:str,
    QoA:int=1,
    ax1:Optional[plt.Axes]=None,
    ax2:Optional[plt.Axes]=None,
    latex:bool=True,
    verbose:bool=False,
    *args,
    **kwargs
):
    """A function that plots the trade-offs between certifiably and effectiveness

    Args:
        y (str): the metric to use for computing certifiability (typically remove_accuracy)
        method_name (str): Guo,Golatkar or deltagrad
        baseline (pd.DataFrame): the fully-retrained model at no noise injection
        baseline_method (pd.DataFrame): the baseline corresponding to the particular method
        method (pd.DataFrame): the results for the method
        QoA_column (str): the column that corresponds to the QoA parameter of the method (minibatch_fraction for Guo/ Golatkar or period for Deltagrad)
        QoA (int, optional): the value to select for QoA. Defaults to 1.
        ax1 (Optional[plt.Axes], optional): The axis for certifiability. Defaults to None.
        ax2 (Optional[plt.Axes], optional): The axis for effectiveness. Defaults to None.

    Returns:
        the axis
    """
    if ax1 is None:
        fig,ax1 = plt.subplots()
    if ax2 is None:
        ax2 = ax1.twinx()
    assert method_name in ["Guo","Golatkar","deltagrad"]
    method = qoa_transform(method,QoA_column,QoA)
    if method_name == "Golatkar":
        baseline_method = qoa_transform(baseline_method,QoA_column,QoA)

    if latex:
        twin_ylabel = r"$\text{Rel.}\alpha$"
    else:
        twin_ylabel = r"$Rel.\alpha$"

    eps_sape,alpha_sape = compute_sape(y,baseline,baseline_method,method,method_name)
    ax = sns.lineplot(data=eps_sape,x="noise",y=y,ax=ax1,marker="o",label="$\epsilon$",color="tab:red",*args,**kwargs)
    ax2 = sns.lineplot(data=alpha_sape,x="noise",y="test_sape",marker="s",linestyle="dashed",label=twin_ylabel,err_style=None,ax=ax2,color="black",*args,**kwargs)
    ax.set_xscale("log")
    if eps_sape[y].mean()==0:
        ax.set_ylim(bottom=-0.005)
    ax.set_xlabel("Noise Parameter $\sigma$")
    ax.set_ylabel(r"sAPE $Acc_{del}$ ($\epsilon$)")
    ax2.set_ylabel(r"sAPE $Acc_{test}$ ($\alpha$)")
    if verbose:
        print("EPS SAPE ",eps_sape.groupby("noise")[y].mean())
        print("ALPHA SAPE ",alpha_sape.groupby("noise").test_sape.mean())
    return ax1,ax2


def plot_unlearn_tradeoffs(
    y:str,
    baseline:pd.DataFrame,
    baseline_guo:pd.DataFrame,
    baseline_gol:pd.DataFrame,
    guo:pd.DataFrame,
    gol:pd.DataFrame,
    QoA_column:str,
    QoA:int=1,
):

    fig,ax = plt.subplots()
    qoa_transform = lambda df,qoa_column,qoa: df[df[qoa_column]==qoa].copy()
    guo = qoa_transform(guo,QoA_column,QoA)
    gol = qoa_transform(gol,QoA_column,QoA)
    baseline_gol = qoa_transform(baseline_gol,QoA_column,QoA)

    guo_eps_sape,guo_alpha_sape = compute_sape(y,baseline,baseline_guo,guo,"Guo")
    gol_eps_sape,gol_alpha_sape = compute_sape(y,baseline,baseline_gol,gol,"Golatkar")
    
    ax = sns.lineplot(data=guo_eps_sape,x="noise",y=y,ax=ax,marker="o",label="Guo",color="tab:blue")
    ax = sns.lineplot(data=gol_eps_sape,x="noise",y=y,ax=ax,marker="s",label="Golatkar",color="tab:orange")
    ax2 = ax.twinx()
    ax2 = sns.lineplot(data=guo_alpha_sape,x="noise",y="test_sape",marker="o",linestyle="dashed",label="Guo",err_style=None,ax=ax2,color="tab:blue")
    ax2 = sns.lineplot(data=gol_alpha_sape,x="noise",y="test_sape",marker="s",linestyle="dashed",label="Golatkar",err_style=None,ax=ax2,color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("Noise Parameter $\sigma$")
    ax.set_ylabel(r"sAPE $Acc_{removed}$ ($\epsilon$)")
    ax.set_ylabel(r"sAPE $Acc_{test}$ ($\alpha$)")

    plt.show()

# %%

def plot_unlearn_certifiability(
    y:str,
    dfs_dict:dict,
    noise:float,
    ax:Optional[plt.Axes]=None,
    legend:bool=False,
    verbose:bool=False,
    **kwargs
):

    noise_transform = lambda df,noise: df[df.noise==noise]
    guo = noise_transform(dfs_dict["guo"],noise)
    gol = noise_transform(dfs_dict["gol"],noise)
    deltagrad = dfs_dict["deltagrad"][dfs_dict["deltagrad"].period.isin([2,5,50,100])]
    deltagrad = noise_transform(deltagrad,noise)
    baseline_guo = noise_transform(dfs_dict["baseline_guo"],noise)
    baseline_gol = noise_transform(dfs_dict["baseline_gol"],noise)
    baseline_deltagrad = noise_transform(dfs_dict["baseline_deltagrad"],noise)
    markers = ["o","s","v","D"]#,"X","P","*",">"]
    xmeans = []
    ymeans = []
    if ax is None:
        fig,ax = plt.subplots()
    for (minibatch_fraction,df),marker in zip(guo.groupby("minibatch_fraction"),markers):
        x_values = baseline_guo.removal_time.dropna().values/df.removal_time.values
        a = baseline_guo[y].values
        b = df[y].values 
        smape = (np.abs(a-b)/(a+b))*100
        y_values = smape
        if verbose:
            print(f"Guo Minibatch Fraction: {minibatch_fraction} X speed-up: {x_values.mean()} Y SPAE: {y_values.mean()}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:blue"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:blue",label="${INF}$",ls="--")        
    ymeans=[]
    xmeans=[]
    for (minibatch_fraction,df),(minibatch_fraction,baseline_df),marker in zip(gol.groupby("minibatch_fraction"),baseline_gol.groupby("minibatch_fraction"),markers):
        x_values = baseline_df.removal_time.dropna().values/df.removal_time.values
        a = baseline_df[y].values.mean()
        b = df[y].values.mean()
        smape = (np.abs(a-b)/(a+b))*100
        y_values = smape
        if verbose:
            print(f"Golatkar Minibatch Fraction: {minibatch_fraction}Baseline x: {x_values}")
            print(f"Golatkar Minibatch Fraction: {minibatch_fraction}\n y: baseline :{a} gol: {b} sape: {y_values}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:orange"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:orange",label="${FISH}$",ls="--")        
    ymeans=[]
    xmeans=[]
    for (period,df),marker in zip(deltagrad.groupby("period"),markers):
        x_values = baseline_deltagrad.removal_time.dropna().values/df.removal_time.values
        a = baseline_deltagrad[y].values.mean()
        b = df[y].values.mean()
        smape = (np.abs(a-b)/(a+b))*100
        y_values = smape
        if verbose :
            print("Baseline x: ",x_values)
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$T_0={period}$",color="tab:green"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:green",label="${DG}$",ls="--")        
    ymeans=[]
    xmeans=[]
    ax.set_xscale("log",base=3)
    ax.set_yscale("symlog",linthresh=0.1)
    ax.set_ylim(-0.1,1e2)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    if legend:
        ax.legend(bbox_to_anchor=(1.05,0.5),loc="center left")
    
    return ax

#%%
def plot_unlearn_effectiveness(
    y:str,
    dfs_dict:dict,
    noise:float,
    ax:Optional[plt.Axes]=None,
    legend:bool=False,
    verbose:bool=False,
    **kwargs
):

    noise_transform = lambda df,noise: df[df.noise==noise]
    guo = noise_transform(dfs_dict["guo"],noise)
    gol = noise_transform(dfs_dict["gol"],noise)
    deltagrad = dfs_dict["deltagrad"][dfs_dict["deltagrad"].period.isin([2,5,50,100])]
    deltagrad = noise_transform(deltagrad,noise)
    baseline_guo = noise_transform(dfs_dict["baseline_guo"],noise)
    baseline_gol = noise_transform(dfs_dict["baseline_gol"],noise)
    baseline_deltagrad = noise_transform(dfs_dict["baseline_deltagrad"],noise)
    # baseline when no noise 
    baseline = dfs_dict["baseline"]
    markers = ["o","s","v","D"]#,"X","P","*",">"]
    xmeans = []
    ymeans = []
    if ax is None:
        fig,ax = plt.subplots()
    for (minibatch_fraction,df),marker in zip(guo.groupby("minibatch_fraction"),markers):
        # x_values = baseline_guo.removal_time.dropna().values/df.removal_time.values
        x_values = np.array(baseline.removal_time.mean()/df.removal_time.mean())
        a = baseline[y].mean()
        b = df[y].values.mean()
        y_values = SAPE(a,b)
        if verbose:
            print(f"Guo Minibatch Fraction: {minibatch_fraction}Baseline x: {x_values}")
            print(f"Guo Minibatch Fraction: {minibatch_fraction} \n y: baseline :{a} guo: {b} sape: {y_values}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:blue"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:blue",label="${INF}$",ls="--")        
    ymeans=[]
    xmeans=[]
    for (minibatch_fraction,df),(minibatch_fraction,baseline_df),marker in zip(gol.groupby("minibatch_fraction"),baseline_gol.groupby("minibatch_fraction"),markers):
        # x_values = baseline_df.removal_time.dropna().values/df.removal_time.values
        x_values = np.array(baseline.removal_time.mean()/df.removal_time.mean())
        a = baseline[y].values.mean()
        b = df[y].values.mean()
        y_values = SAPE(a,b)
        if verbose:
            print(f"Golatkar Minibatch Fraction: {minibatch_fraction} x: {x_values}")
            print(f"Golatkar Minibatch Fraction: {minibatch_fraction}\n y: baseline :{a} gol: {b} sape: {y_values}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$m^\prime=m/{minibatch_fraction}$",color="tab:orange"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:orange",label="${FISH}$",ls="--")        
    ymeans=[]
    xmeans=[]
    for (period,df),marker in zip(deltagrad.groupby("period"),markers):
        # for deltagrad fall back to the baseline running from from the script 
        x_values = baseline_deltagrad.removal_time.dropna().values/df.removal_time.values
        a = baseline[y].values.mean()
        b = df[y].values.mean()
        # print("Baseline x: ",x_values)
        y_values = SAPE(a,b)
        if verbose:
            print(f"Deltagrad period: {period} y: baseline :{a} deltagrad: {b} SAPE: {y_values.mean()}")
        ax.errorbar(
            x=x_values.mean(),
            y=y_values.mean(),
            xerr=x_values.std(),
            yerr=y_values.std(),
            marker=marker,
            **kwargs,
            label=f"$T_0={period}$",color="tab:green"
        )
        xmeans.append(x_values.mean())
        ymeans.append(y_values.mean())
    ax.plot(xmeans,ymeans,color="tab:green",label="${DG}$",ls="--")        
    ymeans=[]
    xmeans=[]
    ax.set_xscale("log",base=3)
    ax.set_yscale("symlog",linthresh=0.1)
    ax.set_ylim(-0.1,1e2)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=1, color='grey', linestyle='--')
    if legend:
        ax.legend(bbox_to_anchor=(1.05,0.5),loc="center left")
    
    return ax

#%%
if __name__ == "__main__":
    mpl.rcParams["figure.dpi"]=100
    mpl.rcParams["font.size"]=20
    dataset = "EPSILON"
    dfs_dict = load_dfs(results_dir,dataset,"binary",plot_deltagrad=True,ratio_index=2)
    plot_unlearn_certifiability(
        "remove_accuracy",
        dfs_dict,
        noise=1,
        legend=True,
        verbose=True,
)
    plot_unlearn_effectiveness(
            "test_accuracy",
            dfs_dict,
            noise=1,
            legend=True,
            verbose=True,
    )
#%%
    ax = plot_default(
        "test_accuracy",
        "Guo",
        dfs_dict["guo"],
        dfs_dict["baseline_guo"],
        QoA_column="minibatch_fraction",
        QoA=1,
    )
    ax.set_title(f"Guo ratio={dfs_dict['ratio']}")
    ax = plot_default(
        "remove_accuracy",
        "Golatkar",
        dfs_dict["gol"],
        dfs_dict["baseline_gol"],
        QoA_column="minibatch_fraction",
        QoA=1
    )
    ax.set_title(f"Golatkar ratio={dfs_dict['ratio']}")
    ax = plot_default(
        "remove_accuracy",
        "deltagrad",
        dfs_dict["deltagrad"],
        dfs_dict["baseline_deltagrad"],
        QoA_column="period",
        QoA=50
    )
    ax.set_title(f"Deltagrad ratio={dfs_dict['ratio']}")
    ax1,ax2=plot_unlearn(
        "remove_accuracy",
        "Guo",
        dfs_dict["baseline"],
        dfs_dict["baseline_guo"],
        dfs_dict["guo"],
        QoA_column="minibatch_fraction",
        QoA=8,
        latex=False,
        verbose=True

    )
    ax1.set_title(f"Guo ratio={dfs_dict['ratio']}")

    ax1,ax2=plot_unlearn(
        "remove_accuracy",
        "Golatkar",
        dfs_dict["baseline"],
        dfs_dict["baseline_gol"],
        dfs_dict["gol"],
        QoA_column="minibatch_fraction",
        QoA=1,
        latex=False,
        verbose=True
    )
    ax1.set_title(f"Golatkar ratio={dfs_dict['ratio']}")
    ax1,ax2=plot_unlearn(
        "remove_accuracy",
        "deltagrad",
        dfs_dict["baseline"],
        dfs_dict["baseline_deltagrad"],
        dfs_dict["deltagrad"],
        QoA_column="period",
        QoA=100,
        latex=False,
        verbose=True
    )
    ax1.set_title(f"Deltagrad ratio={dfs_dict['ratio']}")


#%%
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    guo_temp = qoa_transform(dfs_dict["guo"],"minibatch_fraction",1)
    concat = pd.concat([guo_temp,dfs_dict["baseline_guo"]])
    sns.boxplot(data=concat,x="noise",y="remove_accuracy",hue="minibatch_fraction",ax=ax1)
    ax1.get_legend().remove()
    sns.boxplot(data=concat,x="noise",y="test_accuracy",hue="minibatch_fraction",ax=ax2)
    handles,labels = ax2.get_legend_handles_labels()
    labels[:-1] = list(map(lambda s: f"$m^\prime=m/{s}$",labels[:-1],))
    labels[-1]="Baseline"
    ax1.set_xlabel("$\sigma$ noise")
    ax2.set_xlabel("$\sigma$ noise")
    plt.legend(handles=handles,labels=labels,bbox_to_anchor=(0.5,-0.01),loc="upper center",bbox_transform=fig.transFigure,ncol=5)
    plt.suptitle(f"Dataset: {dataset} Removal Step: Guo")
    fig.subplots_adjust(wspace=0.3)
    # plt.savefig(results_dir/"images"/"boxplot test.pdf",bbox_inches="tight",dpi=300)
#%%
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    gol_temp = qoa_transform(dfs_dict["gol"],"minibatch_fraction",1)
    baseline_gol_temp = qoa_transform(dfs_dict["baseline_gol"],"minibatch_fraction",1)
    baseline_gol_temp["minibatch_fraction"]="baseline"
    concat = pd.concat([gol_temp,baseline_gol_temp])

    sns.boxplot(data=concat,x="noise",y="remove_accuracy",hue="minibatch_fraction",ax=ax1)
    ax1.get_legend().remove()
    sns.boxplot(data=concat,x="noise",y="test_accuracy",hue="minibatch_fraction",ax=ax2)
    handles,labels = ax2.get_legend_handles_labels()
    labels[:-1] = list(map(lambda s: f"$m^\prime=m/{s}$",labels[:-1],))
    labels[-1]="Baseline"
    ax1.set_xlabel("$\sigma$ noise")
    ax2.set_xlabel("$\sigma$ noise")
    plt.legend(handles=handles,labels=labels,bbox_to_anchor=(0.5,-0.01),loc="upper center",bbox_transform=fig.transFigure,ncol=5)
    plt.suptitle(f"Dataset: {dataset} Removal Step: Golatkar")
    fig.subplots_adjust(wspace=0.3)
    # plt.savefig(results_dir/"images"/"boxplot test golatkar.pdf",bbox_inches="tight",dpi=300)
# %%
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    guo_temp = qoa_transform(dfs_dict["guo"],"minibatch_fraction",1)
    concat = pd.concat([guo_temp,dfs_dict["baseline_guo"]])
    sns.pointplot(data=concat,x="noise",y="remove_accuracy",hue="minibatch_fraction",ax=ax1,join=False,dodge=0.5)
    ax1.get_legend().remove()
    sns.pointplot(data=concat,x="noise",y="test_accuracy",hue="minibatch_fraction",ax=ax2,join=False,dodge=0.5)
    handles,labels = ax2.get_legend_handles_labels()
    labels[:-1] = list(map(lambda s: f"$m^\prime=m/{s}$",labels[:-1],))
    labels[-1]="Baseline"
    ax1.set_xlabel("$\sigma$ noise")
    ax2.set_xlabel("$\sigma$ noise")
    plt.legend(handles=handles,labels=labels,bbox_to_anchor=(0.5,-0.01),loc="upper center",bbox_transform=fig.transFigure,ncol=5)
    plt.suptitle(f"Dataset: {dataset} Removal Step: Guo")
    fig.subplots_adjust(wspace=0.3)
    # plt.savefig(results_dir/"images"/"pointplot test.pdf",bbox_inches="tight",dpi=300)
#%%
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    gol_temp = qoa_transform(dfs_dict["gol"],"minibatch_fraction",1)
    baseline_gol_temp = qoa_transform(dfs_dict["baseline_gol"],"minibatch_fraction",1)
    baseline_gol_temp["minibatch_fraction"]="baseline"
    concat = pd.concat([gol_temp,baseline_gol_temp])

    sns.pointplot(data=concat,x="noise",y="remove_accuracy",hue="minibatch_fraction",ax=ax1,join=False,dodge=0.5)
    ax1.get_legend().remove()
    sns.pointplot(data=concat,x="noise",y="test_accuracy",hue="minibatch_fraction",ax=ax2,join=False,dodge=0.5)
    handles,labels = ax2.get_legend_handles_labels()
    labels[:-1] = list(map(lambda s: f"$m^\prime=m/{s}$",labels[:-1],))
    labels[-1]="Baseline"
    ax1.set_xlabel("$\sigma$ noise")
    ax2.set_xlabel("$\sigma$ noise")
    plt.legend(handles=handles,labels=labels,bbox_to_anchor=(0.5,-0.01),loc="upper center",bbox_transform=fig.transFigure,ncol=5)
    plt.suptitle(f"Dataset: {dataset} Removal Step: Golatkar")
    fig.subplots_adjust(wspace=0.3)
    # plt.savefig(results_dir/"images"/"pointplot test golatkarpdf",bbox_inches="tight",dpi=300)
#%%