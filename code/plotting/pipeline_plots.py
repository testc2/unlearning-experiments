#%%
from IPython import get_ipython
from numpy.lib.npyio import save
from torch.nn.functional import threshold
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
from plotting.combined_plots import set_size
#%%
def SAPE(a,b):
    numerator = np.abs(a-b)
    denominator = (np.abs(a)+np.abs(b))
    both_zero = np.array((numerator==0)&(denominator==0),ndmin=1)
    sae = np.array(numerator/denominator,ndmin=1)
    sae[both_zero] = 1 
    return sae*100
#%%
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
#%%
dataset = "CIFAR"
ovr_str = "binary"
exp_dir = results_dir/dataset/"when_to_retrain"

temp = []
for file in exp_dir.glob(f"retrain_{ovr_str}*.csv"):
    df = pd.read_csv(file)
    print(file.stem,len(df))
    temp.append(df)

retrain_df = pd.concat(temp)     
# to make the NaN noise and noise seed 0
retrain_df.noise.fillna(0,inplace=True)
retrain_df.noise_seed.fillna(0,inplace=True)
retrain_df = retrain_df.infer_objects()
# retrain_df = retrain_df.groupby(["num_deletions","sampling_type","noise"]).mean().reset_index()
retrain_df["strategy"] = "retrain"


temp = []
for file in exp_dir.glob(f"golatkar_{ovr_str}*.csv"):
    temp.append(pd.read_csv(file))

gol_df = pd.concat(temp)
# to make the NaN noise and noise seed 0
gol_df.noise.fillna(0,inplace=True)
gol_df.noise_seed.fillna(0,inplace=True)
# gol_df = gol_df.groupby(["num_deletions","sampling_type","noise"]).mean().reset_index()
gol_df["strategy"] = "Golatkar"

temp = []
for file in exp_dir.glob(f"nothing_{ovr_str}_*.csv"):
    temp.append(pd.read_csv(file))

nothing_df = pd.concat(temp)
# to make the NaN noise and noise seed 0
nothing_df.noise.fillna(0,inplace=True)
nothing_df.noise_seed.fillna(0,inplace=True)
# nothing_df = nothing_df.groupby(["num_deletions","sampling_type","noise"]).mean().reset_index()
nothing_df["strategy"] = "nothing"

temp = []
for file in exp_dir.glob(f"golatkar_test_thresh_{ovr_str}_*.csv"):
    df = pd.read_csv(file)
    print(file.stem,len(df))
    temp.append(df)

gol_test_df = pd.concat(temp)
# to make the NaN noise and noise seed 0
gol_test_df.noise.fillna(0,inplace=True)
gol_test_df.noise_seed.fillna(0,inplace=True) 
gol_test_df = gol_test_df.infer_objects()
# gol_test_df = gol_test_df.groupby(["num_deletions","sampling_type","noise","threshold"]).mean().reset_index()
gol_test_df["strategy"] = gol_test_df.threshold.apply(lambda t: f"Golatkar Threshold {t} %")
# %%

def compute_metrics(retrain_df,method_df,nothing_df,window_size=10):
    temp = []
    groupby_cols = ["sampling_type","sampler_seed","noise","noise_seed"]
    for (g1,r_df),(g2,m_df),(g3,n_df) in zip(retrain_df.groupby(groupby_cols),method_df.groupby(groupby_cols),nothing_df.groupby(groupby_cols)):
        assert g1 == g2 == g3
        print(g1,len(r_df),len(m_df),len(n_df))
        r_df.set_index("num_deletions",inplace=True)
        m_df["acc_dis"] = SAPE(m_df["cum_remove_accuracy"].values,r_df.loc[m_df.num_deletions,"cum_remove_accuracy"].values)
        m_df["acc_dis_rolling_avg"] = m_df.acc_dis.rolling(window=window_size).mean()
        m_df["acc_dis_batch"] = SAPE(m_df["batch_remove_accuracy"].values,r_df.loc[m_df.num_deletions,"batch_remove_accuracy"].values)
        m_df["acc_dis_batch_rolling_avg"] = m_df.acc_dis_batch.rolling(window=window_size).mean()
        m_df["acc_err"] = SAPE(m_df["test_accuracy"].values,r_df.loc[m_df.num_deletions,"test_accuracy"].values)
        m_df["acc_err_rolling_avg"] = m_df.acc_err.rolling(window=window_size).mean()
        m_df["acc_err_init"] = SAPE(m_df["test_accuracy"].values,n_df.test_accuracy.values[0])
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
        temp.append(m_df)
    return pd.concat(temp)

gol_df = compute_metrics(retrain_df,gol_df,nothing_df)
gol_test_df = pd.concat([compute_metrics(retrain_df,df,nothing_df) for _,df in gol_test_df.groupby("threshold")])
nothing_df = compute_metrics(retrain_df,gol_df,nothing_df)
retrain_df = compute_metrics(retrain_df,retrain_df,nothing_df)
#%%
noise_filter = lambda df,noise: df[df.noise==noise]
threshold_filter = lambda df,threshold : df[df.threshold==threshold]
sampling_type_filter = lambda df,sampling_type : df[df.sampling_type == sampling_type]
# %%
combined = pd.concat([gol_df,gol_test_df,retrain_df])
noise = 1
combined = noise_filter(combined,noise=noise)
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
plt.suptitle(f"{dataset} $\sigma={noise}$")
fig.align_ylabels(ax[:,0])
if save_fig:
    plt.savefig(f"{dataset}_{ovr_str}_pipeline_strategies.pdf",bbox_inches="tight")
else:
    plt.show()
#%%
combined = pd.concat([gol_df,gol_test_df,retrain_df])
sampling_type = "targeted_informed"
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
plt.suptitle(f"{dataset} {' '.join(sampling_type.split('_'))}")
fig.align_ylabels(ax[:,0])
if save_fig:
    plt.savefig(f"{dataset}_{ovr_str}_pipeline_noise_strategies.pdf",bbox_inches="tight")
else:
    plt.show()

#%%
df = noise_filter( threshold_filter(gol_test_df,1), noise= 1)
sns.lineplot(data=df,x="num_deletions",y="cum_running_time",hue="sampling_type")
plt.yscale("log")
#%%
df = noise_filter(sampling_type_filter(gol_test_df,"targeted_informed"),noise=1)
# filter = filter[filter.threshold.isin([1,0.5])]
sns.lineplot(data=df,x="num_deletions",y="pipeline_acc_err",hue="threshold")
plt.show()

sns.lineplot(data=df,x="num_deletions",y="acc_err",hue="threshold")
# plt.yscale("log")
plt.show()
# %%
df = pd.concat([gol_test_df,retrain_df])
df = noise_filter(sampling_type_filter(df,"targeted_informed"),noise=1)
# filter = filter[filter.threshold.isin([1,0.5])]
sns.lineplot(data=df,x="num_deletions",y="cum_remove_accuracy",hue="strategy")
plt.show()

# %%
df = noise_filter(sampling_type_filter(gol_test_df,"targeted_informed"),0)
# df = sampling_type_filter(retrain_df,"targeted_informed")
sns.lineplot(data=df,x="num_deletions",y="acc_dis_cumsum",hue="threshold",style="threshold")
# %%
df = sampling_type_filter(threshold_filter(gol_test_df,1),"targeted_informed")
sns.lineplot(data=df,x="num_deletions",y="acc_dis_cumsum",hue="noise")


# %%
