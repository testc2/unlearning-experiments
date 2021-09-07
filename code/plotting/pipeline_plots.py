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
from methods.common_utils import SAPE
from plotting.ratio_remove_plots import load_dfs as ratio_load_dfs
from scipy.optimize import curve_fit
from scipy.stats import pearsonr,spearmanr
import json
from plotting.combined_plots import set_size
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
dataset = "COVTYPE"
ovr_str = "binary"
exp_dir = results_dir/dataset/"when_to_retrain"

temp = []
for file in exp_dir.glob(f"retrain_{ovr_str}*.csv"):
    df = pd.read_csv(file)
    # print(file.stem,len(df))
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
gol_df = gol_df.infer_objects()
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
    # print(file.stem,len(df))
    temp.append(df)

gol_test_df = pd.concat(temp)
# to make the NaN noise and noise seed 0
gol_test_df.noise.fillna(0,inplace=True)
gol_test_df.noise_seed.fillna(0,inplace=True) 
gol_test_df = gol_test_df.infer_objects()
# gol_test_df = gol_test_df.groupby(["num_deletions","sampling_type","noise","threshold"]).mean().reset_index()
gol_test_df["strategy"] = gol_test_df.threshold.apply(lambda t: f"Golatkar Threshold {t} %")

temp = []
for file in exp_dir.glob(f"golatkar_disparity_thresh_{ovr_str}*.csv"):
    df = pd.read_csv(file)
    # print(file.st`em,len(df))
    temp.append(df)
if len(temp):
    gol_dis_df = pd.concat(temp)
    # to make the NaN noise and noise seed 0
    gol_dis_df.noise.fillna(0,inplace=True)
    gol_dis_df.noise_seed.fillna(0,inplace=True) 
    gol_dis_df = gol_dis_df.infer_objects()
    # gol_dis_df = gol_dis_df.groupby(["num_deletions","sampling_type","noise","threshold"]).mean().reset_index()
    gol_dis_df["strategy"] = gol_dis_df.threshold.apply(lambda t: f"Golatkar Dis Threshold {t} %")
# %%
noise_filter = lambda df,noise: df[df.noise==noise]
threshold_filter = lambda df,threshold : df[df.threshold==threshold]
sampling_type_filter = lambda df,sampling_type : df[df.sampling_type == sampling_type]
#%%
def compute_metrics(retrain_df,method_df,nothing_df,threshold=None,window_size=10):
    temp = []
    groupby_cols = ["sampling_type","noise"]
    for (sampling_type,noise),df in retrain_df.groupby(groupby_cols):
        method_filter_df = noise_filter(sampling_type_filter(method_df,sampling_type),noise) 
        nothing_filter_df = noise_filter(sampling_type_filter(nothing_df,sampling_type),noise) 
        groupby_cols_1 = ["sampler_seed","noise_seed"]
        if len(method_filter_df):
            # print(sampling_type,noise)
            for ((g1,r_df),(g2,m_df),(g3,n_df)) in zip(df.groupby(groupby_cols_1),method_filter_df.groupby(groupby_cols_1),nothing_filter_df.groupby(groupby_cols_1)):
                assert g1 == g2 == g3
                # print(g1,len(r_df),len(m_df),len(n_df))
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
                if threshold is not None :
                    # find where the acc_err exceeded the threshold
                    # where acc_err was lower than threshold error is considered 0
                    m_df["error_acc_err"] = np.maximum(m_df.acc_err-threshold,0) 
                    m_df["error_acc_err_cumsum"] = m_df.error_acc_err.cumsum()
                    # similarly check for acc_dis
                    m_df["error_acc_dis"] = np.maximum(m_df.acc_dis-threshold,0) 
                    m_df["error_acc_dis_cumsum"] = m_df.error_acc_dis.cumsum()
                temp.append(m_df)
    return pd.concat(temp)

#%%
gol_df = compute_metrics(retrain_df,gol_df,nothing_df)
gol_test_df = pd.concat([compute_metrics(retrain_df,df,nothing_df,threshold=t) for t,df in gol_test_df.groupby("threshold")])
gol_dis_df = pd.concat([compute_metrics(retrain_df,df,nothing_df,threshold=t) for t,df in gol_dis_df.groupby("threshold")])
nothing_df = compute_metrics(retrain_df,gol_df,nothing_df)
retrain_df = compute_metrics(retrain_df,retrain_df,nothing_df)
# %%
combined = pd.concat([gol_df,gol_test_df,retrain_df])
noise = 0
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
# plots to compare impact of threshold on a metric for particular noise level and sampling type 
sampling_type = "uniform_informed"
noise_level = 0
metric = "error_acc_dis_cumsum"
dataframe = gol_dis_df
df = noise_filter(sampling_type_filter(dataframe,sampling_type),noise_level)
sns.lineplot(data=df,x="num_deletions",y=metric,hue="threshold",style="threshold")
plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$")

plt.show()
#%%

# plots to compare the impact of noise on AccDis and AccErr for a threshold and sampling type 
threshold = 0.1
sampling_type = "targeted_informed"
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
df = sampling_type_filter(threshold_filter(gol_test_df,threshold),sampling_type)
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
df = sampling_type_filter(noise_filter(gol_df,noise_level),sampling_type)
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


sampling_type = "targeted_informed"
noise_level = 0
threshold = 1
metric = "acc_dis"
df = threshold_filter(noise_filter(sampling_type_filter(gol_test_df,sampling_type),noise_level),threshold)
sns.lineplot(data=df,x="num_deletions",y=metric)
plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$")

plt.show()
# %%

sampling_type = "uniform_informed"
noise_level = 0
threshold = 1
m_df = noise_filter(sampling_type_filter(gol_dis_df,sampling_type),noise_level)
r_df = noise_filter(sampling_type_filter(retrain_df,sampling_type),noise_level)
n_df = noise_filter(sampling_type_filter(nothing_df,sampling_type),noise_level)
g_df = noise_filter(sampling_type_filter(gol_df,sampling_type),noise_level)
df = pd.concat([compute_metrics(r_df,df,n_df) for _,df in m_df.groupby("threshold")])
df = threshold_filter(df,threshold)
c = df.prop_const.unique()[0]
sns.lineplot(data=df,x="num_deletions",y="pipeline_acc_dis_est",label="pipeline estimate",marker="s")
sns.lineplot(data=df,x="num_deletions",y="acc_dis",label="true",marker="s")
plt.axhline(y=threshold,linestyle="--",color="black",alpha=0.5)
# sns.lineplot(data=df,x="num_deletions",y="acc_dis_cumsum",hue="threshold")
plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$ Threshold:{threshold}")
# %%
sampling_type = "targeted_informed"
noise_level = 0
threshold = 1
m_df = noise_filter(sampling_type_filter(gol_dis_df,sampling_type),noise_level)
r_df = noise_filter(sampling_type_filter(retrain_df,sampling_type),noise_level)
n_df = noise_filter(sampling_type_filter(nothing_df,sampling_type),noise_level)
g_df = noise_filter(sampling_type_filter(gol_df,sampling_type),noise_level)
df = pd.concat([compute_metrics(r_df,df,n_df) for _,df in m_df.groupby("threshold")])
df = threshold_filter(df,threshold)
c = df.prop_const.unique()[0]
c = 4.69518422

acc_dis_pred = df.pipeline_acc_err * c
# predictions = df.cum_remove_accuracy.values + c * df.pipeline_abs_err
# errors = SAPE(predictions,r_df.cum_remove_accuracy.values)
# acc_dis_pred = SAPE(df.cum_remove_accuracy,predictions)

# sns.lineplot(x=df.num_deletions.values,y=r_df.cum_remove_accuracy.values,label="Retrained acc_del",marker="s")
# sns.lineplot(x=df.num_deletions.values,y=predictions,label="Predictions",marker="s")

# sns.lineplot(x=df.num_deletions.values,y=errors,label="Errors",marker="s")
sns.lineplot(x=df.num_deletions.values,y=acc_dis_pred,label="Acc Dis Predictions",marker="s")
sns.lineplot(x=df.num_deletions.values,y=df.acc_dis,label="True Acc Dis",marker="s")
plt.axhline(y=threshold,linestyle="--",color="black",alpha=0.5)
# sns.lineplot(x=df.num_deletions.values,y=df.pipeline_acc_dis_est,label="Acc Dis Predictions",marker="s")

plt.ylabel("")
plt.title(f"{dataset}, {' '.join(sampling_type.split('_'))},$\sigma={noise_level}$ Threshold:{threshold}")
# %%
sampling_type = "targeted_informed"
noise_level = 0
df = noise_filter(sampling_type_filter(gol_test_df,sampling_type),noise_level)
# sns.lineplot(data=df,x="num_deletions",y="pipeline_acc_err",label="pipeline",hue="threshold")
sns.lineplot(data=df,x="num_deletions",y="acc_err",label="pipeline",hue="threshold")

# sns.lineplot(data=df,x="num_deletions",y="abs_dis",label="abs dis")
# sns.lineplot(data=df,x="num_deletions",y="abs_err_init",label="abs err init ")

# %%
