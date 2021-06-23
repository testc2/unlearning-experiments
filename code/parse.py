#%%
from collections import defaultdict
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
import pandas as pd
import xmltodict
import seaborn as sns
from collections import OrderedDict

# %%
scientific = "([+-]?\d(\.\d+)?[Ee][+-]?\d+)"
normal = "(\d+(\.\d+)?)"

accuracy_pat = re.compile("Accuracy:\s*([01].\d+)")
f1_pat = re.compile("F1 Score:\s*([01].\d+)")
difference_pat = re.compile(f"tensor\(({scientific}|{normal})")
time_pat = re.compile(f"time.*::\s*({scientific}|{normal})")

def get_f1_score(s,split):
    try:
        return float(re.search(f"^{split}.*{f1_pat.pattern}",s,flags=re.MULTILINE).group(1))
    except Exception as e:
        # print(e)
        return None
    
def get_accuracy(s,split):
    try:
        return float(re.search(f"^{split}.*{accuracy_pat.pattern}",s,flags=re.MULTILINE).group(1))
    except Exception as e:
        # print(e)
        return None

def get_model_diff(s):
    try:
        return float(difference_pat.search(s).group(1))
    except Exception as e:
        return None

def get_time(s):
    try:
        return float(time_pat.search(s).group(1))
    except Exception as e:
        # print(e)
        return None

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

#%%
def get_deltagrad_dataframes(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"]    

    baseline_dict = d["baseline"]
    baseline_rows = []
    row =dict(
        method="Deltagrad",
        num_removes=6000,
        removal_time=None,
        noise=None,
        test_accuracy=None,
        remove_accuracy=None,
        remain_accuracy=None,
        sgd_seed=None,
    )
    for i,baseline_run in enumerate(baseline_dict):
        row["sgd_seed"] = i
        row["removal_time"]=get_time(baseline_run["#text"])
        for noise_level in baseline_run["noise"]:
            row["noise"] = float(noise_level["@sigma"])
            row["noise_seed"] = float(noise_level["@seed"])
            text = noise_level["#text"]
            row["test_accuracy"]=get_accuracy(text,"Test")
            row["remove_accuracy"]=get_accuracy(text,"Remove")
            row["prime_accuracy"]=get_accuracy(text,"Remain")
            baseline_rows.append(row.copy())
        
    baseline_df = pd.DataFrame(baseline_rows)

    deltagrad_dict = d["deltagrad"]
    deltagrad_rows = []
    for i,deltagrad_run in enumerate(deltagrad_dict):
        row["sgd_seed"] = i
        for time_period in deltagrad_run["time"]:
            row["period"] = int(time_period["@period"])
            row["removal_time"] = get_time(time_period["#text"])
            for noise_level in time_period["noise"]:
                row["noise"]= float(noise_level["@sigma"])
                row["noise_seed"] = float(noise_level["@seed"])
                text = noise_level["#text"]
                row["test_accuracy"]=get_accuracy(text,"Test")
                row["remove_accuracy"]=get_accuracy(text,"Remove")
                row["prime_accuracy"]=get_accuracy(text,"Remain")
                deltagrad_rows.append(row.copy())
    
    deltagrad_df = pd.DataFrame(deltagrad_rows)
    
    return deltagrad_df, baseline_df

def get_deltagrad_guo_dataframes(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"] 
    brow =dict(
        method="Deltagrad_Guo_Baseline",
        num_removes=6000,
        removal_time=None,
        noise=None,
        noise_seed=None,
        test_accuracy=None,
        remove_accuracy=None,
        remain_accuracy=None,
        model_diff=None
    )
    drow = brow.copy()
    drow["method"] = "Deltagrad_Guo"
    baseline_rows = []
    deltagrad_rows = []
    def get_stats(text):
        return dict(
        removal_time = get_time(text),
        test_accuracy = get_accuracy(text,"Test"),
        remove_accuracy = get_accuracy(text,"Remove"),
        prime_accuracy = get_accuracy(text,"Remain"),
        model_diff = get_model_diff(text)
        )
    
    for noise_run in d["noise"]:
        drow["noise"] = brow["noise"] = float(noise_run["@sigma"])
        drow["noise_seed"] = brow["noise_seed"] = float(noise_run["@seed"])
        baseline_text = noise_run["baseline"]
        brow.update(get_stats(baseline_text))
        baseline_rows.append(brow.copy())
        for time_period in noise_run["deltagrad"]["time"]:
            drow["period"] = float(time_period["@period"])
            deltagrad_text = time_period["#text"]
            drow.update(get_stats(deltagrad_text))
            deltagrad_rows.append(drow.copy())

    baseline_df = pd.DataFrame(baseline_rows)    
    deltagrad_df = pd.DataFrame(deltagrad_rows)

    return deltagrad_df,baseline_df

def get_params_test(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"] 
    
    row = dict(
        method=None,
        lr=None,
        epochs=None,
        batch_size=None,
        model_diff=None,
        test_accuracy=None,
        prime_accuracy=None,
        remove_accuracy=None,
        time=None
    )
    rows = []
    results = d["results"]
    def get_stats(text):
        return dict(
        time = get_time(text),
        test_accuracy = get_accuracy(text,"Test"),
        remove_accuracy = get_accuracy(text,"Remove"),
        prime_accuracy = get_accuracy(text,"Remain"),
        model_diff = get_model_diff(text)
        )
    for result in results:
        row["lr"] = result["@lr"]
        row["epochs"] = result["@epochs"]
        row["batch_size"] = result["@bz"]
        for method in ["Baseline","Deltagrad"]:
            row["method"] = method
            row.update(get_stats(result[method]))
            rows.append(row.copy())
    
    return pd.DataFrame(rows)

def get_deltagrad_dist_dataframes(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"] 

    row = [
        "method",
        "lr",
        "batch_size",
        "epochs",
        "sample_prob",
        "sampler_seed",
        "removal_time",
        "test_accuracy",
        "remove_accuracy",
        "prime_accuracy",
        "test_f1_score",
        "remove_f1_score",
        "prime_f1_score",
        "model_diff"
    ]
    row = {k:None for k in row}
    def get_stats(text):
        return dict(
        removal_time = get_time(text),
        test_accuracy = get_accuracy(text,"Test"),
        prime_accuracy = get_accuracy(text,"Remain"),
        remove_accuracy = get_accuracy(text,"Remove"),
        test_f1_score = get_f1_score(text,"Test"),
        prime_f1_score = get_f1_score(text,"Remain"),
        remove_f1_score = get_f1_score(text,"Remove"),
        model_diff = get_model_diff(text)
        )
    d_rows = []
    b_rows = []
    results_dict = d["results"]
    for result in results_dict:
        row["lr"]=float(result["@lr"])
        row["batch_size"]=int(result["@bz"])
        row["epochs"]=int(result["@epochs"])
        row["sample_prob"]=float(result["@sample_prob"])
        row["sampler_seed"]=int(result["@sampler_seed"])
        baseline_results = result["Baseline"]
        row["method"]="baseline"
        row.update(get_stats(baseline_results))
        b_rows.append(row.copy())
        row["method"] = "Deltagrad"
        for deltagrad in result["Deltagrad"]:
            row["period"] = int(deltagrad["@period"])
            row.update(get_stats(deltagrad["#text"]))
            d_rows.append(row.copy())
        del row["period"]
        
    return pd.DataFrame(d_rows),pd.DataFrame(b_rows)

def get_deltagrad_perturb_dataframes(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"]    

    deltagrad_dict = d["deltagrad"]
    deltagrad_rows = []
    row =dict(
        method="Deltagrad",
        num_removes=6000,
        removal_time=None,
        noise=None,
        test_accuracy=None,
        test_f1_score=None,
        model_diff=None,
        sgd_seed=None,
    )
    if not isinstance(deltagrad_dict,list):
        deltagrad_dict = [deltagrad_dict]
    for i,deltagrad_run in enumerate(deltagrad_dict):
        row["sgd_seed"] = i
        row["petrurb_time"]=get_time(deltagrad_run["#text"])
        for noise_level in deltagrad_run["noise"]:
            row["noise"] = float(noise_level["@sigma"])
            row["noise_seed"] = float(noise_level["@seed"])
            text = noise_level["#text"]
            row["test_accuracy"]=get_accuracy(text,"Test")
            row["test_f1_score"]=get_f1_score(text,"Test")
            row["model_diff"]=get_model_diff(text)
            deltagrad_rows.append(row.copy())
        
    deltagrad_df = pd.DataFrame(deltagrad_rows)
    
    return deltagrad_df

def remove_ratio_single(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"] 

    row = [
        "method",
        "lr",
        "batch_size",
        "epochs",
        "remove_ratio",
        "removal_time",
        "test_accuracy",
        "remove_accuracy",
        "prime_accuracy",
        "test_f1_score",
        "remove_f1_score",
        "prime_f1_score",
        "model_diff",
        "sampling_type"
    ]
    row = {k:None for k in row}
    def get_stats(text):
        return dict(
        removal_time = get_time(text),
        test_accuracy = get_accuracy(text,"Test"),
        prime_accuracy = get_accuracy(text,"Remain"),
        remove_accuracy = get_accuracy(text,"Remove"),
        test_f1_score = get_f1_score(text,"Test"),
        prime_f1_score = get_f1_score(text,"Remain"),
        remove_f1_score = get_f1_score(text,"Remove"),
        model_diff = get_model_diff(text)
        )
    d_rows = []
    b_rows = []
    results_dict = [d["results"]]
    for result in results_dict:
        row["lr"]=float(result["@lr"])
        row["batch_size"]=int(result["@bz"])
        row["epochs"]=int(result["@epochs"])
        row["remove_ratio"]=float(result["@remove_ratio"])
        row["sampling_type"]=result["@sampling_type"]
        baseline_results = result["Baseline"]
        row["method"]="baseline"
        row.update(get_stats(baseline_results))
        b_rows.append(row.copy())
        row["method"] = "Deltagrad"
        for deltagrad in result["Deltagrad"]:
            row["period"] = int(deltagrad["@period"])
            row.update(get_stats(deltagrad["#text"]))
            d_rows.append(row.copy())
        del row["period"]
        
    return pd.DataFrame(d_rows),pd.DataFrame(b_rows)

def get_dg_remove_ratio_frames(results_dir,dataset,output_file_prefix):
    deltagrad_frames = []
    baseline_frames = []
    for file in (results_dir/dataset).glob(f"{output_file_prefix}*.xml"):
        dg,bs = remove_ratio_single(file)
        deltagrad_frames.append(dg)
        baseline_frames.append(bs)
    
    return pd.concat(deltagrad_frames),pd.concat(baseline_frames)

def unlearn_single(path_to_xml):
    with open(path_to_xml,"rb") as fp:
        d = xmltodict.parse(fp)["data"] 

    row = [
        "method",
        "lr",
        "batch_size",
        "epochs",
        "remove_ratio",
        "removal_time",
        "test_accuracy",
        "remove_accuracy",
        "prime_accuracy",
        "test_f1_score",
        "remove_f1_score",
        "prime_f1_score",
        "model_diff",
        "sampling_type",
        "noise",
        "noise_seed"
    ]
    row = {k:None for k in row}
    def get_stats(text):
        return dict(
        test_accuracy = get_accuracy(text,"Test"),
        prime_accuracy = get_accuracy(text,"Remain"),
        remove_accuracy = get_accuracy(text,"Remove"),
        test_f1_score = get_f1_score(text,"Test"),
        prime_f1_score = get_f1_score(text,"Remain"),
        remove_f1_score = get_f1_score(text,"Remove"),
        model_diff = get_model_diff(text)
        )
    d_rows = []
    b_rows = []
    results_dict = [d["results"]]
    def get_noise_rows(run,row):
        row_list = []
        row["removal_time"]=get_time(run["#text"])
        for noise_level in run["noise"]:
            row["noise"] = float(noise_level["@sigma"])
            row["noise_seed"] = float(noise_level["@seed"])
            row.update(get_stats(noise_level["#text"]))
            row_list.append(row.copy())
        return row_list
    for result in results_dict:
        row["lr"]=float(result["@lr"])
        row["batch_size"]=int(result["@bz"])
        row["epochs"]=int(result["@epochs"])
        row["remove_ratio"]=float(result["@remove_ratio"])
        row["sampling_type"]=result["@sampling_type"]
        baseline_results = result["baseline"]
        row["method"]="baseline"
        b_rows.extend(get_noise_rows(baseline_results,row))
        row["method"] = "deltagrad"
        for deltagrad in result["deltagrad"]:
            row["period"] = int(deltagrad["@period"])
            d_rows.extend(get_noise_rows(deltagrad,row))
        del row["period"]
        
    return pd.DataFrame(d_rows),pd.DataFrame(b_rows)

def get_dg_unlearn_frames(results_dir,dataset,output_file_prefix):
    deltagrad_frames = []
    baseline_frames = []
    for file in (results_dir/dataset).glob(f"{output_file_prefix}*.xml"):
        dg,bs = unlearn_single(file)
        deltagrad_frames.append(dg)
        baseline_frames.append(bs)
    
    return pd.concat(deltagrad_frames),pd.concat(baseline_frames)

#%%|
if __name__  == "__main__":
    output_file = data_dir/"results"/"MNIST"/"Deltagrad_unlearn_multi_0.075s.xml"
    with open(output_file,"rb") as fp:
        d = xmltodict.parse(fp)["data"]   
# %%