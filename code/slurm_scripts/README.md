# SLURM Sscripts
These are scrips to submit batch jobs to a SLURM HPC cluster to get results for the experiments. The main file to launch jobs is [master.sh](master.sh). 

For example to run the script from the project root for all datasets for the deletion distribution experiments
```
./code/slurm_scripts/master.sh all 1
```

**NOTE**: The slurm jobs options might need to be modified to suit your cluster setting such as partition, etc. So please do look at the scrips in the folder before running them.

## Datasets 
The first cmd line option is the dataset(s) to collect results for. The availble ones are 
```
(MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)
```
You can send space separated names to collect results for multiple datasets or use the `all` option to collect for all the availble datasets.

## Experiments 
The second cmd line argument specifies the experiment to collect results for. There are 4 main experiments, however, we need to collect the DeltaGrad results separately. So the avaible options are described below.( for the results file example, we assume it is a binary dataset. The only difference for a multi-class dataset is that `_binary` will be replaced with `_multi` in the result file name.)

- **1** : 

    Deletion Distribution experiments. The results for will be found in `data/results/{Dataset}/Ratio_binary.csv`
- **2a** :

    QoA Experiments for INFLUENCE and FISHER methods. The results will be found in `data/results/{Dataset}/Remove_Dist_binary_selected.csv`
- **2b** : 

QoA Experiments for DeltaGrad method. The results will be found in `data/results/{Dataset}/Deltagrad_remove_ratio_binary_selected_{remove_ratio}.xml`, where `{remove_ratio}` is one of the chosen deletion ratio for the dataset (see [deltagrad_array_job.sh](deltagrad_array_job.sh)).
- **3a** : 

Noise Injection Experiments for INFLUENCE and FISHER methods. The results will be found in `data/results/{Dataset}/Perturbation_binary.csv`
- **3b** : 

Noise Injetion Experiments for DeltaGrad method. The results will be found in `data/results/{Dataset}/Deltagrad_perturb_binary.xml`.
- **4a** : 

Unlearning Experiments for INFLUENCE and FISHER methods. The results will be found in `data/results/{Dataset}/Unlearn_binary.csv`
- **4b**:

Unlearning Experiments for DeltaGrad method. The results will be found in `data/results/{Dataset}/Deltagrad_unlearn_binary_{remove_ratio}.csv`, where `{remove_ratio}` is one of the chosen deletion ratio for the dataset (see [[deltagrad_array_job.sh](deltagrad_array_job.sh)).

The slurm job also creates error and log files that can be used to monitor progress.

All results are stored in the respective folder in `data/results/{Dataset}`.

**Note:** All the scipts by default overwrite existing results files. If you wish to instead append to the exisitng results then look into the `--overwrite-mode` of the cmd line args of a particular experiment in [args.py](../experiments/args.py) and appropriately modify the scripts corresponding the experiments. 