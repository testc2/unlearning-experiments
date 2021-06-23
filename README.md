Code for the paper:

> Ananth Mahadevan and Michael Mathioudakis, Certifiable Machine Unlearning for Linear Models 

## Requirements 
Please refer to the requirements.txt for a complete list of the requirement
- Python 3.6
- PyTorch v1.2
- Scikit-Learn
- Numpy
- Scipy

## Data
Please download the data and place them in the respective folders as described in the [data](data/) folder readme.

## Code
The paper discusses three unlearning methods, namely INFLUENCE, FISHER and DELTAGRAD. Their codes can be respectively found in [remove.py](code/methods/remove.py), [scrub.py](code/methods/scrub.py) and [main_delete.py](external_code/DeltaGrad/src/main.py).

To learn more about the implementations please refer the following readme files:  [influence and fisher](code/README.md) and [deltagrad](external_code/DeltaGrad/src/README.md).

## Experiments

The implementation code and details of the various experiments is present in the [experiments](code/experiments/) folder for the INFLUENCE and FISHER methods. Please refer to the readme file in the [code](code/) folder for further details on the DELTAGRAD method.

## Scripts

The scripts to run the experiments in present in [slurm_scrips](code/slurm_scripts/master.sh) folder.
We provide SLURM batch job scripts to efficiently collect results.
For example to collect experiment 2b results for all datasets, one can use

```{bash}
./code/slurm_scripts/master.sh all 2b
```

which will schedule SLURM batch jobs and place results in their respective folders in `data/results`.

Refer to the scripts [README](code/slurm_scripts/README.md) for more details on how the arguments and location of results.

## TODOs

- Add possible regular scripts as alternate to slurm scrips
- Add link to Arxiv paper
- Provide a set of results and plotting code
