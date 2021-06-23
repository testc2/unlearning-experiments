# CODE

The code is split into removal methods and experiments.  
The experiments are run on a SLURM system so we provide scripts to run these for ease of collection of results.
Apart from this, you can also run the experiments from run_exp.py

# UNLEARNING METHODS

In the paper we compare three unlearning methods namely, INFLUENCE, FISHER and DELTAGRAD methods.

The details regarding INFLUENCE and FISHER implementations are found in the following [README](methods/README.md).
The DELTAGRAD implementation details are found in the following [README](../external_code/DeltaGrad/src/README.md).

## Experiments

The [experiments](experiments/) subfolder contains the implementation of the experiments for the INFLUENCE and FISHER methods. These can be launched from the [run_exp.py](../run_exp.py) script as subparser arguments. These experiments collect the results and place them in a corresponding `.csv` file. Refer to the [slurm_scripts](../code/slurm_scripts/) folder readme for a better understanding of where and how these files look.

On the other hand the experiments for the DELTAGRAD method are present as scripts in the [deltagrad_scripts](../deltagrad_script) folder, where we use the external modified code from the entry point of [main.py](../external_code/DeltaGrad/src/main.py) and collect the results in an XML file. To process them into data frames which is used in plotting, we utilize the [parse.py](../code/parse.py) file, which contains several XML to DataFrame parsers based on the experiment data.

