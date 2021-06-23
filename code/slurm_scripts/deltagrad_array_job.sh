#!/bin/bash
#SBATCH -o result_%x.txt
#SBATCH -e error_%x.txt
#SBATCH -M ukko2
#SBATCH --ntasks=1
#SBATCH --array=0-2

repo_name="unlearning-experiments"
case $1 in 
    MNISTBinary)
        remove_ratios=(0.2 0.3 0.375)
        ;;
    MNISTOVR)
        remove_ratios=(0.01 0.05 0.075)
        ;;
    COVTYPEBinary)
        remove_ratios=(0.05 0.10 0.15)
        ;;
    HIGGS)
        remove_ratios=(0.01 0.05 0.10)
        ;;
    CIFARBinary)
        remove_ratios=(0.05 0.125 0.2)
        ;;
    EPSILON)
        remove_ratios=(0.1 0.2 0.25)
        ;;
    *)
        echo -n "Error, Useage: sbatch deltagrad_array_job.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) (2b|4b)"
        exit 1
    ;;
esac

# run the appropriate deltagrad script using an array job for each remove ratio for efficiency
case $2 in 
    2b) # for QoA experiments
        srun $WRKDIR/${repo_name}/code/deltagrad_scripts/deltagrad_remove_ratio.sh $1 ${remove_ratios[$SLURM_ARRAY_TASK_ID]}
        ;;
    4b) # for unlearn experiments 
        srun $WRKDIR/${repo_name}/code/deltagrad_scripts/deltagrad_unlearn_exp.sh $1 ${remove_ratios[$SLURM_ARRAY_TASK_ID]}
        ;;
    *)
        echo -n "Error Useage ./deltagrad_array_job.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) (2b|4b)"
        exit 1
    ;;
esac


