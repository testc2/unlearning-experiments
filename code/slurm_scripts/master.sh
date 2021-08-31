#!/bin/bash
if [ "$1" = "all" ]; then
datasets="MNISTBinary MNISTOVR COVTYPEBinary HIGGS CIFARBinary EPSILON"
else
datasets=$1
fi  
repo_name="unlearning-experiments"
# will submit batch jobs for all datasets
for dataset in $datasets; do
    echo $dataset
    # select memory requirements, time limits and cores based on dataset
    case $dataset in 
        MNISTBinary)
            mem="4000M"
            time="5:00:00"
            cores="24"
            ;;
        MNISTOVR)
            mem="10000M"
            time="9:30:00"
            cores="24"
            ;;
        COVTYPEBinary)
            mem="12000M"
            time="9:30:00"
            cores="24"
            ;;
        HIGGS)
            mem="40000M"
            time="9:30:00"
            cores="24"
            ;;
        CIFARBinary)
            mem="20000M"
            time="9:30:00"
            cores="24"
            ;;
        EPSILON)
            mem="180000M"
            time="9:30:00"
            cores="24"
            ;;
        *)
            echo -n "Error, available datasets (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
            exit 1
        ;;
    esac

    case $2 in 
    # Deletion Distribution Experiments 
        1)
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/slurm_scripts/deletion_distribution_exp.sh $dataset
            ;;
    # QoA experiments
        # for  INLUENCE and FISHER
        2a)
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/slurm_scripts/QoA_exp.sh $dataset
            ;;
        # for DeltaGrad
        2b) 
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/slurm_scripts/deltagrad_array_job.sh $dataset $2
            ;;
    # Noise Injection experiments 
        # for INFLUENCE and FISHER
        3a)
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/slurm_scripts/perturb_experiments.sh $dataset
            ;;
        # for DeltaGrad
        3b)
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/deltagrad_scripts/deltagrad_perturb_exp.sh $dataset
            ;;
    # Unlearning Experiments
        # for INFLUENCE and FISHER
        4a)
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/slurm_scripts/unlearn_experiments.sh $dataset
            ;;
        # for DeltaGrad
        4b)
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/slurm_scripts/deltagrad_array_job.sh $dataset $2
            ;;
    # When to retrain experiments
        5)
            $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain_main.sh $dataset $mem $time $cores
            ;;
        *)
            echo -n "Error Usage: ./master.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) (1|2a|2b|3a|3b|4a|4b)"
            exit 1
            ;;
    esac
    
        
done