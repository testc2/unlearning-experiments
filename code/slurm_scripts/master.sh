#!/bin/bash
datasets="MNISTBinary MNISTOVR COVTYPEBinary HIGGS CIFARBinary EPSILON"
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

    case $1 in 
    # Deletion Distribution Experiments 
        1)
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/deletion_distribution_exp.sh $dataset
    # QoA experiments
        2a) # for  INLUENCE and FISHER
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/QoA_exp.sh $dataset
            ;;
        2b) # for DeltaGrad
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/deltagrad_array_job.sh $dataset $1
            ;;
    # Noise Injection experiments 
        3a) # for INFLUENCE and FISHER
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/perturb_experiments.job $dataset
            ;;
        3b) # for DeltaGrad
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/deltagrad_scripts/deltagrad_perturb_exp.sh $dataset
            ;;
    # Unlearning Experiments
        4a) # for INFLUENCE and FISHER
            sbatch --mem $mem -t $time -c $cores -J $dataset $WRKDIR/${repo_name}/code/unlearn_experiments.job $dataset
            ;;
        4b) # for DeltaGrad
            sbatch --mem $mem -t $time -c $cores -J "DG_$dataset" $WRKDIR/${repo_name}/code/deltagrad_array_job.sh $dataset $1
        *)
            echo -n "Error Usage: ./master.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) (1|2a|2b|3a|3b|4a|4b)"
            exit 1
            ;;
    esac
    
        
done