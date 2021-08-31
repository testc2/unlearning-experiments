#!/bin/bash
#SBATCH -o result_unlearn_%x.txt
#SBATCH -M ukko2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ananth.mahadevan@helsinki.fi
#SBATCH --ntasks=1
#SBATCH --array=1-6

repo_name="unlearning-experiments"

case $1 in 
    MNISTBinary)
        dataset="MNIST"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.2 0.3 0.375"
        deletion_batch_size="100"
        ;;
    MNISTOVR)
        dataset="MNIST"
        epochs="200"
        bz="512"
        lr="1"
        ovr="--ovr"
        num_noise_seeds="3"
        remove_ratios="0.01 0.05 0.075"
        ;;
    COVTYPEBinary)
        dataset="COVTYPE"
        epochs="200"
        bz="512"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.05 0.10 0.15"
        deletion_batch_size="1000"
        ;;
    HIGGS)
        dataset="HIGGS"
        epochs="20"
        bz="512"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.01 0.05 0.10"
        ;;
    CIFARBinary)
        dataset="CIFAR"
        epochs="500"    
        bz="512"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.05 0.125 0.2"
        deletion_batch_size="100"
        ;;
    EPSILON)
        dataset="EPSILON"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.1 0.2 0.25"
        deletion_batch_size="500"
        ;;
    *)
        echo -n "Error, Useage: sbatch unlearn_experiments.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

strategy=$2
echo "$1 Dataset Selected"
echo "Check $WRKDIR/${repo_name}/data/results/$dataset/when_to_retrain for results"
echo "Performing When to Retrain Experiments"

sampling_type="uniform_random uniform_random targeted_random targeted_random uniform_informed targeted_informed"
sampler_seeds="0 1 0 1 0 0"

srun python $WRKDIR/${repo_name}/code/run_exp.py --optim SGD --step-size $lr --batch-size $bz --num-steps $epochs --verbose when $dataset ${remove_ratios[2]} $deletion_batch_size $sampling_type[$SLURM_ARRAY_TASK_ID] --num-processes $SLURM_CPUS_ON_NODE --l2-norm $ovr --sampler-seed $sampler_seeds[$SLURM_ARRAY_TASK_ID] --suffix "_$SLURM_ARRAY_TASK_ID" $strategy