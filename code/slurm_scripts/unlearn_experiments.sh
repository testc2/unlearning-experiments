#!/bin/bash
#SBATCH -o result_unlearn_%x.txt
#SBATCH -M ukko2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ananth.mahadevan@helsinki.fi

repo_name="unlearning-experiments"
#suffix="--suffix _test"
suffix=""
# the minibatch fractions for INFLUENCE and FISHER 
minibatch_fraction="1 2 4 8"

case $1 in 
    MNISTBinary)
        dataset="MNIST"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.2 0.3 0.375"
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
        ;;
    EPSILON)
        dataset="EPSILON"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_noise_seeds="3"
        remove_ratios="0.1 0.2 0.25"
        ;;
    *)
        echo -n "Error, Useage: sbatch unlearn_experiments.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

echo "$1 Dataset Selected"
echo "Check $WRKDIR/${repo_name}/data/results/$dataset for results"
echo "Performing Unlearning Experiments"
srun python $WRKDIR/${repo_name}/code/run_exp.py --optim SGD --step-size $lr --batch-size $bz --num-steps $epochs --verbose unlearn $dataset --remove-ratios $remove_ratios --num-processes $SLURM_CPUS_ON_NODE --l2-norm --num-noise-seeds $num_noise_seeds $ovr --minibatch-fraction ${minibatch_fraction} $suffix
