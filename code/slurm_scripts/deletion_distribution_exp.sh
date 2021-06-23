#!/bin/bash
#SBATCH -o result_%x.txt
#SBATCH -M ukko2
repo_name="unlearning-experiments"

# optional suffix 
# suffix="--suffix _extended"
suffix=""

# num_sampler_seed specifies how many random seed to use for the deletion distribution 
case $1 in 
    MNISTBinary)
        dataset="MNIST"
        ratio="0.4"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_sampler_seeds="2"
        ;;
    MNISTOVR)
        dataset="MNIST"
        n_classes="10"
        ratio="0.08"
        epochs="200"
        bz="512"
        lr="1"
        ovr="--ovr"
        num_sampler_seeds="2"
        ;;
    COVTYPEBinary)
        dataset="COVTYPE"
        epochs="200"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="2"
        ;;
    HIGGS)
        dataset="HIGGS"
        epochs="20"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="2"
        ;;
    CIFARBinary)
        dataset="CIFAR"
        epochs="500"    
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="2"
        ;;
    EPSILON)
        dataset="EPSILON"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="2"
        # for 24 cores uses 164GB memor
        ;;
    *)
        echo -n "Error, Useage: sbatch ratio_experiments.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

echo "$1 Dataset Selected"
echo "Check $WRKDIR/${repo_name}/data/results/$dataset for results"
echo "Performing Deletion Distribution Experiments"
srun python $WRKDIR/${repo_name}/code/run_exp.py --optim SGD --step-size $lr --batch-size $bz --num-steps $epochs --verbose ratio $dataset --num-processes $SLURM_CPUS_ON_NODE --l2-norm --num-sampler-seed $num_sampler_seeds $ovr $suffix
