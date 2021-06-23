#!/bin/bash
#SBATCH -o result_%x.txt
#SBATCH -M ukko2

repo_name="unlearning-experiments"

# an optional suffix
suffix="--suffix _selected"
# we only use the targeted-informed for the QoA experiments 
sampling_type="targeted_informed"
# the minibatch fractions for INFLUENCE and FISHER 
minibatch_fraction="1 2 4 8"
# the training details and deletion details for each dataset
case $1 in 
    MNISTBinary)
        dataset="MNIST"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_sampler_seeds="6"
        remove_ratios="0.2 0.3 0.375"
        ;;
    MNISTOVR)
        dataset="MNIST"
        epochs="200"
        bz="512"
        lr="1"
        ovr="--ovr"
        num_sampler_seeds="2"
        remove_ratios="0.01 0.05 0.075"
        ;;
    COVTYPEBinary)
        dataset="COVTYPE"
        epochs="200"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="3"
        remove_ratios="0.05 0.10 0.15"
        ;;
    HIGGS)
        dataset="HIGGS"
        epochs="20"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="3"
        remove_ratios="0.01 0.05 0.10"
        ;;
    CIFARBinary)
        dataset="CIFAR"
        epochs="500"    
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="3"
        remove_ratios="0.05 0.125 0.2"
        ;;
    EPSILON)
        dataset="EPSILON"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_sampler_seeds="3"
        remove_ratios="0.1 0.2 0.25"
        ;;
    *)
        echo -n "Error, Useage: sbatch ratio_experiments.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

echo "$1 Dataset Selected"
echo "Check $WRKDIR/${repo_name}/data/results/$dataset for results"
echo "Performing QoA Experiments"
srun python $WRKDIR/${repo_name}/code/run_exp.py --optim SGD --step-size $lr --batch-size $bz --num-steps $epochs --verbose dist $dataset --remove-ratios $remove_ratios --sampling-types $sampling_type --num-processes $SLURM_CPUS_ON_NODE --l2-norm --num-sampler-seed $num_sampler_seeds $ovr --minibatch-fraction ${minibatch_fraction} $suffix
