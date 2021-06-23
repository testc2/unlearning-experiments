#!/bin/bash
#SBATCH -o result_perturb_%x.txt
#SBATCH -M ukko2

repo_name="unlearning-experiments"

case $1 in 
    MNISTBinary)
        dataset="MNIST"
        ratio="0.4"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_seeds="6"
        ;;
    MNISTOVR)
        dataset="MNIST"
        n_classes="10"
        ratio="0.08"
        epochs="200"
        bz="512"
        lr="1"
        ovr="--ovr"
        num_seeds="3"
        ;;
    COVTYPEBinary)
        dataset="COVTYPE"
        epochs="200"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        ;;
    HIGGS)
        dataset="HIGGS"
        epochs="20"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        ;;
    CIFARBinary)
        dataset="CIFAR"
        epochs="500"    
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        ;;
    EPSILON)
        dataset="EPSILON"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        ;;
    *)
        echo -n "Error, Useage: sbatch perturb_experiments.job (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

echo "$1 Dataset Selected"
echo "Check $WRKDIR/${repo_name}/data/results/$dataset for results"
echo "Performing Noise Injection Experiments"
srun python $WRKDIR/${repo_name}/code/run_exp.py --optim SGD --step-size $lr --batch-size $bz --num-steps $epochs --verbose perturb $dataset --num-processes $SLURM_CPUS_ON_NODE --l2-norm --num-seeds $num_seeds $ovr
