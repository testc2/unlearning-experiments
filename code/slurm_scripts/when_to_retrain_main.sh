#!/bin/bash
repo_name="unlearning-experiments"
flag=false
case $5 in 

    retrain|all)
        # retrain results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Retrain" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 retrain
        flag=true
        ;;&
    nothing|all)
        # nothing results 
        sbatch --mem $2 -t $3 -c $4 -J "$1_Nothing" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 nothing
        flag=true
        ;;&
    golatkar|all)
        # golatkar results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar
        flag=true
        ;;&
    golatkar_test|all)
        # golatkar threshold results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Test" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_test_thresh
        flag=true
        ;;

esac

if [ "$flag" = false ]; then
    echo -n "Error Usage: ./master.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) 5 (retrain|nothing|golatkar|golatkar_test|all)"
    exit 1
fi
   