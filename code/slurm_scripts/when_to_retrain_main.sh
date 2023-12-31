#!/bin/bash
repo_name="unlearning-experiments"
flag=false

if [ "$6" == "noise" ]; then
echo "noise experiments"
case $5 in 

    retrain|all)
        # retrain results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Retrain" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 retrain $6
        flag=true
        ;;&
    nothing|all)
        # nothing results 
        sbatch --mem $2 -t $3 -c $4 -J "$1_Nothing" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 nothing $6
        flag=true
        ;;&
    golatkar|all)
        # golatkar results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar $6
        flag=true
        ;;&
    golatkar_test|all)
        # golatkar threshold results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Test" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_test_thresh $6
        flag=true
        ;;&
    golatkar_disparity_v1|all)
        # golatkar disparity v1 strategy
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Dis_v1" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_disparity_thresh_v1 $6
        flag=true
        ;;&
    golatkar_disparity_v2|all)
        # golatkar disparity v2 strategy
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Dis_v2" --array=0-35 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_disparity_thresh_v2 $6
        flag=true
        ;;

esac
else
echo "no noise experiments"
case $5 in 
    retrain|all)
        # retrain results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Retrain" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 retrain
        flag=true
        ;;&
    nothing|all)
        # nothing results 
        sbatch --mem $2 -t $3 -c $4 -J "$1_Nothing" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 nothing
        flag=true
        ;;&
    golatkar|all)
        # golatkar results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar
        flag=true
        ;;&
    golatkar_test|all)
        # golatkar threshold results
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Test" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_test_thresh
        flag=true
        ;;&
    golatkar_disparity_v1|all)
        # golatkar disparity v1 strategy
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Dis_v1" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_disparity_thresh_v1
        flag=true
        ;;&
    golatkar_disparity_v2|all)
        # golatkar disparity v2 strategy
        sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Dis_v2" --array=0-5 $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_disparity_thresh_v2
        flag=true
        ;;
esac

fi
if [ "$flag" = false ]; then
    echo -n "Error Usage: ./master.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON) 5 (retrain|nothing|golatkar|golatkar_test|golatkar_disparity_v1|golatkar_disparity_v2|all) (noise)"
    exit 1
fi
   