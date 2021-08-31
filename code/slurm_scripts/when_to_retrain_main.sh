#!/bin/bash
repo_name="unlearning-experiments"

# retrain results
sbatch --mem $2 -t $3 -c $4 -J "$1_Retrain" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 retrain

# nothing results 
sbatch --mem $2 -t $3 -c $4 -J "$1_Nothing" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 nothing

# golatkar results
sbatch --mem $2 -t $3 -c $4 -J "$1_Gol" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar

# golatkar threshold results
sbatch --mem $2 -t $3 -c $4 -J "$1_Gol_Test" $WRKDIR/${repo_name}/code/slurm_scripts/when_to_retrain.sh $1 golatkar_test_thresh


