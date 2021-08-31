#!/bin/bash
repo_name="unlearning-experiments"

# retrain results
sbatch --mem $2 -t $3 -c $4 -J "Pipeline retrain" $WRKDIR/${repo_name}/code/deltagrad_scripts/when_to_retrain.sh $1 retrain

# nothing results 
sbatch --mem $2 -t $3 -c $4 -J "Pipeline nothing" $WRKDIR/${repo_name}/code/deltagrad_scripts/when_to_retrain.sh $1 nothing

# golatkar results
sbatch --mem $2 -t $3 -c $4 -J "Pipeline golatkar" $WRKDIR/${repo_name}/code/deltagrad_scripts/when_to_retrain.sh $1 golatkar

# golatkar threshold results
sbatch --mem $2 -t $3 -c $4 -J "Pipeline golatkar test" $WRKDIR/${repo_name}/code/deltagrad_scripts/when_to_retrain.sh $1 golatkar_test_thresh


