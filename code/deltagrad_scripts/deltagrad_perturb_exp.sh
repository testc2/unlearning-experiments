#!/bin/bash
#SBATCH -o result_perturb_%x.txt
#SBATCH -e error_perturb_%x.txt
#SBATCH -M ukko2

repo_name="unlearning-experiments"
# directory of the deltagrad code
src_dir="$WRKDIR/${repo_name}/external_code/DeltaGrad/src"

gpu="" #"--GPU --GID 0"
# remove no data in noise injection experiments
ratio="0.0"


# specify other fixes paramters for method
# specify base data directory so that a symbolic link can be created to it to work in array jobs
case $1 in 
    MNISTBinary)
        j0="10"
        dataset="MNIST"
        ovr_str="binary"
        epochs="1000"
        bz="1024"
        lr="1"
        ovr=""
        num_seeds="6"
        remove_ratios="0.2 0.3 0.375"
        base_data_dir="$WRKDIR/${repo_name}/data/MNIST/"
        ;;
    MNISTOVR)
        j0="20"
        dataset="MNIST"
        ovr_str="multi"
        epochs="200"
        bz="512"
        lr="1"
        ovr="--ovr"
        num_seeds="3"
        base_data_dir="$WRKDIR/${repo_name}/data/MNIST/"
        ;;
    COVTYPEBinary)
        j0="10"
        dataset="COVTYPE"
        ovr_str="binary"
        epochs="200"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        base_data_dir="$WRKDIR/${repo_name}/data/COVTYPE/"
        ;;
    HIGGS)
        j0="500"
        dataset="HIGGS"
        ovr_str="binary"
        epochs="20"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        base_data_dir="$WRKDIR/${repo_name}/data/HIGGS/"
        ;;
    CIFARBinary)
        j0="20"
        dataset="CIFAR"
        ovr_str="binary"
        epochs="500"    
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        base_data_dir="$WRKDIR/${repo_name}/data/CIFAR/"
        ;;
    EPSILON)
        j0="10"
        dataset="EPSILON"
        ovr_str="binary"
        epochs="60"
        bz="512"
        lr="1"
        ovr=""
        num_seeds="3"
        base_data_dir="$WRKDIR/${repo_name}/data/epsilon/"
        ;;
    *)
        echo -n "Error, Useage: sbatch deltagrad_perturb.sh (MNISTBinary|MNISTOVR|COVTYPEBinary|HIGGS|CIFARBinary|EPSILON)"
        exit 1
    ;;
esac

data_dir="$WRKDIR/${repo_name}/data"
results_dir="$data_dir/results/$dataset"
file="${results_dir}/Deltagrad_perturb_${ovr_str}${suffix}.xml"
temp_dir="${data_dir}/${dataset}_${ovr_str}_temp_perturb/"
# create symbolic copy of data
cp -rs "$base_data_dir" "$temp_dir"
# switch to source dir
cd $src_dir

# preprocess and generate required dataset
python3 generate_dataset_train_test.py --model LR --dataset $1 $ovr --repo $temp_dir $ovr

# start the XML file
echo "<data>" > $file
echo "<generate>" >> $file
python3 generate_rand_delta_ids.py --dataset $1 --ratio $ratio --restart --repo $temp_dir >> $file
echo "</generate>" >> $file

echo "<training>" >> $file
python3 main.py --bz $bz --epochs $epochs --model LR --dataset $1 --wd 0.0001  --lr $lr --lrlen $epochs  --train $gpu --repo $temp_dir $ovr >> $file 
echo "</training>" >> $file

echo "<deltagrad>" >> $file
python3 main.py --bz $bz --epochs $epochs --model LR --dataset $1 --wd 0.0001  --lr $lr --lrlen $epochs  --method baseline $gpu --repo $temp_dir $ovr --privacy --num-seeds $num_seeds >> $file
echo "</deltagrad>" >> $file

echo "</data>" >> $file
# remove something that looks like an XML tag in the output
sed -i "s/<SqrtBackward>//g" $file

# delete temp symbolic dir
rm -rf ${temp_dir}
