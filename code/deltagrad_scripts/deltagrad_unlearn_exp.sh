#!/bin/bash

repo_name="unlearning-experiments"
# directory of the deltagrad code
src_dir="$WRKDIR/${repo_name}/external_code/DeltaGrad/src"

gpu="" #"--GPU --GID 0"
ratio=$2
# time period for deltagrad
t0s="2 5 10 20 50 75 100 200"
sampling_type="targeted_informed"

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
        remove_class="0"
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
        remove_class="3"
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
        remove_class="0"
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
        remove_class="0"
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
        remove_class="0"
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
        remove_class="0"
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
file="${results_dir}/Deltagrad_unlearn_${ovr_str}${suffix}_${ratio}.xml"
temp_dir="${data_dir}/${dataset}_${ovr_str}_temp_unlearn_${ratio}/"
cp -rs "$base_data_dir" "$temp_dir"

cd $src_dir

# start XML
echo "<data>" > $file
# preprocess and generate required dataset
echo "<generate>" >> $file
python3 generate_dataset_train_test.py --model LR --dataset $1 --repo $temp_dir $ovr >> $file
echo "</generate>" >> $file

echo "<training>" >> $file
python3 main.py --bz $bz --epochs $epochs --model LR --dataset $1 --wd 0.0001  --lr $lr --lrlen $epochs  --train $gpu --repo $temp_dir $ovr >> $file 
echo "</training>" >> $file

# sample the removed indices
python3 generate_sampled_delta_ids.py $sampling_type --dataset $1 --ratio $ratio --class-id $remove_class --repo $temp_dir >> $file
echo "<results lr=\"$lr\" epochs=\"$epochs\" bz=\"$bz\" remove_ratio=\"$ratio\" sampling_type=\"$sampling_type\">" >> $file
# train from scratch
echo "<baseline>" >> $file
python3 main.py --bz $bz --epochs $epochs --model LR --dataset $1 $ovr --wd 0.0001  --lr $lr  --lrlen $epochs  --method baseline $gpu --repo $temp_dir --privacy --num-seeds $num_seeds>> $file
echo "</baseline>" >> $file

for t0 in $t0s; do 
    echo "<deltagrad period=\"$t0\">" >> $file
    python3 main.py --bz $bz --epochs $epochs --model LR --dataset $1 $ovr --wd 0.0001  --lr $lr  --lrlen $epochs --method deltagrad --period $t0 --init $j0 -m 2 --cached_size 20 $gpu --repo $temp_dir --privacy --num-seeds $num_seeds >> $file
    echo "</deltagrad>" >> $file
done
echo "</results>" >> $file
echo "</data>" >> $file
# remove something that looks like an XML tag in the output
sed -i "s/<SqrtBackward>//g" $file
rm -rf ${temp_dir}
