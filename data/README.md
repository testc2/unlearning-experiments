# DATA
This folder contains the datasets and the results from the experiments.

## DATASETS
All datasets are from the [LIBSVM repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) or from the PyTorch dataloader.
Download the datasets from the provided links. Unzipped the downloaded files and place them in the appropriate subfolder.

When the experiments are first run for each dataset, the raw LIBSVM datafiles will be loaded and [pre-processed](../code/methods/common_utils.py) and the training and test files are placed in the appropriate `processed/` subfolder. This might take quite a while for the larger datasets such as HIGGS and EPSILON. After the first time, the processed files will be used.

In the future, we will add a link to directly download the processed files (which are smaller in size).

1. **MNIST**:
Just run any experiment that uses the MNIST dataset. It will download the raw files from PyTorch and place the raw files in `./MNIST/raw/` and the processed files in `./MNIST/processed/`. No other downloads required.
2. **COVTYPE**:
Download the binary file [covtype.libsvm.binary.scale.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary), unzip and place it in the `./COVTYPE/raw/` subfolder.
3. **HIGGS**:
Download the [HIGGS.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#HIGGS), unzip and place it in `./HIGGS/raw/`. Note, this is a rather large dataset, unzipped it is around **7.9GB**.
4. **CIFAR**
Dowload the [cifar10.bz1 and cifar10.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#cifar10) files, unzip and place it in the `./CIFAR/raw/` subfolder.
5. **EPSILON**:
Download the [epsilon_normalized.bz2 and epsilon_normalized.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) files, unzip and place them in `./EPSILON/raw/`. Note, this is a rather large dataset, unzipped **>15GB.**

## RESULTS
