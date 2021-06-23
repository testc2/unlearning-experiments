# Modifications

Brief on the modifications provided to interface the original DeltaGrad code.

## DATA LOADING

We override the data loading to ensure that all the unlearning methods and experiments use the same [pre-processing](../../../code/methods/common_utils.py).
To acheive this we add functions in the [DataPreparer](Models/Data_preparer.py) module that correspond to the pre-processing.
The modified function include the names and the binary/multi-class datasets found in [datasets](../../../data/README.md).

For example, `prepare_MNISTBinary` and `prepare_MNISTOVR`. We also define hyperparameter functions such as `get_hyperparameters_MNISTBinary` to ensure that all unlearning methods use only vanilla `optim.SGD` with the `BCEWithLogitsLoss` as the binary loss function.

## One VS Rest Functionality

The original DeltaGrad paper considers a multinomial Logistic regression model for even binary datasets. This is not optimal if we are comparing the other unlearning methods.

Therefore, we define a custom ``LR`` class in [DNN_single.py](Models/DNN_single.py) to ensure a binary Logistic Regression model for all the binary datasets, which interfaces directly with the rest of the DeltaGrad code.

To achieve multi-class classification for $k>2$ classes, we train $k$ independent binary models ins a One-VS-Rest manner. This is primarily achieved in [main_delete_ovr.py](main_delete_ovr.py) where we write code to handle training and updating these $k$ models and storing their SGD information that DeltaGrad requires.

## Deletion Distributions

For the experiments related to deletion distributions, we have the [generate_sampled_delta_ids.py](generate_sampled_delta_ids.py) which allows us to procure the ids of the deleted data using one of the four distribution types <tt>uniform-random</tt>,<tt>uniform-informed</tt>,<tt>targeted-random</tt> or <tt>targeted-informed</tt> along with the deletion ratio indicating how much of the training data we wish to delete.

## Noise Injection Mechanism

We extend the DeltaGrad method with a Gaussian noise injection method as described in the paper. The code correspond to this is found in the [privacy_experiments.py](privacy_experiments.py), where we include adding the random noise to the trained/updated parameters and outputting the results.

## Results

As we don't modify the output for the DeltaGrad experiments, we collect all the results in an XML file and later parse them to obtain the metrics and data.
This code can be found in []
