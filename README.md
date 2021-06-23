Code for the paper:

> Ananth Mahadevan and Michael Mathioudakis, Certifiable Machine Unlearning for Linear Models 

## Requirements 
Please refer to the requirements.txt for a complete list of the requirement
- Python 3.6
- PyTorch v1.2
- Scikit-Learn
- Numpy
- Scipy

## Data
Please download the data and place them in the respective folders as described in the [dataset README](data/README.md) file.

## Code
The paper discusses three unlearning methods, namely INFLUENCE, FISHER and DELTAGRAD. Their codes can be respectively found in [remove.py](code/methods/remove.py), [scrub.py](code/methods/scrub.py) and [main_delete.py](external_code/DeltaGrad/src/main.py).

To learn more about the implementations please refer the following readme files:  [influence and fisher](code/README.md) and [deltagrad](external_code/DeltaGrad/README.md).

## Experiments


## TODOs
- Add the rest of the code
- Add link to Arxiv paper
- Provide a set of results and plotting code