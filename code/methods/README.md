# Unlearning Methods

This subdirectory contains the implementation of the unlearning methods and related code.

## INFLUENCE

This unlearning method is based on the paper
> Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. **[Certified Data Removal from Machine Learning Models](https://arxiv.org/pdf/1911.03030.pdf)**. ICML 2020.

The code for this method is present in [remove.py](remove.py) which is mostly using the [authors original code](https://github.com/facebookresearch/certified-removal). It contains methods for mini-batch and OVR removal of deleted data from a trained ML model

## FISHER

This unlearning method is based on the paper
> Aditya Golatkar, Alessandro Achille, Stefano Soatto [Eternal Sunshine of the Spotless Net : Selective forgetting in Deep Networks](https://arxiv.org/abs/1911.04933) CVPR 2020.

The code for the method is present in [scrub.py](scrub.py) which contains the methods to scrub deleted information from trained models as described in the paper.
