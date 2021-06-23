#%%
# some code to import autoreload if used as a notebook
from IPython import get_ipython
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False

from methods.common_utils import load_cifar, load_covtype, load_epsilon, load_higgs, load_mnist
import argparse
from pathlib import Path
from experiments.removal_distribution import execute_distribution_exp
from experiments.unlearn import execute_unlearning_exp
from experiments.perturbation import execute_perturbation_exp
from experiments.removal_ratio import execute_ratio_exp
from experiments.args import add_perturb_subparser, add_ratio_subparser, add_dist_subparser, add_unlearn_subparser

# set project dir relative to the file 
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

# add the SGD optimizer args
parser = argparse.ArgumentParser(description="Optimizer arguments")
parser.add_argument("--optim", type=str, default="Adam", choices=["Adam","SGD"], help="Optimizer to use")
parser.add_argument("--lam", type=float, default=1e-4, help="L2 regularization")
parser.add_argument("--step-size", type=float, default=1e-1, help="Adam Step Size")
parser.add_argument("--lr-schedule", action="store_true", help="Whether to schedule learning rate")
parser.add_argument("--batch-size", type=int, default=64, help="What batch size to use for training")
parser.add_argument("--num-steps", type=int, default=1000, help="number of optimization steps")
parser.add_argument("--verbose", action="store_true", default=False, help="verbosity in optimizer")
# create a subparse for the experiments 
subparser = parser.add_subparsers(description="Additional arguments for the experiments",dest="experiment")
# add the args for the experiments to the subparser
dist_parser = add_dist_subparser(subparser)
perturb_parser = add_perturb_subparser(subparser)
ratio_parser = add_ratio_subparser(subparser)
unlearn_parser = add_unlearn_subparser(subparser)
#%%
if __name__ == "__main__":
    # to test the script in the interactive notebook
    if notebook:
        args = parser.parse_args(["dist", "MNIST","--l2-norm"])
    else:
        args = parser.parse_args()
    
    # load the dataset based on the args 
    if args.dataset == "MNIST":
            X_train, X_test, y_train, y_test = load_mnist(data_dir,ovr=args.ovr,l2_norm=args.l2_norm)
    elif args.dataset == "COVTYPE":
        X_train, X_test, y_train, y_test = load_covtype(data_dir,ovr=args.ovr,l2_norm=args.l2_norm)
    elif args.dataset == "HIGGS":
        X_train, X_test, y_train, y_test = load_higgs(data_dir,l2_norm=args.l2_norm)
    elif args.dataset == "CIFAR":
        X_train, X_test, y_train, y_test = load_cifar(data_dir,l2_norm=args.l2_norm,ovr=args.ovr)
    elif args.dataset == "EPSILON":
        X_train, X_test, y_train, y_test = load_epsilon(data_dir,l2_norm=args.l2_norm)
    
    # place them in a dict that will be sent to the experiments
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    
    # choose the experiment based on the args

    # "remove_ratio" is the deletion distribution experiments
    if args.experiment == "ratio":
        execute_ratio_exp(args,data)
    # "dist" is the QoA parameter experiments
    elif args.experiment == "dist":
        execute_distribution_exp(args,data)
    # "perturb" is the noise injection experiments
    elif args.experiment == "perturb":
        execute_perturbation_exp(args,data)
    # "unlearn" is the complete experiments
    elif args.experiment == "unlearn":
        execute_unlearning_exp(args,data)
#%%
