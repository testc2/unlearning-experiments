import argparse 
from pathlib import Path

from traitlets.traitlets import default

project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / "data"

def add_dist_subparser(subparser):
    dist_parser = subparser.add_parser("dist")
    dist_parser.add_argument(
        "dataset",
        type=str,
        default="MNIST",
        choices=["MNIST","COVTYPE","HIGGS","CIFAR","EPSILON"]
    )
    dist_parser.add_argument(
        "--ovr",
        action="store_true",
        default=False,
        help="if multi-class or binary"
    )
    dist_parser.add_argument(
        "--num-processes",
        type=int,
        default=4
    )
    dist_parser.add_argument(
        "--sampling-types",
        type=str,
        nargs="+",
        choices=["uniform_random","uniform_informed","targeted_random","targeted_informed"],
        default=["uniform_random","uniform_informed","targeted_random","targeted_informed"]
    )
    dist_parser.add_argument(
        "--remove-ratios",
        type=float,
        nargs="+",
        default=[0.10]
    )
    dist_parser.add_argument(
        "--num-sample_prob",
        type = int,
        default = 10,
        help = "Number of probabilities from 1/k to 1 for sampler"
    )
    dist_parser.add_argument(
        "--minibatch-fractions",
        type=int,
        nargs="+",
        help="The size of minibatch in terms of fraction of the removed samples",
        default=[1]
    )
    dist_parser.add_argument(
        "--sgd-seed",
        type=int,
        nargs="+",
        default=[0],
        help="seeds for adam"
    )
    dist_parser.add_argument(
        "--num-sampler-seed",
        type = int,
        default = 2,
        help = "Number of seeds for sampler"
    )
    dist_parser.add_argument(
        "--remove-class",
        type = int,
        default = 0,
        help = "Class to target for sample removal"
    )
    dist_parser.add_argument(
        "--overwrite-mode",
        type=str,
        default="w",
        choices=["w", "a"]
    )
    dist_parser.add_argument(
        "--l2-norm",
        action="store_true",
        default=False,
        help="Whether to L2 normalize the data or not"
    )
    dist_parser.add_argument(
        "--results-dir",
        type=Path,
        default=data_dir / "results"
    )
    dist_parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix to file"
    )

def add_perturb_subparser(subparser):
    perturb_parser = subparser.add_parser("perturb")
    perturb_parser.add_argument(
        "dataset",
        type=str,
        default="MNIST",
        choices=["MNIST","COVTYPE","HIGGS","CIFAR","EPSILON"]
    )
    perturb_parser.add_argument(
        "--ovr",
        action="store_true",
        default=False,
        help="if multi-class or binary"
    )
    perturb_parser.add_argument(
        "--num-processes",
        type=int,
        default=4
    )
    perturb_parser.add_argument(
        "--num-seeds",
        type=int,
        default=6
    )
    perturb_parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0,0.01,0.1,1,10,100]
    )
    perturb_parser.add_argument(
        "--results-dir",
        type=Path,
        default=data_dir / "results"
    )
    perturb_parser.add_argument(
        "--overwrite-mode",
        type=str,
        default="w",
        choices=["w", "a"]
    )
    perturb_parser.add_argument(
        "--sgd-seed",
        type=int,
        nargs="+",
        default=[0],
        help="seeds for optimizer"
    )
    perturb_parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix to file"
    )
    perturb_parser.add_argument(
        "--l2-norm",
        action="store_true",
        default=False,
        help="Whether to L2 normalize the data or not"
    )
    return perturb_parser

def add_ratio_subparser(subparser):
    ratio_parser = subparser.add_parser("ratio")
    ratio_parser.add_argument(
        "dataset",
        type=str,
        default="MNIST",
        choices=["MNIST","COVTYPE","HIGGS","CIFAR","EPSILON"]
    )  
    ratio_parser.add_argument(
        "--ovr",
        action="store_true",
        default=False,
        help="if multi-class or binary"
    )
    ratio_parser.add_argument(
        "--num-processes",
        type=int,
        default=4
    )
    ratio_parser.add_argument(
        "--sgd-seed",
        type=int,
        nargs="+",
        default=[0],
        help="seeds for adam"
    )
    ratio_parser.add_argument(
        "--num-sampler-seed",
        type = int,
        default = 1,
        help = "Number of seeds for sampler"
    )
    ratio_parser.add_argument(
        "--overwrite-mode",
        type=str,
        default="w",
        choices=["w", "a"]
    )
    ratio_parser.add_argument(
        "--l2-norm",
        action="store_true",
        default=False,
        help="Whether to L2 normalize the data or not"
    )
    ratio_parser.add_argument(
        "--results-dir",
        type=Path,
        default=data_dir / "results"
    )
    ratio_parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix to file"
    )

def add_unlearn_subparser(subparser):
    unlearn_parser = subparser.add_parser("unlearn")
    unlearn_parser.add_argument(
        "dataset",
        type=str,
        default="MNIST",
        choices=["MNIST","COVTYPE","HIGGS","CIFAR","EPSILON"]
    )
    unlearn_parser.add_argument(
        "--ovr",
        action="store_true",
        default=False,
        help="if multi-class or binary"
    )
    unlearn_parser.add_argument(
        "--num-processes",
        type=int,
        default=4
    )
    unlearn_parser.add_argument(
        "--sgd-seed",
        type=int,
        nargs="+",
        default=[0],
        help="seeds for adam"
    )
    # noise args 
    unlearn_parser.add_argument(
        "--num-noise-seeds",
        type=int,
        default=6
    )
    unlearn_parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.01,0.1,1,10,100]
    )
    # Removal Step args
    unlearn_parser.add_argument(
        "--minibatch-fractions",
        type=int,
        nargs="+",
        help="The size of minibatch in terms of fraction of the removed samples",
        default=[1]
    )
    unlearn_parser.add_argument(
        "--remove-ratios",
        type=float,
        nargs="+",
        default=[0.10]
    )
    # General arguments
    unlearn_parser.add_argument(
        "--overwrite-mode",
        type=str,
        default="w",
        choices=["w", "a"]
    )
    unlearn_parser.add_argument(
        "--l2-norm",
        action="store_true",
        default=False,
        help="Whether to L2 normalize the data or not"
    )
    unlearn_parser.add_argument(
        "--results-dir",
        type=Path,
        default=data_dir / "results"
    )
    unlearn_parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix to file"
    )

    return unlearn_parser