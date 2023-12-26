import os
import torch
import random
import argparse
import warnings
import numpy as np

from engine import Solver
from Utils.data_utils import load_data
from Models.flow_nets import FlowNet
from Models.embedding_net import AutoEncoder


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Experiment", add_help=False)

    # args for dataset

    parser.add_argument('--data-root', type=str, default="./Data",
                        help="Root dir of .csv file")
    parser.add_argument('--run-name', type=str, default="OUTPUT", help='The folder name used to save model,'
                        ' output and evaluation metrics. This can be set to any word')
    parser.add_argument('--dataset', type=str, default="abilene",
                        choices=["abilene", "geant"], help="The dataset name")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--rm-fn', type=str, default="abilene_rm.csv", 
                        help="Route Matrix (Deterministic or Probabilistic Routing)")
    
    # args for models

    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout Ratio")
    parser.add_argument('--hd-dims', type=int, default=100,
                        help="Embedding Dimension Size")
    parser.add_argument('--n-blocks', type=int, default=3,
                        help="Number of Stacked Flow Blocks")
    parser.add_argument('--shuffle', type=bool, default=False,
                        help="Using Shuffle Oprations in Coupling Blocks")
    
    # args for training

    parser.add_argument('--train-epoch', type=int, default=100,
                        help="Training Epoch")
    parser.add_argument('--loss-type', type=str, default="l1norm",
                        choices=["l1norm", "mse"],
                        help="Loss Function for Training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Training Learning Rate")
    parser.add_argument('--lambda-forward', type=float, default=10.,
                        help="Loss Weight 1")
    parser.add_argument('--lambda-backward', type=float, default=2.,
                        help="Loss Weight 2")
    parser.add_argument('--lambda-inv', type=float, default=2.,
                        help="Loss Weight 3")
    parser.add_argument('--lambd-mmd-backward', type=float, default=1.,
                        help="Loss Weight 4")
    parser.add_argument('--lambd-mmd-forward', type=float, default=0.5,
                        help="Loss Weight 5")
    parser.add_argument('--use-log-max', type=bool, default=False,
                        help="Use Max Likelihood Loss")
    parser.add_argument('--lambda-distribution', type=float, default=None,
                        help="Loss Weight 6")
    parser.add_argument('--eval', action="store_true", 
                        help='Whether to perform evaluation after training')
    
    # args for random

    parser.add_argument('--cudnn-deterministic', action='store_true', 
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--data-seed', type=int, default=123, 
                        help='seed for loading data')

    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)
    
    os.makedirs(args.run_name, exist_ok=True)

    train_loader, test_loader, *_ = load_data(data_root=args.data_root, dataset_name=args.dataset, batch_size=args.batch_size, random_seed=args.data_seed)

    model = FlowNet(block_num=args.n_blocks, size_1=train_loader.dataset.dim_1, size_2=args.hd_dims, shuffle=args.shuffle)
    ae_model = AutoEncoder(feature_dim=train_loader.dataset.dim_2, hidden_dim=args.hd_dims, dropout=args.dropout)

    solver = Solver(args, ae_model, model, train_loader, results_folder=os.path.join(args.run_name, f'flow_ckpt'))

    if args.eval:
        solver.load()
    else:
        solver.train()

    estimations, reals = solver.estimate(data_loader=test_loader, rm=None)
    np.save(os.path.join(args.run_name, f'{args.dataset}_gt.npy'), reals)
    np.save(os.path.join(args.run_name, f'{args.dataset}_estimation.npy'), estimations)

    loss_nmae = np.abs(estimations - reals).sum() / np.abs(reals).sum()
    loss_nrmse = np.sqrt(np.square(estimations - reals).sum()) / np.sqrt(np.square(reals).sum())
    print(f'NMAE: {loss_nmae}, NRMSE: {loss_nrmse}')


if __name__ == "__main__":
    args = parse_args()
    main(args)