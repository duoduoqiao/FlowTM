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
from Models.inspection_nets import Inspector, Inspector3D
from Utils.function_utils import MMD


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
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
        warnings.warn('You have chosen to seed training.'
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def parse_args():
    parser = argparse.ArgumentParser(description="FlowTM+", add_help=False)

    # args for dataset
    parser.add_argument('--data-root', type=str, default="./Data",
                        help="Root dir of .csv file")
    parser.add_argument('--run-name', type=str, default="Output", help='The folder name used to save model,'
                        ' output and evaluation metrics. This can be set to any word')
    parser.add_argument('--dataset', type=str, default="abilene",
                        choices=["abilene", "geant", "cernet"], help="The dataset name")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--rm-fn', type=str, default="abilene_rm.csv", 
                        help="Route Matrix (Deterministic or Probabilistic Routing)")
    parser.add_argument('--window', type=float, default=12,
                        help="Window size")
    
    # args for models
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout Ratio")
    parser.add_argument('--hd-dims', type=int, default=144,
                        help="Embedding Dimension Size")
    parser.add_argument('--n-blocks', type=int, default=5,
                        help="Number of Stacked Flow Blocks")
    parser.add_argument('--shuffle', action='store_true',
                        help="Using Shuffle Operations in Coupling Blocks")

    # args for training
    parser.add_argument('--train-epoch', type=int, default=200,
                        help="Training Epoch")
    parser.add_argument('--loss-type', type=str, default="l1norm",
                        choices=["l1norm", "mse"],
                        help="Loss Function for Training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Training Learning Rate")
    parser.add_argument('--lr-ae', type=float, default=1e-2,
                        help="Training Learning Rate of AE")
    parser.add_argument('--lr-ins', type=float, default=1e-2,
                        help="Training Learning Rate of Inspector3D")
    parser.add_argument('--lambda-inv', type=float, default=0.5,
                        help="Loss Weight 1")   # L_{inv}, Loss_1
    parser.add_argument('--lambda-backward', type=float, default=0.5,
                        help="Loss Weight 2")   # L_{link}, Loss_2
    parser.add_argument('--lambda-mmd-forward', type=float, default=0.5,
                        help="Loss Weight 3")   # L_{indep}, Loss_3
    parser.add_argument('--lambda-forward', type=float, default=0.,
                        help="Loss Weight 4")   # L_{est}_FlowTM, Loss_4
    parser.add_argument('--lambda-rec', type=float, default=1.,
                        help="Loss Weight 5")   # L_{est}_FlowTM+, Loss_5
    parser.add_argument('--lambda-gen', type=float, default=1.,
                        help="Loss Weight 6")   # L_{gen}, Loss_6
    parser.add_argument('--lambda-gp', type=float, default=0.1,
                        help="Weight of gridient penalty")
    parser.add_argument('--known-rate', type=float, default=0.02,
                        help='The known rate of ground truth.')
    parser.add_argument('--use-conv3d', action="store_true",
                        help='Whether to use 3D Convolutional Inspector.')
    parser.add_argument('--eval', action="store_true",
                        help='Whether to perform evaluation after training')
    
    # args for random
    parser.add_argument('--cudnn-deterministic', action='store_true', 
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=123,
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

    train_loader, test_loader = load_data(args)

    ae_model = AutoEncoder(feature_dim=train_loader.dataset.dim_2, hidden_dim=args.hd_dims, dropout=args.dropout)
    model = FlowNet(block_num=args.n_blocks, size_1=train_loader.dataset.dim_1, size_2=args.hd_dims, shuffle=args.shuffle)
    if args.use_conv3d:
        inspector = Inspector3D(args.dataset)
    else:
        inspector = Inspector(args.dataset)

    solver = Solver(args, ae_model, inspector, model, train_loader, results_folder=os.path.join(args.run_name, f'flow_ckpt'))

    if args.eval:
        solver.load()
    else:
        solver.train()

    estimations, reals = solver.estimate(data_loader=test_loader)
    np.save(os.path.join(args.run_name, f'{args.dataset}_gt.npy'), reals)
    np.save(os.path.join(args.run_name, f'{args.dataset}_estimation.npy'), estimations)

    loss_nmae = np.abs(estimations - reals).sum() / np.abs(reals).sum()
    loss_nrmse = np.sqrt(np.square(estimations - reals).sum()) / np.sqrt(np.square(reals).sum())
    loss_mmd = MMD(estimations, reals, 2)
    print(f'NMAE: {loss_nmae}, NRMSE: {loss_nrmse}, MMD: {loss_mmd}')

    return loss_nmae, loss_nrmse, loss_mmd


if __name__ == "__main__":
    args = parse_args()
    main(args)