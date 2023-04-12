import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import RUNCSP
from loss import csp_loss
from csp_data import CSP_Data

from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

from train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/maxcol/test', help="Model directory")
    parser.add_argument("--data_path", type=str, default='data/3COL_100_Train/*/*.dimacs', help="Path to the training data")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of loader workers")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

    parser.add_argument("--batch_size", type=int, default=10, help="The batch size used for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")

    parser.add_argument("--discount", type=float, default=0.9, help="Discount factor")

    parser.add_argument("--num_col", type=int, default=3, help="Number of Colors")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden Dimension of the network")
    parser.add_argument("--network_steps", type=int, default=32, help="Number of network steps during training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading Graphs from {args.data_path}...')
    data = [CSP_Data.load_graph_maxcol(p, args.num_col) for p in tqdm(glob(args.data_path))]
    const_lang = data[0].const_lang

    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=CSP_Data.collate
    )

    model = RUNCSP(args.model_dir, args.hidden_dim, const_lang)
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(model, opt, loader, device, args)
