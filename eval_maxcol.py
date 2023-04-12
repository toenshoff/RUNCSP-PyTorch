import torch
import numpy as np
from torch.utils.data import DataLoader

from model import RUNCSP
from csp_data import CSP_Data
from eval import evaluate

from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/maxcol/test', help="Model directory")
    parser.add_argument("--data_path", type=str, default='data/3COL_100_Train/positive/*.dimacs', help="Path to the training data")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")

    parser.add_argument("--num_boost", type=int, default=64, help="Number of parallel runs")
    parser.add_argument("--network_steps", type=int, default=10000000, help="Number of network steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RUNCSP.load(args.model_dir)
    model.to(device)
    model.eval()

    print(f'Loading Graphs from {args.data_path}...')
    data = [CSP_Data.load_graph_maxcol(p, model.const_lang.domain_size) for p in tqdm(glob(args.data_path))]
    const_lang = data[0].const_lang

    loader = DataLoader(
        data,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=CSP_Data.collate
    )

    evaluate(model, loader, device, args)
