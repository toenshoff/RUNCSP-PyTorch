from data_utils import write_dimacs_graph
import os
import numpy as np
import networkx as nx
from argparse import ArgumentParser

from tqdm import tqdm


def generate_gnm_graph(k):
    n = np.random.randint(50, 101)
    m = np.random.randint(2*n, n*k)
    G = nx.gnm_random_graph(n,m)
    return G


def generate_geo_graph(k):
    n = np.random.randint(80, 28*k)
    r = np.random.uniform(0.1, 0.2)
    G = nx.random_geometric_graph(n, r)
    return G


def generate_pwl_graph(k):
    n = np.random.randint(20, 20*k)
    m = np.random.randint(1, 4)
    p = np.random.uniform(0.0, 1.0)
    G = nx.powerlaw_cluster_graph(n, m, p)
    return G


def generate_cc_graph(k):
    l = np.random.randint(10, 20)
    k = np.random.randint(max(4, k-2), k+3)
    G = nx.caveman_graph(l, k)
    return G


def generate_graph(k):
    t = np.random.randint(4)
    if t == 0:
        G = generate_gnm_graph(k)
    elif t == 1:
        G = generate_geo_graph(k)
    elif t == 2:
        G = generate_pwl_graph(k)
    else:
        G = generate_cc_graph(k)
    return G


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--data_path", type=str, default='data/K-Col-Graphs', help="Path to the training data")
    parser.add_argument("--k", type=int, default=17, help="Number of colours")
    parser.add_argument("--num_graphs", type=int, default=4000, help="Number of graphs")
    args = parser.parse_args()
    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.data_path, f'{args.k}-Col'), exist_ok=True)

    for i in tqdm(range(args.num_graphs)):
        G = generate_graph(args.k)
        path = os.path.join(args.data_path, f'{args.k}-Col', f'{i}.dimacs')
        write_dimacs_graph(G, path)
