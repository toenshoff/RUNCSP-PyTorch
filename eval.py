import torch
from csp_data import CSP_Data
from timeit import default_timer as timer


def evaluate(model, loader, device, args):

    with torch.inference_mode():
        for data in loader:
            start = timer()
            path = data.path
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
            data.to(device)

            assignment = model(data, args.network_steps)
            num_unsat = data.count_unsat(assignment)
            min_unsat = num_unsat.min().cpu().numpy()
            solved = min_unsat == 0

            end = timer()
            time = end - start

            print(f'{path} -- Num Unsat: {min_unsat}')
            print(f'{"Solved" if solved else "Unsolved"} {time:.2f}s')
