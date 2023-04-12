import torch
from loss import csp_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(model, opt, loader, device, args):
    writer = SummaryWriter(args.model_dir)
    step = 0
    for e in range(args.epochs):
        num_unsat_list = []
        solved_list = []
        for data in tqdm(loader):
            opt.zero_grad()
            data.to(device)

            assignment = model(data, args.network_steps)
            loss = csp_loss(data, assignment, discount=args.discount)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            num_unsat = data.count_unsat(assignment)
            num_unsat = num_unsat.min(dim=1)[0]
            solved = num_unsat == 0
            num_unsat_list.append(num_unsat)
            solved_list.append(solved)

            if (step + 1) % args.logging_steps == 0:
                num_unsat = torch.cat(num_unsat_list, dim=0)
                solved = torch.cat(solved_list, dim=0)
                writer.add_scalar('Train/Loss', loss.mean(), step)
                writer.add_scalar('Train/Solved_Ratio', solved.float().mean(), step)
                writer.add_scalar('Train/Unsat_Count', num_unsat.float().mean(), step)
                num_unsat_list = []
                solved_list = []

            step += 1
        model.save()
