import numpy as np
import torch
import argparse
import time
from util import *
import random

def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set as {seed}")

seed_torch(2023)

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
args = parser.parse_args()
torch.set_num_threads(3)

device = torch.device(args.device)
dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']

predefined_A = load_adj(args.adj_data)
predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
predefined_A = predefined_A.to(device)

