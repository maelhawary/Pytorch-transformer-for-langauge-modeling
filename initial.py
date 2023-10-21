import torch
import torch.nn as nn
import train as tr
from config import get_config


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    dir='save_models'+'/'
    config=get_config()
    tr.train(device,config,dir)