from self_training import selfTrain
from sentiment import RNN
import torch
import numpy as np
import sys, os

def runner():


def main():
    source_name = sys.argv[1]
    target_name = sys.argv[2]
    source = torch.tensor(np.load(f"data/npys/{source_name}"))
    target = torch.tensor(np.load(f"data/npys/{target_name}"))
    target = torch.tensor(np.load(target_name))
    bridges = os.listdir('data/npys')
      

    for bridge in bridges:
        selfTrain(RNN)
        


if __name__ == "__main__":
    pass