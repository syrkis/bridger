# comparer.py
#    compares two domains
# by: Group 2

# imports
import os
import json

import numpy as np
from scipy.stats import entropy


def renyi_divergence(P,Q,alpha):
    assert alpha != 1
    tmp = np.power(P,alpha) / np.power(Q,alpha-1) 
    divergence = 1/(alpha-1) * np.log(np.sum(tmp))
    return divergence


def compare_all():
    dir_path = 'data/npys'
    targets = os.listdir(dir_path)[:3]

    results = {}

    for i,f1 in enumerate(targets):
        data1 = np.load(f"data/npys/{f1}")[:,:-1]
        results[f1] = {}
        counts1 = np.bincount(data1.flatten())[:500].astype(float)
        counts1 += 0.01
        print("count1:",counts1.shape)
        for j, f2 in enumerate(targets):
            if i == j:
                continue
            data2 = np.load(f"data/npys/{f2}")[:,:-1]
            counts2 = np.bincount(data2.flatten())[:500].astype(float)
            counts2 += 0.01
            KLD = round(entropy(counts1,counts2),6)
            results[f1][f2] = KLD
    
    with open("data/pair_kld.json","w") as outf:
        json.dump(results,outf, indent = 2)


def main():
    compare_all()


if __name__ == '__main__':
    main()