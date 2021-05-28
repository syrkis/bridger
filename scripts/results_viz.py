import matplotlib.pyplot as plt
from scipy import stats
import json
import numpy as np
import os

KLDs = []
dists = []
all_results = os.listdir('../data/results')
for fil in all_results:
    with open(f'../data/results/{fil}', 'r') as of:
        data = json.load(of)
    base_f1 = list(data.values())[0]['Direct_score']
    base_KLD = list(data.values())[0]['KLD']
    
    for entry in list(data.keys())[1:]:
        dists.append(data[entry]['f1_target'] - base_f1)

        KLDs.append(data[entry]['max_KLD'] < base_KLD)

KLDs = np.array(KLDs)
dists = np.array(dists)
print(stats.describe(dists))

