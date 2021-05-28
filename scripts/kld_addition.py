import json
import numpy as np
import os
import sys

sys.path.append('bin/')
from runner import bridge_KLD

KLDs = []
dists = []
all_results = os.listdir('data/results')
for fil in all_results:
    with open(f'data/results/{fil}', 'r') as of:
        data = json.load(of)
    source, target = list(data.keys())[0].split('__')
    for bridge in list(data.keys())[1:]:
        print(source, bridge, target)
        val = bridge_KLD(source, bridge, target)
        print(val)
        data[bridge]['harmonic'] = val
    with open(f'data/results/{fil}','w') as of:
        json.dump(data,of,indent=2)
        

    