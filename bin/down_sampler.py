import numpy as np
import os

files = os.listdir('data/npys')
for fil in files:
    print(fil)
    data = np.load(f'data/npys/{fil}')
    pos = data[data[:,-1] == 1]
    neg = data[data[:,-1] == 0]
    mask = np.random.randint(pos.shape[0],size = 1000)
    small = np.concatenate((pos[mask],neg[mask]),axis=0)
    np.random.shuffle(small)
    with open(f'data/small/{fil}',"wb") as of:
        np.save(of,small)