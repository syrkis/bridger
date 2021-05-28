import json





with open("data/pair_kld.json","r") as infil:
    values = json.load(infil)
    best = 0
    best_pair = (None,None)
    for k1 in values:
        for k2 in values[k1]:
            if values[k1][k2] > best:
                best = values[k1][k2]
                best_pair = (k1,k2) 

    print(best_pair,best)
