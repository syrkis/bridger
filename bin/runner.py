from self_training import selfTrain
from sentiment import RNN
from tiny_model import TNN
import torch
import json
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
import sys, os
import gc
import tracemalloc

def metric(val1,val2):
    return np.mean([val1,val2])


def bridge_KLD(source,bridge,target):
    with open("data/pair_kld.json","r") as infil:
        dic = json.load(infil)
    val1 = dic[bridge][source]
    val2 = dic[target][bridge]
    return metric(val1,val2)
    

def run_bridge(source_name,bridge_name,target_name,ST):
    curr_results = {}
    
    source = torch.tensor(np.load(f"data/small/{source_name}.npy"))
    bridge = torch.tensor(np.load(f"data/small/{bridge_name}.npy"))
    target = torch.tensor(np.load(f"data/small/{target_name}.npy"))
    print(bridge)
    bridge_labels = deepcopy(bridge[:,-1])
    bridge[:,-1] = -1
    target_labels = deepcopy(target[:,-1])
    target[:,-1] = -1
    D1 = torch.cat((source,bridge),dim=0)
    X1, y1 = D1[:,:128] , D1[:,-1].float()
    ST.fit(X1,y1)
    f1 = ST.score(bridge[:,:128],bridge_labels)
    curr_results['f1_bridge'] = f1
    
    bridge_mask = ST.ass_labels_ != -1
    
    Xassigned = bridge[:,:128][bridge_mask]
    yassigned = ST.ass_labels_[bridge_mask]
    Xtarget, ytarget = target[:,:128] , target[:,-1].float()
    X2 = torch.cat((Xassigned,Xtarget),dim=0)
    y2 = torch.cat((yassigned,ytarget),dim=0)
    
    ST.fit(X2,y2)
    f1 = ST.score(Xtarget[:,:128],target_labels)
    curr_results['f1_target'] = f1
    del ST.estimator_
    gc.collect()
    return curr_results



def main():
    results = {}
    logger = logging.getLogger('runner')
    source_name = 'Movies_and_TV'
    target_name = 'Pet_Supplies'
    source = torch.tensor(np.load(f"data/small/{source_name}.npy"))
    target = torch.tensor(np.load(f"data/small/{target_name}.npy"))
    model = RNN()
    ST = selfTrain(model)
    true_target_lab = deepcopy(target[:,-1])
    target[:,-1] = -1
    logger.info('Joining data')
    D = torch.cat((source,target),dim=0)
    X, y = D[:,:128], D[:,-1].float()
    ST.fit(X,y)
    AC_score = ST.score(target[:,:128],true_target_lab)
    with open("data/pair_kld.json","r") as infil:
        dic = json.load(infil)
        KLD = dic[target_name][source_name]
    results[source_name + "__" + target_name] = {"Direct_score":AC_score, 'KLD': KLD}

    bridges = os.listdir('data/small')
    values = []
    for bridge in bridges:
        bridge = bridge[:-4]
        if bridge == source_name or bridge == target_name:
            continue
        bridge_metric = bridge_KLD(source_name, bridge, target_name)
        values.append((bridge_metric,bridge))
    values.sort()
    
    gc.collect()
    tracemalloc.start()
    for bridge_metric, bridge in values:
        logger.info(f"Starting run with {bridge}")
        bridge_results = run_bridge(source_name, bridge, target_name, ST)
        results[bridge] = bridge_results
        results[bridge]["bridge_KLD"] = bridge_metric
        current,peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current/10**6}MB. Peak memory usage: {peak/10**6}MB")
    tracemalloc.stop()
    
    if not os.path.isdir("data/results_small"):
        os.mkdir("data/results_small")

    with open(f"data/results_small/{source_name}__{target_name}.json","w") as of:
        json.dump(results,of, indent=2)
      

        


if __name__ == "__main__":
    main()
