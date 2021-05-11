
from copy import deepcopy
from tiny_model import TNN
import numpy as np
import logging
import torch

logger = logging.getLogger("selfTrain")
class selfTrain():

    def __init__(self,base_estimator, tol = 0.95):
        self.base_estimator = base_estimator
        self.tol = tol
    
    def fit(self,X,y):
        """
        X has data from both
        y has labels for all, but -1 is for no label
        """
        estimator = deepcopy(self.base_estimator)

        mask = y == -1
        prev = 0
        unlabelled = sum(mask)

        while prev != unlabelled:
            logger.info("START ITERATION")
            labelled_X = X[~mask]
            labels = y[~mask]
            logger.info("START ESTIMATOR FIT")
            estimator.fit(labelled_X, labels, E=5)
            predictions = estimator.predict_proba(X[mask])
            to_label = predictions > self.tol or predictions < (1-self.tol)
            y[mask][to_label] = predictions[to_label]

            prev = unlabelled
            mask = y == -1
            unlabelled = sum(mask)
            logger.info(f"{prev} , {unlabelled}")
            
def main():
    clf = TNN(128)
    ST = selfTrain(clf)
    source = torch.tensor(np.load('data/npys/Books.npy'))
    target = torch.tensor(np.load('data/npys/All_Beauty.npy'))
    X = torch.cat((source[:,:128],target[:,:128]),dim=0)
    y = torch.cat((source[:,-1],-torch.ones([target.shape[0]])),dim=0)
    ST.fit(X,y)


if __name__ == '__main__':
    main()