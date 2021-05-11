
import resource
from copy import deepcopy
from tiny_model import TNN
import numpy as np
import logging
import torch

class selfTrain():
    logger = logging.getLogger("selfTrain")

    def __init__(self,base_estimator, tol = 0.95):
        self.base_estimator = base_estimator
        self.tol = tol
    
    def fit(self,X,y):
        """
        X has data from both
        y has labels for all, but -1 is for no label
        """
        self.logger.info('fit method called')
        assert X.shape[0] == y.shape[0]
        estimator = deepcopy(self.base_estimator)

        mask = y == -1
        prev = 0
        unlabelled = sum(mask)

        while prev != unlabelled:
            self.logger.info("Start iteration")
            labelled_X = X[torch.logical_not(mask),:]
            self.logger.info("First works")
            labels = y[~mask]
            self.logger.info("Fitting estimator")
            estimator.fit(labelled_X, labels, E=5)
            predictions = estimator.predict_proba(X[mask])
            to_label = predictions > self.tol or predictions < (1-self.tol)
            y[mask][to_label] = predictions[to_label]

            prev = unlabelled
            mask = y == -1
            unlabelled = sum(mask)
            self.logger.info(f"# of unlabelled data (prev,now): ({prev} , {unlabelled})")
            
def main():
    clf = TNN()
    ST = selfTrain(clf)
    source = torch.tensor(np.load('data/npys/Books.npy')).float()
    target = torch.tensor(np.load('data/npys/Books.npy')).float()
    ST.logger.info('Joining data')
    D = torch.cat((source,target),dim=0)
    del source, target
    memo = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ST.logger.info(f'{memo} bytes of memory is in use')
    X, y = D[:,:128].float() , D[:,-1].float()

    ST.fit(X,y)


if __name__ == '__main__':
    main()