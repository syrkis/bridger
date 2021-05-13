
import resource
from copy import deepcopy
from tiny_model import TNN
from sentiment import RNN
import numpy as np
from sklearn.metrics import classification_report, f1_score
import logging
import torch

class selfTrain():
    logger = logging.getLogger("selfTrain")

    def __init__(self,base_estimator, tol = 0.95):
        self.base_estimator = base_estimator
        self.base_state_dict = deepcopy(base_estimator.state_dict())
        self.tol = tol
    
    def fit(self,X,y,max_iter=5):
        """
        X has data from both
        y has labels for all, but -1 is for no label
        """
        self.logger.info('fit method called')
        assert X.shape[0] == y.shape[0]

        mask = y == -1
        init_unlabel = deepcopy(mask)
        prev = 0
        unlabelled = torch.sum(mask).item()

        iteration = 0
        while abs(prev - unlabelled) > 500 or iteration < max_iter:
            iteration += 1
            self.logger.info(f"Start iteration {iteration}")
            self.base_estimator.load_state_dict(self.base_state_dict)
            labelled_X = X[torch.logical_not(mask),:]
            labels = y[~mask]
            self.base_estimator.fit(labelled_X, labels, E=1)
            predictions = self.base_estimator.predict_proba(X[mask])
            to_label = (predictions > self.tol) | (predictions < (1-self.tol))
            new_labels = torch.round(predictions[to_label].reshape(torch.sum(to_label).item(),1))

            y[torch.nonzero(mask)[to_label]] = new_labels

            prev = unlabelled
            mask = y == -1
            unlabelled = torch.sum(mask).item()
            self.logger.info(f"# of unlabelled data (prev,now): ({prev} , {unlabelled})")

        self.ass_labels_ = y[init_unlabel]
        self.logger.info(f"Done self training. {unlabelled} unlabelled data left")
        self.estimator_ = deepcopy(self.base_estimator)
        self.base_estimator.load_state_dict(self.base_state_dict)


    def ass_acc(self, true_labels):
        mask = self.ass_labels_ !=-1
        print("\n",classification_report(true_labels[mask], self.ass_labels_[mask]),"\n")
        
    
    def report(self,X,y):
        pred = self.estimator_.predict(X)
        print("\n",classification_report(y, pred),"\n")

    def score(self, X, y):
        pred = self.estimator_.predict(X)
        return f1_score(y,pred)

        
def main():
    clf = RNN()
    ST = selfTrain(clf)
    source = torch.tensor(np.load('data/npys/Books.npy'))
    target = torch.tensor(np.load('data/npys/All_Beauty.npy'))
    true_target_lab = deepcopy(target[:,-1])
    target[:,-1] = -1
    ST.logger.info('Joining data')
    D = torch.cat((source,target),dim=0)
    X, y = D[:,:128], D[:,-1].float()
    ST.fit(X,y)


    ST.ass_acc(true_target_lab)
    ST.report(target[:,:128],true_target_lab)


if __name__ == '__main__':
    main()