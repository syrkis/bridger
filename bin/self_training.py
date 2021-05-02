# self_training.py
#	framework for self training
# by: Group 2

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.semi_supervised import SelfTrainingClassifier

from sklearn.svm import SVC

import numpy as np
from icecream import ic
from copy import deepcopy
import os

import sentiment

SEQUENCE_LENGTH = 128
PSUEDO_SELF_TRAIN = False


class RNNEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrapper for the RNN model implementing a variation of the scikit-learn API.
    Specifically, the re_fit method is a variation of fit that violates the standard
    convention of ignoring previous calls to the method.
    """

    def __init__(self,sequence_length=1):
        self.sequence_length = sequence_length
        self.base_model = sentiment.RNN(self.sequence_length)

    def fit(self,X,y):
        X,y = check_X_y(X,y)
        self.estimator_ = deepcopy(self.base_model)
        self.estimator_.fit(X,y,E=1)
        return self

    def re_fit(self,X,y):
        X,y = check_X_y(X,y)
        try:
            check_is_fitted(self)
        except NotFittedError:
            self.estimator_ = deepcopy(self.base_model)
        
        self.estimator_.fit(X,y,E=1)
        return self

    def predict(self,X):
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.forward(X) >= 0.5

    def predict_proba(self,X):
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.forward(X)


class PseudoSelfTrainingClassifier(SelfTrainingClassifier):
    def __init__(self, *args, **kwargs):
        super(PseudoSelfTrainingClassifier, self).__init__(*args, **kwargs)
        self.base_estimator = clone(self.base_estimator)
        self.base_estimator.fit = self.base_estimator.re_fit


def self_train(estimator,l_X,l_y,u_X,):
    STC = SelfTrainingClassifier(estimator,max_iter=10)
    if PSUEDO_SELF_TRAIN:
        STC = PseudoSelfTrainingClassifier(estimator,max_iter=None)

    u_y = np.array(u_X.shape[0]*[-1])
    X = np.concatenate((l_X,u_X))
    y = np.concatenate((l_y,u_y))
    STC.fit(X,y)
    return STC.base_estimator_


def main():
    data_dir = '../data/npys'
    samples = os.listdir(data_dir)
    with open(f"{data_dir}/{samples[0]}", 'rb') as f:
        print(f"Labelled data from {samples[0]}")
        D = np.load(f)
        X, y = D[: , :SEQUENCE_LENGTH], D[: , -1]

    with open(f"{data_dir}/{samples[1]}", 'rb') as f:
        print(f"Unlabelled data from {samples[1]}")
        D = np.load(f)
        u_X, u_y = D[: , :SEQUENCE_LENGTH], D[: , -1]

    model = SVC(probability=True, gamma="auto")
    #model = RNNEstimator(sequence_length=SEQUENCE_LENGTH)
    self_trained_model = self_train(model,X,y,u_X)

    ic(self_trained_model.predict_proba(X[:32]))


if __name__ == '__main__':
    main()