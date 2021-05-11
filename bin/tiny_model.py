# sentiment.py
#   rnn for sentiment classification
# by: Group 2

# imports
import torch
from torch import nn
import sys
from torch.optim import Adam
import random
import numpy as np
from tqdm import tqdm
import os
import time
from copy import deepcopy
import json
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger('model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# declare nn
class RNN(nn.Module):
        
    batch_size = 128

    def __init__(self, sentence_len):
        super(RNN, self).__init__()
        self.best_state_dict = {}
        self.state_dict_score = float('inf')
        self.sentence_len = sentence_len
        
        self.lin = nn.Linear(self.sentence_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        N_samples = X.shape[0]
        X = X.float()
        output = self.lin(X)
        output = self.sigmoid(output)
        return output.reshape(N_samples) # Vector/Tensor of shape (N_samples)

    def fit(self, X, y, x_dev = None, y_dev = None, E = 1):
        assert X.shape[0] == y.shape[0]
        assert type(x_dev) == type(y_dev)
        logger.info(X.shape)
        optimizer = Adam(self.parameters())
        loss = nn.BCELoss()

        logger.info("INITIAL LOSS")
        L = [loss(self.predict_proba(X), y).item()]
        if x_dev is not None:
            L_dev = [loss(self.predict_proba(x_dev), y_dev).item()]
        logger.info("FINISH INITIAL LOSS")

        X_batches = torch.split(X,RNN.batch_size)
        if X_batches[-1].shape[0] != RNN.batch_size:
            X_batches = X_batches[:-1]
        y_batches = torch.split(y,RNN.batch_size)

        for e in range(E):
            losses = []
            for i in tqdm(range(len(X_batches))):
                X_batch = X_batches[i].to(device)
                y_batch = y_batches[i].to(device)
                optimizer.zero_grad()
                pred = self.forward(X_batch)
                cross_entropy_loss = loss(pred, y_batch)
                cross_entropy_loss.backward()    
                optimizer.step() 
                losses.append(cross_entropy_loss.item())
                torch.cuda.empty_cache()

            logger.info(f"EPOCH {e} LOSS: {np.mean(losses)}")
            L.append(np.mean(losses))
            if x_dev is not None:
                L_dev.append(loss(self.predict_proba(x_dev),y_dev).item())
                if L_dev[-1] > self.state_dict_score:
                    self.state_dict_score = L_dev[-1]
                    self.best_state_dict = deepcopy(self.state_dict())

        if x_dev is not None:
            return L, L_dev
        return L

    def predict(self,X):
        return self.predict_proba(X) > 0.5
        
    def predict_proba(self,X):
        self.eval() # sets model in evaluation mode, to not use dropout
        with torch.no_grad():
            if str(device) == 'cuda':
                res = torch.cuda.memory_reserved(0)
                alo = torch.cuda.memory_allocated(0)
                free_mem = res - alo
                logger.info(f"free memory {free_mem}")
                row_mem = X.element_size() * X.shape[1]
                pred_batch_size = int((free_mem*0.9) //row_mem)
                logger.info(f"batch size for predictions is: {pred_batch_size}")
                print(f"batch size for predictions is: {pred_batch_size}")
            else:
                pred_batch_size = 2**11
            X_batches = torch.split(X,pred_batch_size)
            probas = []
            for batch in tqdm(X_batches):
                prob = self.forward(batch.to(device)).cpu()
                probas.append(prob)
                torch.cuda.empty_cache()
            self.train() # sets model back to training mode which is default
            return torch.cat(probas,dim=0) 

    def save(path, save_current_state = False):
        if save_current_state or not self.best_state_dict:
            torch.save(self.state_dict(), path)
            return
        torch.save(self.best_state_dict, path)
        
      
def main(): 
    logger.info(f'is using {device}')
    D = torch.tensor(np.load('data/npys/Books.npy'))
    x, y = D[:,:128] , D[:,-1].float()
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=10**5)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train,y_train, test_size=10**5)
    model = RNN(128).to(device) 
    L, L_dev = model.fit(x_train, y_train, x_dev, y_dev, E=7)
    try:
        logger.info(f"L is: {str(L)}")
        logger.info(f"L_dev is: {str(L_dev)}")
        with open('data/tiny_model_loss.csv',"w") as of:
            of.write(",".join(map(str,L))) 
            of.write("\n")
            of.write(",".join(map(str,L_dev))) 
        model.save('data/models/RNN025.pt')
    except Exception as e:
        logger.info('error occurred:\n\t-' + str(e))
        pass
    P = model.predict(x_test)
    print("\n",classification_report(y_test, P),"\n")

if __name__ == '__main__':
    main()
