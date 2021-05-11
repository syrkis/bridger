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


# declare nn
class TNN(nn.Module):
    logger = logging.getLogger('TNN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    batch_size = 128
    sentence_len = 128

    def __init__(self):
        super(TNN, self).__init__()
        self.best_state_dict = {}
        self.state_dict_score = float('inf')
        self.train_loss = []
        self.dev_loss = []

        # Layers
        self.lin = nn.Linear(self.sentence_len, 1)
        self.sigmoid = nn.Sigmoid()

        self.logger.info(f"Model is using '{self.device}' for training")
        self.to(self.device)

    def forward(self, X):
        N_samples = X.shape[0]
        output = self.lin(X)
        output = self.sigmoid(output)
        return output.reshape(N_samples) # Vector/Tensor of shape (N_samples)

    def fit(self, X, y, E = 1, dev_size = None):
        assert X.shape[0] == y.shape[0]
        self.logger.info("Started training")
        if dev_size:
            X, x_dev, y, y_dev = train_test_split(X,y, test_size=dev_size)

        optimizer = Adam(self.parameters())
        loss = nn.BCELoss()

        self.train_loss.append(loss(self.predict_proba(X), y).item())
        if dev_size:
            self.dev_loss.append(loss(self.predict_proba(x_dev), y_dev).item())

        X_batches = torch.split(X,self.batch_size)
        if X_batches[-1].shape[0] != self.batch_size:
            X_batches = X_batches[:-1]
        y_batches = torch.split(y,self.batch_size)

        for e in range(E):
            losses = []
            for i in tqdm(range(len(X_batches))):
                X_batch = X_batches[i].to(self.device)
                y_batch = y_batches[i].to(self.device)
                optimizer.zero_grad()
                pred = self.forward(X_batch)
                cross_entropy_loss = loss(pred, y_batch)
                cross_entropy_loss.backward()    
                optimizer.step() 
                losses.append(cross_entropy_loss.item())
                torch.cuda.empty_cache()

            self.logger.info(f"Epoch {e} loss: {np.mean(losses)}")
            self.train_loss.append(np.mean(losses))
            if dev_size:
                self.dev_loss.append(loss(self.predict_proba(x_dev),y_dev).item())
                if self.dev_loss[-1] > self.state_dict_score:
                    self.state_dict_score = self.dev_loss[-1]
                    self.best_state_dict = deepcopy(self.state_dict())

        return

    def predict(self,X):
        return self.predict_proba(X) > 0.5
        
    def predict_proba(self,X):
        self.eval() # sets model in evaluation mode, to not use dropout
        with torch.no_grad():
            if str(self.device) == 'cuda':
                res = torch.cuda.memory_reserved(0)
                alo = torch.cuda.memory_allocated(0)
                free_mem = res - alo
                row_mem = X.element_size() * X.shape[1]
                pred_batch_size = int((free_mem*0.9) //row_mem)
                self.logger.info(f"Predicting with batches of size: {pred_batch_size}")
            else:
                pred_batch_size = 2**11

            X_batches = torch.split(X,pred_batch_size)
            probas = []
            for batch in tqdm(X_batches):
                prob = self.forward(batch.to(self.device)).cpu()
                probas.append(prob)
                torch.cuda.empty_cache()
            self.train() # sets model back to training mode which is default
            return torch.cat(probas,dim=0) 

    def save(self, path, save_current_state = False):
        if save_current_state or not self.best_state_dict:
            torch.save(self.state_dict(), path)
            return
        torch.save(self.best_state_dict, path)

    def save_training_losses(self,path):
        assert self.train_loss
        with open(path,"w") as of:
            of.write(','.join(map(str,self.train_loss)))
            if self.dev_loss:
                of.write('\n')
                of.write(','.join(map(str,self.dev_loss)))
        
      
def main(): 
    D = torch.tensor(np.load('data/npys/Books.npy'))
    x, y = D[:,:128].float() , D[:,-1].float()
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=10**5)
    model = TNN()
    model.fit(x_train, y_train, E=10, dev_size = 0.1)
    model.save_training_losses('data/train_dev_loss.csv')
    model.save('data/models/RNN025.pt')
    P = model.predict(x_test)
    print("\n",classification_report(y_test, P),"\n")

if __name__ == '__main__':
    main()
