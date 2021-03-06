# sentiment.py
#   rnn for sentiment classification
# by: Group 2

# imports
import torch
from torch import nn
import sys
import gensim
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
class RNN(nn.Module):
    # General
    logger = logging.getLogger('RNN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)
        
    # Model variables
    sentence_len = 128
    batch_size = 2**4

    lstm_layers = 3
    lstm_dropout = 0.25
    bidirectional = True
    lstm_hid_dim = 32

    lin_hid_dim = 250
    lin_dropout = 0.25

    def __init__(self):
        super(RNN, self).__init__()
        self.best_state_dict = {}
        self.state_dict_score = float('inf')
        self.train_loss = []
        self.dev_loss = []

        # Embedding
        emb_weights = torch.FloatTensor(self.word2vec.vectors)
        self.embedding = nn.Embedding.from_pretrained(emb_weights,padding_idx = self.word2vec.key_to_index["<PAD>"])  
        emb_dim = emb_weights.shape[1]

        # LSTM layer(s)
        self.lstm = nn.LSTM(emb_dim, self.lstm_hid_dim, self.lstm_layers, dropout = self.lstm_dropout, bidirectional=self.bidirectional, batch_first=True)

        # Linear layer
        dirs = 2 if self.bidirectional else 1
        self.lin1 = nn.Linear(self.sentence_len*dirs*self.lstm_hid_dim, self.lin_hid_dim)
        
        self.dropout = nn.Dropout(p=self.lin_dropout)

        self.lin2 = nn.Linear(self.lin_hid_dim,1)
        
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()
        
        self.logger.info(f"Model is using '{self.device}' for training")
        self.to(self.device)

    def forward(self, X):
        self.lstm.flatten_parameters()

        N_samples = X.shape[0]
        
        X = self.embedding(X) 
        X, _ = self.lstm(X)
        
        # Makes all of features for a sample into 1 vector
        X = X.reshape(N_samples,-1)
        X = self.lin1(X)

        X = self.dropout(X)
        output = self.lin2(X)
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
            pred_batch_size = 2**12
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
        else:
            torch.save(self.best_state_dict, path)

    def save_training_losses(self,path):
        assert self.train_loss
        with open(path,"w") as of:
            of.write(','.join(map(str,self.train_loss)))
            if self.dev_loss:
                of.write('\n')
                of.write(','.join(map(str,self.dev_loss)))

        
      
def main(): 
    D = torch.tensor(np.load('data/small/Automotive.npy'))
    x, y = D[:,:128] , D[:,-1].float()
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    model = RNN()
    model.fit(x_train, y_train, E=10)
    #model.save_training_losses('data/train_dev_loss.csv')
    #model.save('data/models/RNN025.pt')
    P = model.predict(x_test)
    print("\n",classification_report(y_test, P),"\n")
    wrong = P != y_test
    for idx,review in enumerate(x_test[wrong][:10]):
        print([model.word2vec.index_to_key[word] for word in review])
        print(y_test[wrong][idx])  

if __name__ == '__main__':
    main()
