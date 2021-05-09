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
import json
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger('model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# globals 
model = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)

# declare nn
class RNN(nn.Module):
        
    batch_size = 32

    lstm_layers = 3
    lstm_dropout = 0.25
    bidirectional = True
    lstm_hid_dim = 32

    lin_hid_dim = 250
    lin_dropout = 0.25

    def __init__(self, sentence_len):
        super(RNN, self).__init__()
        self.sentence_len = sentence_len

        # Embedding
        emb_weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrained(emb_weights)  
        emb_dim = emb_weights.shape[1]

        # LSTM layer(s)
        self.lstm = nn.LSTM(emb_dim, RNN.lstm_hid_dim, RNN.lstm_layers, dropout = RNN.lstm_dropout, bidirectional=RNN.bidirectional, batch_first=True)

        # Linear layer
        dirs = 2 if RNN.bidirectional else 1
        self.lin1 = nn.Linear(self.sentence_len*dirs*RNN.lstm_hid_dim, RNN.lin_hid_dim)
        
        self.dropout = nn.Dropout(p=RNN.lin_dropout)

        self.lin2 = nn.Linear(RNN.lin_hid_dim,1)
        
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        N_samples = inp.shape[0]
        X = self.embedding(inp) 
        X, _ = self.lstm(X)

        # Makes all of features for a sample into 1 vector
        X = X.reshape(N_samples,-1)
        X = self.lin1(X)

        X = self.dropout(X)
        output = self.lin2(X)
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
        if x_dev is not None:
            return L, L_dev
        return L

    def predict(self,X):
        return self.predict_proba(X) > 0.5
        

    def predict_proba(self,X):
        self.eval() # sets model in evaluation mode, to not use dropout
        with torch.no_grad():
            if device == 'cuda':
                res = torch.cuda.memory_reserved(0)
                alo = torch.cuda.memory_allocated(0)
                free_mem = res - alo
                logger.info(f"free memory {free_mem}")
                row_mem = X.element_size() * X.shape[1]
                pred_batch_size = (free_mem*0.9) //row_mem
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
      
def main(): 
    logger.info(f'is using {device}')
    logger.info(f"dropout is set to {RNN.lstm_dropout}")
    D = torch.tensor(np.load('data/npys/Books.npy'))
    x, y = D[:,:128] , D[:,-1].float()
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=10**5)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train,y_train, test_size=10**5)
    model = RNN(128).to(device) 
    L, L_dev = model.fit(x_train, y_train, x_dev, y_dev, E=30)
    try:
        logger.info(f"L is: {str(L)}")
        logger.info(f"L_dev is: {str(L_dev)}")
        with open('/home/timp/repositories/bringo/data/train_losses_025_drop.csv',"w") as of:
            of.write(",".join(map(str,L))) 
            of.write("\n")
            of.write(",".join(map(str,L_dev))) 
    except:
        pass
    P = model.predict(x_test)
    print("\n",classification_report(y_test, P),"\n")

if __name__ == '__main__':
    main()
