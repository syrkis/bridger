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
import logging

logger = logging.getLogger('model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# globals 
model = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)

# declare nn
class RNN(nn.Module):
        
    lstm_layers = 2
    bidirectional = True
    batch_size = 128
    hidden_dim = 32

    def __init__(self, sentence_len):
        super(RNN, self).__init__()
        self.sentence_len = sentence_len

        # Embedding
        emb_weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrained(emb_weights)  
        emb_dim = emb_weights.shape[1]

        # LSTM layer(s)
        self.lstm = nn.LSTM(emb_dim, RNN.hidden_dim, RNN.lstm_layers, bidirectional=RNN.bidirectional, batch_first=True)

        # Linear layer
        dirs = 2 if RNN.bidirectional else 1
        self.linear = nn.Linear(self.sentence_len*dirs*RNN.hidden_dim,1)

        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        X = self.embedding(batch) 
        X, _ = self.lstm(X)

        # Makes all of features for a sample into 1 vector
        X = X.reshape(RNN.batch_size,-1)
        output = self.linear(X)

        output = self.sigmoid(output)
        return output.reshape(RNN.batch_size) # Vector/Tensor of shape (batch_size)

    def fit(self, X, y, E = 1):
        L = []
        assert X.shape[0] == y.shape[0]
        optimizer = Adam(self.parameters())
        loss = nn.BCELoss()
        logger.info(X.shape)
        X_batches = torch.split(X,RNN.batch_size)
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
            L.append(np.mean(losses))
        return L
      
def main(): 
    logger.info(f'is using {device}')
    D = torch.tensor(np.load('data/npys/Books.npy')).to(device)
    train, test = D[:9 * 10 ** 5], D[9 * 10 ** 5:]
    X_train, y_train = train[:, :128], train[:, -1].float()
    X_test, y_test = test[:, :128], test[:, -1].float()
    model = RNN(128).to(device) 
    L = model.fit(X_train, y_train)
    P = model.forward(X_test) > 0.5
    logger.info(classification_report(y_test, P))

if __name__ == '__main__':
    main()
