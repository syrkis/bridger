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
import logging
FORMAT = '%(name)s: %(asctime)s %(message)s'
timeformat = '%m-%d %H:%M:%S'
logging.basicConfig(format=FORMAT, datefmt=timeformat, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('model')
logger.info("STARTING LOADING")


# globals 
model = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)



# declare nn
class RNN(nn.Module):
        
    lstm_layers = 2
    bidirectional = True
    batch_size = 32
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

    def fit(self, X, y, E = 4):
        L = []
        assert X.shape[0] == y.shape[0]
        optimizer = Adam(self.parameters())
        loss = nn.BCELoss()
        logger.info(X.shape)
        num_batches = X.shape[0] // RNN.batch_size
        X = X[:RNN.batch_size * num_batches].reshape(num_batches, RNN.batch_size, 128)
        y = y[:RNN.batch_size * num_batches].reshape(num_batches, RNN.batch_size)
        for e in range(E):
            losses = []
            for i in tqdm(range(num_batches)):
                optimizer.zero_grad()
                pred = self.forward(X[i])
                cross_entropy_loss = loss(pred, y[i])
                cross_entropy_loss.backward()    
                optimizer.step() 
                losses.append(cross_entropy_loss.item())
            L.append(np.mean(losses))
        return L
      
def main(): 
  
    data_dir = '../data/npys'
    samples = os.listdir(data_dir)
    with open(f"{data_dir}/{samples[0]}", 'rb') as f:
        D = np.load(f)
        X, y = D[: , :-1], D[: , -1]
    rnn = RNN(128)

    rnn.fit(X, y)

if __name__ == '__main__':
    main()
