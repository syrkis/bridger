# sentiment.py
#   rnn for sentiment classification
# by: Group 2

# imports
import torch
from torch import nn
import gensim

import time

# globals 
model = gensim.models.KeyedVectors.load_word2vec_format('../data/twitter.bin', binary=True)



# declare nn
class RNN(nn.Module):
        
    def __init__(self, sentence_len, hidden_dim, lstm_layers):
        super(RNN, self).__init__()
        self.sentence_len = sentence_len
        self.bidirectional = True

        # Embedding
        emb_weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrained(emb_weights)  
        emb_dim = emb_weights.shape[1]

        # LSTM layer(s)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, lstm_layers, bidirectional=self.bidirectional, batch_first=True)

        # Linear layer
        dirs = 2 if self.bidirectional else 1
        self.linear = nn.Linear(self.sentence_len*dirs*hidden_dim,1)

        # Sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        batch_size, seq_len = batch.shape
        assert seq_len == self.sentence_len
        print(batch.shape)

        X = self.embedding(batch) 
        print(X.shape)

        X, _ = self.lstm(X)
        print(X.shape)

        # Makes all of features for a sample into 1 vector
        X = X.reshape(batch_size,-1)
        print(X.shape)
        output = self.linear(X)
        print(output.shape)

        output = self.sigmoid(output)
        print(output.shape)
        return output.reshape(batch_size) # Vector/Tensor of shape (batch_size)

rnn = RNN(128, 16, 2)

samples = [[j for i in range(128)] for j in range(64) ]
samples = torch.tensor(samples)
t0 = time.time()
predictions = rnn.forward(samples)
t1 = time.time()
print("Forward of batch size 64:",t1-t0)
print(predictions.shape)
print(predictions)


