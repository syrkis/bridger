# sentiment.py
#   rnn for sentiment classification
# by: Group 2

# imports
import torch
from torch import nn
import gensim


# globals 
model = gensim.models.KeyedVectors.load_word2vec_format('../data/twitter.bin', binary=True)



# declare nn
class RNN(nn.Module):
        
    def __init__(self, sample_size, hidden_size, layers):
        super(RNN, self).__init__()
        self.sample_size = sample_size
        emb_weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrained(emb_weights)  
        self.emb_dim = emb_weights.shape[1]

        # layer 1
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, layers, bidirectional=False, batch_first=True)

    def forward(self, samples):
        X = self.embedding(samples) 
        X = self.lstm(X)
        print(X[0].shape)

         
rnn = RNN(128, 20, 1)

samples = [[1 for i in range(128)], [0 for i in range(128)]]
samples = torch.tensor(samples)
rnn.forward(samples)

