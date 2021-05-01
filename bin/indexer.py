# indexer.py
#   binary sentiment classification
# by: Group B

# imports
import os
import torch
from torch import nn
from tqdm import tqdm
import json 
import numpy as np
import nltk; # nltk.download('punkt')
from nltk.tokenize import word_tokenize
import gensim
from reader import parse

# global varaibles 
sequence_length = 128
data_path = "/home/common/datasets/amazon_review_data_2018/reviews"
model = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)

# parsing
def parser(file_name):
	for review in parse(f"{data_path}/{file_name}"): 
		rating = review['overall'] >= 4
		text = word_tokenize(review.get('reviewText', ""))
		text = truncating(text)
		title = word_tokenize(review.get('summary', ""))
		title = truncating(title)
		D = torch.cat((text, title, [rating]), dim=1)
	with open(f"data/npys/{k}."
	return domains


# embedding this baby
def truncating(sample): 
    for idx, sample in enumerate(domain):
        sample_index = []
        for word in sample['text'][-min(sequence_length, len(sample['text'])):]:
            word_index = model.key_to_index.get(word, 1)
            sample_index.append(word_index)  
        if len(sample_index) < sequence_length: 
            sample_index = [0 for _ in range(sequence_length - len(sample_index))] + sample_index
        domain[idx]['text'] = sample_index
        title_index = []
        for word in sample['title'][-min(sequence_length, len(sample['title'])):]:
            word_index = model.key_to_index.get(word, 1)
            title_index.append(word_index) 
        if len(title_index) < sequence_length:
            title_index = [0 for _ in range(sequence_length - len(title_index))] + title_index
        domain[idx]['title'] = title_index
    return domain

# call stack
def main():
    domains = parser()
    for k, v in domains.items():
        domains[k] = truncating(v)
    for k, v in domains.items():
        X, y = [sample['text'] for sample in v], [float(sample['sentiment']) for sample in v]
        X, y = map(torch.tensor, [X, y]) 
        D = torch.cat((X, y.reshape(len(v), 1)), dim=1)
        with open(f'data/npys/{k}.npy', 'wb') as f:
            np.save(f, D)  

if __name__ == "__main__":
    main()
