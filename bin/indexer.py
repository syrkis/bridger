# indexer.py
#   binary sentiment classification
# by: Group B

# imports
import os
import sys
import logging
from multiprocessing import Pool
from multiprocessing import cpu_count
import torch
from torch import nn
from tqdm import tqdm
import json
import numpy as np
import nltk;  #nltk.download('punkt')
from nltk.tokenize import word_tokenize
import gensim
from reader import parse

# global varaibles
sequence_length = 128
model = gensim.models.KeyedVectors.load_word2vec_format('data/twitter.bin', binary=True)
MAX_REVIEWS = 1000000


# parsing
def parser(tup):
	results = []
	reviews, index = tup
	logger = logging.getLogger("INDEXER")
	logger.info(f"STARTED WITH SNIPPET {index}")
	num_reviews = len(reviews)
	for rev_i in range(num_reviews):
		if rev_i%10000==0: logger.info(f"INDEX {index} has reached {round(rev_i/num_reviews,2)*100} %")
		rating = reviews[rev_i]['overall'] >= 4
		text = word_tokenize(reviews[rev_i].get('reviewText',''))
		text = truncating(text)
		title = word_tokenize(reviews[rev_i].get('summary',''))
		title = truncating(title)
		for word in title:
			text.append(word)
		text.append(rating)
		results.append(text)
	logger.info(f"SPLIT {index} IS DONE")
	return results

# embedding this baby
def truncating(sample): 
	sample_index = []
	to_use = sample[-min(sequence_length, len(sample)):]
	sample_index =[model.get_index(word,default=1) for word in to_use]
	if len(sample_index) == sequence_length:
		return sample_index
	
	final_index = [0]*(sequence_length-len(sample_index))
	for index in sample_index:
		final_index.append(index)
	return final_index

# call stack
def main(file_path):
	name = os.path.basename(file_path)[:-8]
	logger = logging.getLogger(name)
	logger.info("STARTING LOADING")

	data = []
	for review in parse(file_path):
		data.append(review)
		if len(data) >=	MAX_REVIEWS:
			break
	logger.info("DATA IS LOADED")
	cpus = cpu_count()
	size_of_snip = len(data)//cpus 
	snippets = []
	for i in range(cpus-1):
		snippets.append(data[i*size_of_snip:(i+1)*size_of_snip])
	snippets.append(data[(cpus-1)*size_of_snip:])
	logger.info(f"DATA IS SPLITTET INTO {cpus} SECTIONS")
	with Pool(cpus) as p:
		all_results = p.map(parser, list(zip(snippets,range(cpus))))
	logger.info("ALL SNIPPETS DONE")
	final = []
	for i in range(cpus):
		for review in all_results[i]:
			final.append(review)
	logger.info("ALL SNIPPETS JOINED")
	D = torch.tensor(final)
	logger.info("CREATED TENSOR OF DATA")
	name = os.path.basename(file_path)[:-8]
	with open(f'data/npys/{name}.npy','wb') as f:
		np.save(f,D) 
	logger.info(f"DATA IS SAVED AT data/npys/{name}.npy")
	del snippets

if __name__ == "__main__":
    main()
