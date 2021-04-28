# comparer.py
#	compares two domains
# by: Group 2

# imports
import os
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from scipy.stats import entropy

# Global vars
count_path = '../data/counts/'
targets = os.listdir(count_path)
lap_smooth = 0.1

def KLD(vcP,vcQ):
	"""
	Inputs:
		vcP, vcQ: dataframes
			- pandas dataframes with index being words and column being count for word  
	"""

	if vcP.columns == vcQ.columns:
		return 0
	# Joins data and add smoothing
	full = vcP.join(vcQ,how='outer')
	lackP = full.iloc[:,0].isna().sum()
	lackQ = full.iloc[:,1].isna().sum()	
	full += lap_smooth
	full.fillna(lap_smooth,inplace=True)
	
	return entropy(full.iloc[:,0],full.iloc[:,1])
	
# vocab
def vocab_count(D,num):
	"""
	Inputs:
		D: string
			- Path to count file for the dataset

		num: int
			- Number to give to column name

	Returns:
		vocab_df: dataframe
			- Pandas dataframe with words as index and counts in a column
	"""
	pattern = re.compile(r"(\d+)\s(.*)")
	vocabP = defaultdict()
	with open(count_path + '/' + D,'r') as infileP:
		all_lines = infileP.readlines()
		for i in range(1,len(all_lines)-1):
			line = all_lines[i].strip()
			m = pattern.match(line)
			vocabP[m.group(2)] = int(m.group(1))

	vocab_df = pd.DataFrame.from_dict(vocabP,orient='index',columns=[f'count{num}']) 
	return vocab_df

def compare_all():
	loaded = [] 
	for i,f in enumerate(targets):
		loaded.append(vocab_count(f,i))
	values = []
	for df1 in loaded:
		print("new")
		temp_values  = []
		for df2 in loaded:
			temp_values.append(KLD(df1,df2))
		values.append(temp_values)
	arr = np.array(values)
	print(arr)
	np.savetxt('../data/all_pair_KLD.csv',arr,delimiter=',',header=','.join(targets))

# call stack
def main():
	#P, Q = targets[1], targets[3]
	#print(P,"\t",Q)	
	#print(KLD(P,Q))
	compare_all()


if __name__ == '__main__':
	main()
