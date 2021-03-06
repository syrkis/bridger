# imports
import os
import json
import gzip
import random


# next line gives file not found when run in singularity
data_root = '/home/common/datasets/amazon_review_data_2018'

core_path = os.path.join(data_root, '5_core')
reviews_path = os.path.join(data_root, 'reviews')
meta_data_path = os.path.join(data_root, 'metadata')

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def sample_data(number_of_files, size_of_files, seed = None):
	"""
	Input:

		number_of_files: int
			- Number of files to sample from. Will be randomly selected	

		size_of_files int
			- Number of reviews to sample from each data file. Will not be chosen randomly, is picked as the first 'size_of_files' reviews

		seed: int
			- Seed to set for random collection
	Returns:
	
		data: list[list[dict]]
			- Each entry in first list is a data file, each entry in inner list is a list of reviews, each dictionary is a review

		names: list[string]
			- List of names for data files, corresponding to same index in data
	"""
	random.seed(seed)
	
	core_data_files = os.listdir(core_path)
	if number_of_files > len(core_data_files) or number_of_files == -1:
		number_of_files = len(core_data_files)

	if size_of_files == -1:
		size_of_files = float('inf') 

	files_to_use = random.sample(core_data_files, number_of_files)
	data, names = [], []
	for core_file in files_to_use: 
		base_file_name = core_file[:-10]
		full_path = os.path.join(core_path, core_file)
		print('READING %s' % base_file_name)
		
		counter = 0
		data_sample = []
		for line in parse(full_path):
			data_sample.append(line)
			counter += 1
			if counter >= size_of_files: break

		data.append(data_sample)
		names.append(base_file_name)

	return data, names

def sample_domain(domain):
	return parse(f"{data_root}/reviews/{domain}")


def read_all_files():
	return sample_data(-1,-1)

def main():
	data,names = sample_data(3, 10)
	open('data/data_test.txt', 'w').write(json.dumps(data))

if __name__ == '__main__':
	main()
