# imports
import os
import json
import gzip

# reading
data_root = '/home/common/datasets/amazon_review_data_2018'
data_dirs = os.listdir(data_root)
_5_core = os.listdir(f"{data_root}/{data_dirs[1]}")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def main():
	for line in parse(f"{data_root}/{data_dirs[1]}/{_5_core[0]}"):
		print(line.keys())

if __name__ == '__main__':
	main()
