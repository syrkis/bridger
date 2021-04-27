# comparer.py
#	compares two domains
# by: Group 2

# imports
import os


# reading
count_path = '../data/counts/'
targets = os.listdir(count_path)

# evaluator
def compare(P, Q):
	# temporarily ignores words not in both domains
	V = vocab(P).intersection(vocab(Q))
	return V
	
	

# vocab
def vocab(D):
	with open(f"{count_path}/{D}", 'r') as f:
		data = f.read()
	words = data.strip().split('\n')
	vocab = [word.split()[1] for word in words[1:]]	
	return vocab 

# call stack
def main():
	P, Q = targets[-4], targets[-3]
	print(compare(P, Q))

if __name__ == '__main__':
	main()
