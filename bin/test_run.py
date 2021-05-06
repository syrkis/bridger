# imports
from sentiment import RNN
import numpy as np
from sklearn.metrics import classification_report
import torch

# reading
def test_run():
	D = torch.tensor(np.load('data/npys/Books.npy'))
	train, test = D[:9 * 10 ** 5], D[9 * 10 ** 5:]
	X_train, y_train = train[:, :128], train[:, -1].float()
	X_test, y_test = test[:, :128], test[:, -1].float()
	model = RNN(128) 
	L = model.fit(X_train, y_train)
	P = model.forward(X_test) > 0.5
	print(classification_report(y_test, P))

