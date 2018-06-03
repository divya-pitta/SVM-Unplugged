import numpy as np 

"""
Loads the Iris dataset, only the first two classes
y=-1 means Iris-setosa
y=1 means Iris-versicolor
"""
def load_data_binary(file='iris.txt'):
	X = []
	y = []
	with open(file) as f:
		for line in f:
			vals = line.strip().split(',')
			feats = map(float, vals[:4])
			if vals[4]=='Iris-virginica':
				X.append(feats)
				y.append(-1.)
			elif vals[4]=='Iris-versicolor':
				X.append(feats)
				y.append(1.)

	return np.array(X), np.array(y)

def load_data_multiclass():
	pass