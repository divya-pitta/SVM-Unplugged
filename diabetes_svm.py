import numpy as np

"""
Returns filled matrix X and vector y
Note that y=-1 corresponds to true label 2 and y=1 corresponds to true label 4
"""
def load_data(file='diabetes_scale.txt'):
	X = []
	y = []
	with open(file) as f:
		for line in f:
			vals = line.strip().split()
			val_arr = [float(feature_val.split(':')[-1]) for feature_val in vals[1:]]
			if(len(val_arr)==8):
				X.append(val_arr)
				if vals[0] == '-1':
					y.append(-1.0)
				else:
					y.append(1.0)

	return np.array(X, dtype=np.float32), np.array(y)