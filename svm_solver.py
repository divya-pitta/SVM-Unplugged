import sys
import cvxopt
import numpy as np
import breast_cancer_svm
import iris_svm
import diabetes_svm

def svm_dual(XiXj, y):
	P = y[:, None]*y
	P = cvxopt.matrix(P*XiXj)
	q = cvxopt.matrix(-np.ones((y.shape[0], 1)))
	G = cvxopt.matrix(-np.eye(y.shape[0]))
	h = cvxopt.matrix(np.zeros(y.shape[0]))
	A = cvxopt.matrix(y.reshape(1, -1))
	b = cvxopt.matrix(np.zeros(1))
	solver = cvxopt.solvers.qp(P, q, G, h, A, b)
	alphas = np.array(solver['x'])
	return alphas

def rbf_kernel(X, sigma=10):
	XiXj = np.zeros((X.shape[0], X.shape[0]))
	for i in range(X.shape[0]):
		for j in range(i, X.shape[0]):
			XiXj[i, j] = np.exp(-np.sum((X[i]-X[j])**2)/(sigma**2))
			XiXj[j, i] = XiXj[i, j]
			# print 'olo', XiXj[i, j]

	return XiXj

def quadratic_kernel(X, power=2):
	XiXj = np.zeros((X.shape[0], X.shape[0]))
	for i in range(X.shape[0]):
		for j in range(i, X.shape[0]):
			XiXj[i, j] = (X[i].dot(X[j].T) + 1)**power
			XiXj[j, i] = XiXj[i, j]

	return XiXj

def cubic_kernel(X):
	return quadratic_kernel(X, power=3)	

def svm_softmargin_dual(X, y, C):
	P_sqrt = y[:, None]*X
	P = cvxopt.matrix(P_sqrt.dot(P_sqrt.T))
	q = cvxopt.matrix(-np.ones((y.shape[0], 1)))
	G1 = -np.eye(y.shape[0])
	G2 = np.eye(y.shape[0])
	G = cvxopt.matrix(np.vstack((G1, G2)))
	h1 = np.zeros(y.shape[0])
	h2 = C*np.ones(y.shape[0])
	h = cvxopt.matrix(np.append(h1, h2))
	A = cvxopt.matrix(y.reshape(1, -1))
	b = cvxopt.matrix(np.zeros(1))
	solver = cvxopt.solvers.qp(P, q, G, h, A, b)
	alphas = np.array(solver['x'])
	loss = 0.5*np.array(np.dot(np.dot(alphas.T, P), alphas)) + np.dot(q.T, alphas)
	return alphas, loss

def calc_bias(alphas, y, XiXj):
	non_zero_indices = np.where(alphas>1e-4)[0]
	print(alphas[non_zero_indices].shape, y[non_zero_indices].shape)
	# exit()
	y = y.reshape(-1, 1)
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,:]),axis=0)
	bias = 1.0/np.array(y) - WXi
	bias = np.mean(bias)
	return bias, WXi

if sys.argv[1]=='iris':
	X, y = iris_svm.load_data_binary()
elif sys.argv[1]=='breast_cancer':
	X, y = breast_cancer_svm.load_data()
elif sys.argv[1]=='diabetes':
	X,y = diabetes_svm.load_data()

print X.shape, y.shape
if len(sys.argv) > 2:
	if sys.argv[2]=='rbf':
		XiXj = rbf_kernel(X)
	elif sys.argv[2]=='quadratic':
		XiXj = quadratic_kernel(X)
	elif sys.argv[2]=='cubic':
		XiXj = cubic_kernel(X)
	else:
		print 'No such kernel available'
		exit()
else:
	print 'solving linear now...'
	XiXj = X.dot(X.T)

alphas = svm_dual(XiXj, y)
bias, WXi = calc_bias(alphas, y, XiXj)
print("Bias", bias)
print("Weights:,", WXi)

predicted = []
for xi, yi in zip(range(len(X)), y):
	if(WXi[xi]+bias>=0):
		predicted.append(1)
	else:
		predicted.append(-1)
predicted = np.array(predicted)
print("Accuracy:",np.sum(y == predicted)*1.0/len(predicted))

# print np.where(alphas>1e-4)[0].shape
# print alphas[np.where(alphas>1e-4)[0]]

# print("------SVM-Soft Margin-------")
# c_arr = [0.001, 0.01, 1, 10, 100, 1000, 10000, 100000, 1000000]
# min_loss = float("inf")
# c_opt = 0
# for c in c_arr:
# 	alphas, loss = svm_softmargin_dual(X, y, c)
# 	if(loss<min_loss):
# 		min_loss = loss
# 		c_opt = c

# print("Optimal c: ", c_opt)

# alphas, loss = svm_softmargin_dual(X, y, c_opt)
# print np.where(alphas>1e-4)[0].shape
# print alphas[np.where(alphas>1e-4)[0]]