import sys
import cvxopt
import numpy as np
import breast_cancer_svm
import iris_svm

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

if sys.argv[1]=='iris':
	X, y = iris_svm.load_data_binary()
else:
	X, y = breast_cancer_svm.load_data()

print X.shape, y.shape
if len(sys.argv) > 2:
	if sys.argv[2]=='rbf':
		XiXj = rbf_kernel(X)
	elif sys.argv[2]=='quadratic':
		XiXj = quadratic_kernel(X)
	elif sys.argv[3]=='cubic':
		XiXj = cubic_kernel(X)
	else:
		print 'No such kernel available'
		exit()
else:
	XiXj = X.dot(X.T)

alphas = svm_dual(XiXj, y)
print np.where(alphas>1e-4)[0].shape
print alphas