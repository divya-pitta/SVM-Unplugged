import sys
import cvxopt
import numpy as np
import breast_cancer_svm
import iris_svm

def svm_dual(X, y):
	P_sqrt = y[:, None]*X
	P = cvxopt.matrix(P_sqrt.dot(P_sqrt.T))
	q = cvxopt.matrix(-np.ones((y.shape[0], 1)))
	G = cvxopt.matrix(-np.eye(y.shape[0]))
	h = cvxopt.matrix(np.zeros(y.shape[0]))
	A = cvxopt.matrix(y.reshape(1, -1))
	b = cvxopt.matrix(np.zeros(1))
	solver = cvxopt.solvers.qp(P, q, G, h, A, b)
	alphas = np.array(solver['x'])
	return alphas

if sys.argv[1]=='iris':
	X, y = iris_svm.load_data_binary()

print X.shape, y.shape
alphas = svm_dual(X, y)
print np.where(alphas>1e-4)[0].shape
print alphas