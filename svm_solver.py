import sys
import cvxopt
import numpy as np
import breast_cancer_svm
import iris_svm
import diabetes_svm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def svm_dual(XiXj, y):
	P = y[:, None]*y
	P = cvxopt.matrix(P*XiXj)
	print 'Rank', np.linalg.matrix_rank(P)
	# print 'Eigenvalues', np.linalg.eig(P)
	q = cvxopt.matrix(-np.ones((y.shape[0], 1)))
	G = cvxopt.matrix(-np.eye(y.shape[0]))
	h = cvxopt.matrix(np.zeros(y.shape[0]))
	A = cvxopt.matrix(y.reshape(1, -1))
	b = cvxopt.matrix(np.zeros(1))
	solver = cvxopt.solvers.qp(P, q, G, h, A, b)
	alphas = np.array(solver['x'])
	return alphas

def rbf_kernel(x, z, sigma=10):
	return np.exp(-np.sum(x-z)**2)/(sigma**2)

def quadratic_kernel(x, z):
	return (x.T.dot(z)+1)**2

def sigmoid_kernel(x, z, a=0.01, r=-0.9):
	return np.tanh(a*x.dot(z.T)+r)

def svm_softmargin_dual(X, y):
	print 'Calculating Soft Margin'
	P_sqrt = y[:, None]*X
	P = cvxopt.matrix(P_sqrt.dot(P_sqrt.T))
	print 'Rank', np.linalg.matrix_rank(P)
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
	print 'Loss', loss
	return alphas

def calc_bias(alphas, y, XiXj):
	non_zero_indices = np.where(alphas>1e-7)[0]
	print 'Number of support vectors', len(non_zero_indices)
	y = y.reshape(-1, 1)
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,:]), axis=0)
	bias = y[non_zero_indices, 0] - WXi[non_zero_indices]
	bias = np.mean(bias)
	return bias, WXi

def calc_bias_softmargin(alphas, y, XiXj):
	non_zero_indices = np.where((alphas>1e-7) & (alphas<C))[0]
	psi_indices = np.where(abs(alphas-C)<=1e-7)[0]
	print 'Number of psis via cmplementary slackness', len(psi_indices)
	print 'Number of support vectors', len(non_zero_indices)
	y = y.reshape(-1, 1)
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,:]), axis=0)
	bias = y[non_zero_indices, 0] - WXi[non_zero_indices]
	bias = np.mean(bias)
	psis = 1-y[psi_indices, 0]*(WXi[psi_indices]+bias)
	return bias, WXi, psis	

def predict_kernel(alphas, bias, X, sample):
	non_zero_indices = np.where(alphas>1e-7)[0]
	pred = bias
	for idx in non_zero_indices:
		if KERNEL=='rbf':
			pred += alphas[idx]*y[idx]*rbf_kernel(X[idx], sample)
		elif KERNEL=='quadratic':
			pred += alphas[idx]*y[idx]*quadratic_kernel(X[idx], sample)
		elif KERNEL=='sigmoid':
			pred += alphas[idx]*y[idx]*sigmoid_kernel(X[idx], sample)
		else:
			pred += alphas[idx]*y[idx]*X[idx].dot(sample)

	return pred

if sys.argv[1]=='iris':
	X, y = iris_svm.load_data_binary()
elif sys.argv[1]=='breast_cancer':
	X, y = breast_cancer_svm.load_data()
elif sys.argv[1]=='diabetes':
	X,y = diabetes_svm.load_data()

print X.shape, y.shape
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
print 'After PCA', X.shape
print 'Visualizing now...'
data_pts = [[] for _ in range(2)]
for i in range(X.shape[0]):
	data_pts[min(int(y[i]), 0)].append(X[i,:])

markers = ['^', 'o']
labels = ['negative', 'positive']
for c in range(2):
	this_class = np.array(data_pts[c])
	plt.scatter(this_class[:,0], this_class[:,1], marker=markers[c], label=labels[c])

plt.legend()
XiXj = np.zeros((X.shape[0], X.shape[0]))
KERNEL = 'linear'
IS_SOFT = False
if len(sys.argv) > 2:
	KERNEL = sys.argv[2]
	if sys.argv[2]=='rbf':
		for i in range(X.shape[0]):
			for j in range(i, X.shape[0]):
				XiXj[i, j] = rbf_kernel(X[i], X[j])
				XiXj[j, i] = XiXj[i, j]
	elif sys.argv[2]=='quadratic':
		for i in range(X.shape[0]):
			for j in range(i, X.shape[0]):
				XiXj[i, j] = quadratic_kernel(X[i], X[j])
				XiXj[j, i] = XiXj[i, j]
	elif sys.argv[2]=='sigmoid':
		for i in range(X.shape[0]):
			for j in range(i, X.shape[0]):
				XiXj[i, j] = sigmoid_kernel(X[i], X[j])
				XiXj[j, i] = XiXj[i, j]
	else:
		XiXj = X.dot(X.T)

if len(sys.argv) > 3:
	IS_SOFT = sys.argv[3]=='soft'

print 'Prepared XiXj matrix'
if IS_SOFT:
	C=0.1
	alphas = svm_softmargin_dual(XiXj, y)
	bias, WXi, psis = calc_bias_softmargin(alphas, y, XiXj)
else:	
	alphas = svm_dual(XiXj, y) #if not IS_SOFT else svm_softmargin_dual(XiXj, y)
	bias, WXi = calc_bias(alphas, y, XiXj) #if not IS_SOFT else calc_bias_softmargin(alphas, y, XiXj)

print "Bias", bias
# print("Weights:,", WXi)

predicted = np.sign(WXi+bias)
print("Accuracy:", np.sum(y == predicted)*1.0/len(predicted))

my_points = np.mgrid[-2:5:0.05, -2:2:0.05].reshape(2, -1).T
if IS_SOFT:
	colors = ['green', 'red', 'blue']
	region_colors = []
	for i in range(len(my_points)):
		prediction = predict_kernel(alphas, bias, X, my_points[i])
		if prediction>=0 and prediction:
			region_colors.append(colors[2])
		else:
			region_colors.append(colors[max(0, np.sign(prediction))])
else:
	colors = ['green', 'red']
	region_colors = []
	for i in range(len(my_points)):
		region_colors.append(colors[max(0, np.sign(predict_kernel(alphas, bias, X, my_points[i])))])

plt.scatter(my_points[:,0], my_points[:,1], c=region_colors, alpha=0.05)
plt.savefig('breast_cancer_{}_{}'.format(KERNEL, 'soft' if IS_SOFT else 'hard'))
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