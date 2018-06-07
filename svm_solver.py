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

def quadratic_kernel(x, z, power=2):
	return (x.dot(z.T) + 1)**power

def sigmoid_kernel(x, z, a=10.0, r=-1e50):
	return np.tanh(a*x.dot(z.T)+r)

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
	# print(alphas[non_zero_indices].shape, y[non_zero_indices].shape)
	# exit()
	y = y.reshape(-1, 1)
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,:]),axis=0)
	bias = 1.0/np.array(y) - WXi
	bias = np.mean(bias)
	return bias, WXi

def predict_rbf(alphas, bias, X, sample):
	non_zero_indices = np.where(alphas>1e-4)[0]
	pred = bias
	for idx in non_zero_indices:
		pred += alphas[idx]*y[idx]*rbf_kernel(X[idx], sample)

	return pred

def predict_quadratic(alphas, bias, X, sample):
	non_zero_indices = np.where(alphas>1e-4)[0]
	pred = bias
	for idx in non_zero_indices:
		pred += alphas[idx]*y[idx]*quadratic_kernel(X[idx], sample)

	return pred

def predict_sigmoid(alphas, bias, X, sample):
	non_zero_indices = np.where(alphas>1e-4)[0]
	pred = bias
	for idx in non_zero_indices:
		pred += alphas[idx]*y[idx]*sigmoid_kernel(X[idx], sample)

	return pred

def predict_linear(alphas, bias, X, sample):
	non_zero_indices = np.where(alphas>1e-4)[0]
	pred = bias
	for idx in non_zero_indices:
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
if len(sys.argv) > 2:
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

	elif sys.argv[2]=='cubic':
		for i in range(X.shape[0]):
			for j in range(i, X.shape[0]):
				XiXj[i, j] = cubic_kernel(X[i], X[j])
				XiXj[j, i] = XiXj[i, j]

	elif sys.argv[2]=='sigmoid':
		for i in range(X.shape[0]):
			for j in range(i, X.shape[0]):
				XiXj[i, j] = sigmoid_kernel(X[i], X[j])
				XiXj[j, i] = XiXj[i, j]
	else:
		print 'No such kernel available'
		exit()
else:
	XiXj = X.dot(X.T)

print 'Prepared XiXj matrix'
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


my_points = np.mgrid[-2:5:0.05, -2:2:0.05].reshape(2, -1).T
colors = ['green', 'red']
region_colors = []
for i in range(len(my_points)):
	region_colors.append(min(0, np.sign(predict_sigmoid(alphas, bias, X, my_points[i]))))

plt.scatter(my_points[:,0], my_points[:,1], c=region_colors, alpha=0.05)
plt.savefig('breast_cancer_sigmoid_svm.png')
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