from mnist import MNIST
import numpy as np
import cvxopt

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

def rbf_kernel(X1, X2, sigma=10):
	XiXj = np.zeros((X1.shape[0], X2.shape[0]))
	for i in range(X1.shape[0]):
		for j in range(i, X2.shape[0]):
			XiXj[i, j] = np.exp(-np.sum((X1[i]-X2[j])**2)/(sigma**2))
			XiXj[j, i] = np.exp(-np.sum((X2[i]-X1[j])**2)/(sigma**2))

	return XiXj

def calc_wxi(alphas, XiXj, j):
	non_zero_indices = np.where(alphas>1e-4)[0]
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,j]),axis=0)
	return WXi

def calc_bias(alphas, y, XiXj):
	non_zero_indices = np.where(alphas>1e-4)[0]
	# print(alphas[non_zero_indices].shape, y[non_zero_indices].shape)
	# exit()
	y = y.reshape(-1, 1)
	WXi = np.sum((alphas[non_zero_indices]*y[non_zero_indices]*XiXj[non_zero_indices,:]),axis=0)
	bias = 1.0/np.array(y) - WXi
	bias = np.mean(bias)
	return bias

# ============ Data Loader ===========================
mndata = MNIST('/home/tejash/datasets/MNIST/')
X_train, y_train = mndata.load_training()
X_valid, y_valid = X_train[-10000:], y_train[-10000:]
X_train, y_train = X_train[:-10000], y_train[:-10000]
X_test, y_test = mndata.load_testing()
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)
X_valid = np.array(X_valid, dtype=np.float32)
y_valid = np.array(y_valid)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)
print X_train.shape, y_train.shape
NUM_CLASSES = 10

# ============== Preprocessing ========================
MEANS = [np.mean(X_train[:, dim]) for dim in range(NUM_CLASSES)]
STDS = [np.std(X_train[:, dim]) for dim in range(NUM_CLASSES)]
for dim in range(NUM_CLASSES):
	X_train[:, dim] -= MEANS[dim]
	X_train[:, dim] /= STDS[dim]
	
	X_valid[:, dim] -= MEANS[dim]
	X_valid[:, dim] /= STDS[dim]

	X_test[:, dim] -= MEANS[dim]
	X_test[:, dim] /= STDS[dim]

print 'Done preprocessing'

# ============== Train 10 SVMs ===================
XiXj = rbf_kernel(X_valid, X_valid)
SVM_ALPHAS = []
for n in range(NUM_CLASSES):
	this_y_train = y_valid.copy()
	this_y_train[np.where(this_y_train==n)[0]] = 1.0
	this_y_train[np.where(this_y_train!=n)[0]] = -1.0
	this_alphas = svm_dual(XiXj, this_y_train)
	SVM_ALPHAS.append(this_alphas)

# ============== Test the SVMs ====================
print 'testing now...'
XiXj = rbf_kernel(X_train, X_test)
BIASES = [calc_bias(alphas, y_test, XiXj) for alphas in SVM_ALPHAS]

predicted = []
for j in range(len(X)):
	predicted.append(np.argmax([calc_wxi(SVM_ALPHAS[i], XiXj, j)+BIASES[i] for i in range(NUM_CLASSES)]))

predicted = np.array(predicted)
print "Accuracy:", np.sum(y == predicted)*1.0/len(predicted)
