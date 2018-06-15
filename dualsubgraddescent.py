'''
This code is based on https://github.com/josiahw/SimpleSVM
In order to run this, comment/uncomment the data loader of your required dataset. 
Data loading codes are written in lines 148-151.
For example, to run on the Iris dataset, uncomment line 151 and comment line 152
The value of C can be changed in line 154
'''

import numpy
import numpy.linalg
import iris_svm
import breast_cancer_svm
import diabetes_svm

def polyKernel(a,b,pwr):
    return numpy.dot(a,b)**pwr #numpy.dot(a,a) - numpy.dot(b,b) # -1 #

def rbfKernel(a,b,gamma):
    return numpy.exp(-gamma * numpy.linalg.norm(a - b))

def linearKernel(a,b,notused):
    return numpy.dot(a,b)

class SimpleSVM:
    w = None
    a = None
    b = None
    C = None
    sv = None
    kernel = None
    kargs = ()
    tolerance = None
    verbose = True
    
    def __init__(self, 
                 C, 
                 tolerance = 0.001, 
                 kernel = numpy.dot, 
                 kargs = () 
                 ):
        """
        The parameters are: 
         - C: SVC cost
         - tolerance: gradient descent solution accuracy
         - kernel: the kernel function do use as k(a, b, *kargs)
         - kargs: extra parameters for the kernel
        """
        self.C = C
        self.kernel = kernel
        self.tolerance = tolerance
        self.kargs = kargs
        
    
    def fit(self, X, y):
        """
        Fit to data X with labels y.
        """
        
        """
        Construct the Q matrix for solving
        """       
        # ysigned = y * 2 - 1
        ysigned = y
        Q = numpy.zeros((len(data),len(data)))
        # y = y.reshape(y.shape+(1,))
        # Q = y.dot(y.T).dot(X).dot(X.T) # linear
        # Q = y.dot(y.T).dot(numpy.exp(-gamma * numpy.linalg.norm(X - ))) # rbf
        for i in xrange(len(data)):
            for j in xrange(i,len(data)):
                Qval = ysigned[i] * ysigned[j]
                Qval *= self.kernel(*(
                                (data[i,:], data[j,:])
                                + self.kargs
                                ))
                Q[i,j] = Q[j,i] = Qval
        
        print("Q calculator")
        """
        Solve for a and w simultaneously by coordinate descent.
        This means no quadratic solver is needed!
        The support vectors correspond to non-zero values in a.
        """
        self.w = numpy.zeros(X.shape[1])
        self.a = numpy.zeros(X.shape[0])
        delta = 10000000000.0
        while delta > self.tolerance:
            delta = 0.
            for i in xrange(len(data)):
                g = numpy.dot(Q[i,:], self.a) - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g/Q[i,i], 0.0), self.C) 
                self.w += adelta * X[i,:]
                delta += abs(adelta)
                self.a[i] -= adelta
            if self.verbose:
                print("Descent step magnitude:", delta)
        #print Q #self.a
        self.sv = X[self.a > 0.0, :]
        self.a = (self.a * ysigned)[self.a > 0.0]
        
        if self.verbose:
            print("Number of support vectors:", len(self.a))
        
        """
        Select support vectors and solve for b to get the final classifier
        """
        self.b = self._predict(self.sv[0,:])[0]
        if self.a[0] > 0:
            self.b *= -1
        
        
        if self.verbose:
            print("Bias value:", self.b)
    
    def _predict(self, X):
        if (len(X.shape) < 2):
            X = X.reshape((1,-1))
        clss = numpy.zeros(len(X))
        for i in xrange(len(X)):
            for j in xrange(len(self.sv)):
                clss[i] += self.a[j] * self.kernel(* ((self.sv[j,:],X[i,:]) + self.kargs))
        return clss
    
    def predict(self, X):
        """
        Predict classes for data X.
        """
        
        return self._predict(X) > self.b


if __name__ == '__main__':
    import sklearn.datasets
    from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata('MNIST original')
    # data = []
    # labels = []
    # for i in range(len(mnist.target)):
    #     if(mnist.target[i]==1):
    #         data.append(mnist.data[i])
    #         labels.append(1.0)
    #     elif mnist.target[i]==2:
    #         data.append(mnist.data[i])
    #         labels.append(-1.0)

    # data = numpy.array(data)
    # labels = numpy.array(labels)
    # print(len(data))
    # print(len(labels))
    # print(numpy.unique(labels))
    # data = sklearn.datasets.load_digits(2).data
    # labels = sklearn.datasets.load_digits(2).target
    # data, labels = iris_svm.load_data_binary()
    data, labels = breast_cancer_svm.load_data()
    C = 100.0
    clss = SimpleSVM(C,0.01,linearKernel,(0.01,))
    clss.fit(data,labels)
    print(labels)
    
    t = clss.predict(data)
    t1 = [1 if x else -1 for x in t]

    print(t1)
    print("Accuracy:",numpy.sum(t1 == labels)*1.0/len(labels))
    print("Error", numpy.sum((labels-t)**2) / float(len(data)))