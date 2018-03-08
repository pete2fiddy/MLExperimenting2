from regression.regressor import Regressor
import linalg.matrix.matmath as matmath
import numpy as np



class LinearRegression(Regressor):

    def fit(self, X, y, **kwargs):
        #TODO: add boolean to **kwargs to use gradient descent or not
        weight_mat = np.identity(X.shape[0], dtype = np.float64) if not "weights" in kwargs else np.diag(kwargs["weights"])
        if len(y.shape) == 1:
            y = np.array([y]).T
        self.__plane_vecs = np.zeros((y.shape[1], X.shape[1]))
        pseudo_inv = np.dot(np.linalg.inv(X.T.dot(weight_mat).dot(X)), X.T)
        for i in range(0, self.__plane_vecs.shape[0]):
            self.__plane_vecs[i] = np.dot(pseudo_inv, y[:,i])

    def predict(self, X):
        return np.dot(X, self.__plane_vecs.T)
