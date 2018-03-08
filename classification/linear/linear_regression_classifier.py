import numpy as np
from classification.classifier import Classifier
import classification.classify_datahelper as classify_datahelper
from regression.linear_regression import LinearRegression

class LinearRegressionClassifier(Classifier):

    def fit(self, X, y, **kwargs):
        linear_model = kwargs["linear_model"]
        assert isinstance(linear_model, LinearRegression)#ensures that the linear_model is some kind of linear regression (whether
        #lasso, shrinkage, etc will be coded in extensions to LinearRegression or accomplish in just the LinearRegression class
        #is not yet known)
        X_affine = self.__affine_X(X)
        self.__linear_model = kwargs["linear_model"]
        y_one_of_K, self.__uniques = classify_datahelper.one_of_K(y)
        self.__linear_model.fit(X_affine, y_one_of_K)

    def __affine_X(self, X):
        X_affine = np.ones((X.shape[0], X.shape[1]+1), dtype = X.dtype)
        X_affine[:, :X.shape[1]] = X
        return X_affine

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.array([X])
        X_affine = self.__affine_X(X)
        predict_one_of_K = self.__linear_model.predict(X_affine)
        print("predict one of K: ", predict_one_of_K)
        argmaxes = np.argmax(predict_one_of_K, axis = 1)
        return self.__uniques[argmaxes]
