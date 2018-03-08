from abc import ABC, abstractmethod


class Regressor(ABC):

    #Returns the regressed output of X. X can be in the following forms depending on
    #the type of model:
    #scalar or vector of scalars: if the regression model is univariate
    #vector or matrix of vectors: if the regression model is multivariate
    @abstractmethod
    def predict(self, X):
        pass

    #Regresses matrix of inputs X onto y, subject to extra arguments **kwargs
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass
