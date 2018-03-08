from abc import ABC, abstractmethod

class Classifier(ABC):
    #fits the model using inputs X and targets y, with train paramters stored in **kwargs
    @abstractmethod
    def fit(X, y, **kwargs):
        pass

    #predicts the target value (int) of either x, a single input, or x, a matrix of inputs
    @abstractmethod
    def predict(x):
        pass
