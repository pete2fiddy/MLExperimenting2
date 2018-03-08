from abc import ABC, abstractmethod
from classification.classifier import Classifier
import numpy as np



class TreeNode(Classifier, ABC):

    @abstractmethod
    def _init_node(self):
        pass

    def fit(self, X, y, **kwargs):
        impurity_func = kwargs["impurity_func"]
        depth = kwargs["depth"]
        max_split_impurity = kwargs["max_split_impurity"]
        min_points_to_split = kwargs["min_points_to_split"]
        if depth <= 0 or X.shape[0] < min_points_to_split or impurity_func([y]) > max_split_impurity:
            values, counts = np.unique(y, return_counts = True)
            self.__leaf_value = values[np.argmax(counts)]
            return None
        self.__leaf_value = None
        self._prefit(X, y)
        X_transformed = self._pretransform(X)
        best_split_impurity = None
        best_split_X_sorted = None
        best_split_y_sorted = None
        best_split_index = None
        self.__split_component = None

        for iter_index in range(0, X_transformed.shape[1]):
            X_transformed_axis_sorted, X_sorted, y_sorted = self.__sort(X_transformed[:,iter_index], X, y)
            for split_index in range(1, X_transformed.shape[0]):
                split_impurity = impurity_func((y_sorted[:split_index], y_sorted[split_index:]))
                if best_split_impurity is None or split_impurity > best_split_impurity:
                    best_split_impurity = split_impurity
                    best_split_X_sorted = X_sorted
                    best_split_y_sorted = y_sorted
                    best_split_index = split_index
                    self.__split_component = iter_index
                    self.__split_value = (X_transformed_axis_sorted[split_index-1] +\
                    X_transformed_axis_sorted[split_index])/2.0
        self.__less_child = self._init_node()
        self.__more_child = self._init_node()
        self.__less_child.fit(best_split_X_sorted[:best_split_index], best_split_y_sorted[:best_split_index], \
        impurity_func = impurity_func, depth = depth - 1, max_split_impurity = max_split_impurity, min_points_to_split = min_points_to_split)
        self.__more_child.fit(best_split_X_sorted[best_split_index:], best_split_y_sorted[best_split_index:], \
        impurity_func = impurity_func, depth = depth - 1, max_split_impurity = max_split_impurity, min_points_to_split = min_points_to_split)


    def predict(self, X):
        if self.__leaf_value is not None:
            #may crash here if it's not initialized
            if len(X.shape) == 1:
                return self.__leaf_value
            return np.full(X.shape[0], self.__leaf_value)
        if len(X.shape) == 1:
            X = np.array([X])
        X_transformed = self._pretransform(X)
        predictions = np.zeros(X.shape[0], dtype = np.int)
        less_indices = np.where(X_transformed[:,self.__split_component] < self.__split_value)
        more_indices = np.where(X_transformed[:,self.__split_component] >= self.__split_value)
        predictions[less_indices] = self.__less_child.predict(X[less_indices])
        predictions[more_indices] = self.__more_child.predict(X[more_indices])
        return predictions


    def __sort(self, X_transformed_axis, X, y):
        sorted_inds = np.asarray(sorted([(i, X_transformed_axis[i]) for i in range(len(X_transformed_axis))], key = lambda ind : ind[1])).astype(np.int)
        X_transformed_axis_sorted = X_transformed_axis[sorted_inds[:,0]]
        X_sorted = X[sorted_inds[:,0], :]
        y_sorted = y[sorted_inds[:,0]]
        return X_transformed_axis_sorted, X_sorted, y_sorted

    #"pretransforms" X, a matrix of vectors, before fitting
    #or classifying using X. If X is a matrix of vectors, each row of the output
    #will be a single pretransformed input vector
    @abstractmethod
    def _pretransform(self, X):
        pass

    #"Prefits" the node using X, if necessary (for example, to instantiate the
    #parameters of _pretransform), and if necessary, y (the targets)
    @abstractmethod
    def _prefit(self, X, y):
        pass
