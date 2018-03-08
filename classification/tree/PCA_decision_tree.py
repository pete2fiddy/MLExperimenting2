import numpy as np
from classification.tree.general_decision_tree import TreeNode

class PCATree(TreeNode):

    def __init__(self, num_eigvecs):
        self.__eigvecs = None
        self.__num_eigvecs = num_eigvecs

    def _init_node(self):
        return PCATree(self.__num_eigvecs)

    def _pretransform(self, X):
        return np.dot(X - self.__mean, self.__eigvecs.T)

    def _prefit(self, X, y):
        self.__mean = np.average(X, axis = 0)
        X_minus_mean = X - self.__mean
        cov = np.dot(X_minus_mean.T, X_minus_mean)/float(X.shape[0] - 1)#subtract by 1 in divisor for unbiased estimate
        _, self.__eigvecs = np.linalg.eigh(cov)
        self.__eigvecs = self.__eigvecs[::-1, :]
        self.__eigvecs = self.__eigvecs[:self.__num_eigvecs]



'''
class PCANode:


    def __init__(self):
        self.__less_child = None
        self.__more_child = None
        self.__split_val = None
        self.__axis = None
        self.__mean = None
        self.__terminal_value = None

    def predict(self, x):
        if self.__terminal_value is not None:
            return self.__terminal_value if len(x.shape) == 1 else np.full(x.shape[0], self.__terminal_value)
        dots = self.__calc_dots(x)
        less_indices = np.where(dots < self.__split_val)
        more_indices = np.where(dots >= self.__split_val)
        predictions = np.zeros(x.shape[0], dtype = np.int)
        predictions[less_indices] = self.__less_child.predict(x[less_indices])
        predictions[more_indices] = self.__more_child.predict(x[more_indices])
        return predictions

    def __calc_dots(self, x, eig = None):
        eig = self.__axis if eig is None else eig
        if len(x.shape) == 1:
            return np.array([np.dot(x-self.__mean, eig)])
        return np.dot(x-self.__mean, eig)

    def __sort_dots(self, eig, X, y):
        dots = self.__calc_dots(X, eig = eig)
        sorted_inds = np.asarray(sorted([(i, dots[i]) for i in range(len(dots))], key = lambda ind : ind[1])).astype(np.int)
        X_sorted = X[sorted_inds[:,0], :]
        y_sorted = y[sorted_inds[:,0]]
        dots_sorted = dots[sorted_inds[:,0]]
        return X_sorted, y_sorted, dots_sorted



    def fit(self, X, y, impurity_func, thresh_impurity, remaining_depth, min_to_split = 3, num_eigenaxises = 1):
        if X.shape[0] < min_to_split or remaining_depth <= 0:
            values, counts = np.unique(y, return_counts = True)
            self.__terminal_value = values[np.argmax(counts)]
            return None
        self.__set_mean(X)
        eigs = self.__ranked_eigs(X)
        best_eigenimpurity = None
        best_split_index = None
        X_sorted = None
        y_sorted = None
        dots_sorted = None

        for i in range(num_eigenaxises):
            iter_impurity, iter_split_index, iter_X_sorted, iter_y_sorted, iter_dots_sorted = self.__fit_eigenaxis(X, y, impurity_func, eigs[i])
            if best_eigenimpurity is None or iter_impurity > best_eigenimpurity:
                best_eigenimpurity = iter_impurity
                self.__axis = eigs[i]
                self.__split_val = (iter_dots_sorted[iter_split_index-1] + iter_dots_sorted[iter_split_index])/2.0
                X_sorted = iter_X_sorted
                y_sorted = iter_y_sorted
                dots_sorted = iter_dots_sorted
                best_split_index = iter_split_index
        self.__less_child = PCANode()
        self.__more_child = PCANode()
        self.__less_child.fit(X_sorted[:best_split_index], y_sorted[:best_split_index], impurity_func, thresh_impurity, remaining_depth - 1)
        self.__more_child.fit(X_sorted[best_split_index:], y_sorted[best_split_index:], impurity_func, thresh_impurity, remaining_depth - 1)


    def __fit_eigenaxis(self, X, y, impurity_func, eig):
        X_sorted, y_sorted, dots_sorted = self.__sort_dots(eig, X, y)
        best_impurity = None
        best_fork_index = None
        for i in range(1, X_sorted.shape[0]):
            iter_impurity = impurity_func([y_sorted[:i], y_sorted[i:]])
            if best_impurity is None or iter_impurity > best_impurity:
                best_impurity = iter_impurity
                best_fork_index = i
        return best_impurity, best_fork_index, X_sorted, y_sorted, dots_sorted



    def __set_mean(self, X):
        self.__mean = np.average(X, axis = 0)

    def __ranked_eigs(self, X):
        X_minus_mean = X - self.__mean
        cov = np.dot(X_minus_mean.T, X_minus_mean)/float(X.shape[0] - 1)#subtract by 1 in divisor for unbiased estimate
        vals, vecs = np.linalg.eigh(cov)
        return vecs[::-1, :]
        #self.__axis = vecs[len(vecs)-1]

'''
