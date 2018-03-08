import numpy as np

#may have problems -- not sure
#returns the one of K representation of targets y, where for a given one of k vector, uniques[argmax(vec)] is the value of the original target variable
def one_of_K(y):
    assert len(y.shape) == 1 #makes sure y is a vector of class labels
    uniques, unique_indices = np.unique(y, return_inverse = True)
    one_of_K = np.zeros((y.shape[0], len(uniques)), np.int)
    for i in range(0, one_of_K.shape[0]):
        one_of_K[i, unique_indices[i]] = 1
    print("one of k: ", one_of_K)
    return one_of_K, uniques
