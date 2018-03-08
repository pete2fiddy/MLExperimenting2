import numpy as np

def euclidian(x1, x2):
    return np.linalg.norm(x2-x1)


def gaussian_kernel(x1, x2):
    return np.exp(-np.linalg.norm(x2-x1)**2)
