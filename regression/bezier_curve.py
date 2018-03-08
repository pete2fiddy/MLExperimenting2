import numpy as np
from matplotlib import pyplot as plt
import linalg.matrix.matmath as matmath


def curve(p, t):
    A = __prodA(p.shape[0], t)
    p_fin = np.dot(p.T, A[0])
    return p_fin


def fit_gd(Y, T, n_points = 3, n_iter = 100, learn_rate = 0.03):
    '''TODO: solve in closed form if possible'''
    p = (np.random.rand(n_points, Y.shape[1])*10)-5
    As = np.zeros((T.shape[0], n_points))
    for n in range(0, T.shape[0]):
        As[n] = __prodA(p.shape[0], T[n])[0]
    for iter in range(n_iter):
        grad = np.zeros(p.shape)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for n in range(0, T.shape[0]):
                    grad[i,j] += -(Y[n,j]-curve(p, T[n])[j]) * As[n,i]
        p -= learn_rate*grad
    return p

def fit(Y, T, n_points = 3):
    A = np.zeros((T.shape[0], n_points))
    for n in range(0, A.shape[0]):
        A[n] = __prodA(n_points, T[n])[0]
    A_inv = matmath.inv(np.dot(A.T, A)).dot(A.T)#np.linalg.inv(np.dot(A.T, A)).dot(A.T)
    p = np.zeros((n_points, Y.shape[1]))
    for j in range(0, p.shape[1]):
        p[:,j] = np.dot(A_inv, Y[:,j])
    return p


def __prodA(num_points, t):
    A = np.identity(num_points)
    for n in range(num_points, 1, -1):
        A = np.dot(__A(num_points, n, t), A)
    return A


def __A(num_points, n, t):
    A1 = np.identity(num_points)
    A1[n-1:] = 0
    A2 = np.zeros((num_points, num_points))
    A2[:A2.shape[0]-1, 1:] = np.identity(A2.shape[0]-1)
    A2[n:] = 0
    return (1-t)*A1 + t*A2

def graph(p, color):
    t = 0
    X = []
    Y = []
    while t < 1.01:
        f_t = curve(p, t)
        X.append(f_t[0])
        Y.append(f_t[1])
        t += 0.001
    plt.plot(X, Y, color = color)
    plt.scatter(p[:, 0], p[:, 1], color = color)
